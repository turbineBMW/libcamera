/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_denoise.cpp - Temporal and chroma noise reduction on NV12 frames
 */

/*
 * The per-pixel loops run on every frame on the pipeline thread; the
 * vectoriser needs O3 for most of them (4.1 ms -> 2.7 ms at 1080p).
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif

#include "camss_denoise.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace libcamera {

void CamssDenoise::configure(const Size &size, unsigned int stride, const Params &params)
{
	params_ = params;
	size_ = size;
	stride_ = stride;
	bw_ = size.width / 2;
	bh_ = size.height / 2;
	accY_.assign(size.width * size.height, 0);
	accUV_.assign(size.width * bh_, 0);
	dLuma_.assign(bw_ * bh_, 0);
	dChroma_.assign(bw_ * bh_, 0);
	dRegion_.assign(bw_ * bh_, 0);
	blockW_.assign(bw_ * bh_, 0);
	tmpW_.assign(bw_ * bh_, 0);
	colSums_.assign(bw_, 0);
	pixW_.assign(size.width, 0);
	/* the blur's row of sums carries radius pairs of edge padding each side */
	rowSums_.assign(size.width + 4 * params.chromaRadius, 0);
	boxSums_.assign(size.width, 0);
	chromaSrc_.assign(size.width * bh_, 0);
	primed_ = false;

	double minW = std::clamp(params_.minWeight, 1.0 / 32, 1.0);
	double ramp = std::max(params_.ramp, 0.0625);
	for (unsigned int i = 0; i < weights_.size(); i++) {
		double k = std::clamp((i / 16.0 - params_.threshold) / ramp, 0.0, 1.0);
		weights_[i] = static_cast<uint16_t>(std::lround((minW + (1.0 - minW) * k) * 256));
	}
}

/*
 * One row moved towards the new frame by each pixel's weight (Q8), the
 * accumulator kept in Q4. Plain loops over arrays, for the vectoriser.
 */
static void blendRow(uint8_t *__restrict p, uint16_t *__restrict acc,
		     const uint16_t *__restrict wt, unsigned int n)
{
	for (unsigned int x = 0; x < n; x++) {
		int32_t a = acc[x];
		a += (((p[x] << 4) - a) * wt[x] + 128) >> 8;
		acc[x] = a;
		p[x] = (a + 8) >> 4;
	}
}

/* Each block's luma and chroma change against the accumulator */
void CamssDenoise::measure(const uint8_t *y, const uint8_t *uv)
{
	const unsigned int w = size_.width;

	for (unsigned int by = 0; by < bh_; by++) {
		const uint8_t *__restrict y0 = y + 2 * by * stride_;
		const uint8_t *__restrict y1 = y0 + stride_;
		const uint8_t *__restrict c = uv + by * stride_;
		const uint16_t *__restrict a0 = &accY_[2 * by * w];
		const uint16_t *__restrict a1 = a0 + w;
		const uint16_t *__restrict ac = &accUV_[by * w];
		int16_t *__restrict dl = &dLuma_[by * bw_];
		uint16_t *__restrict dc = &dChroma_[by * bw_];

		for (unsigned int i = 0; i < bw_; i++) {
			unsigned int x = 2 * i;
			int32_t dy = ((y0[x] + y0[x + 1] + y1[x] + y1[x + 1]) << 4) -
				     (a0[x] + a0[x + 1] + a1[x] + a1[x + 1]);
			int32_t du = std::abs((c[x] << 4) - ac[x]);
			int32_t dv = std::abs((c[x + 1] << 4) - ac[x + 1]);
			dl[i] = dy;
			dc[i] = std::max(du, dv);
		}
	}
}

/* |mean luma change| over the 5x5 blocks around each block, edges replicated */
void CamssDenoise::regional()
{
	constexpr int r = 2;
	const int bw = bw_, bh = bh_;
	int32_t *__restrict cs = colSums_.data();

	std::fill(colSums_.begin(), colSums_.end(), 0);
	for (int i = -r; i <= r; i++) {
		const int16_t *row = &dLuma_[std::clamp(i, 0, bh - 1) * bw];
		for (int x = 0; x < bw; x++)
			cs[x] += row[x];
	}

	for (int by = 0; by < bh; by++) {
		uint16_t *__restrict out = &dRegion_[by * bw];
		int32_t sum = 0;
		for (int t = -r; t <= r; t++)
			sum += cs[std::clamp(t, 0, bw - 1)];
		for (int x = 0; x < bw; x++) {
			out[x] = std::min(std::abs(sum) / 25, 65535);
			sum += cs[std::min(x + r + 1, bw - 1)] - cs[std::max(x - r, 0)];
		}

		const int16_t *add = &dLuma_[std::min(by + r + 1, bh - 1) * bw];
		const int16_t *sub = &dLuma_[std::max(by - r, 0) * bw];
		for (int x = 0; x < bw; x++)
			cs[x] += add[x] - sub[x];
	}
}

/*
 * The noise of each measure from the median of its magnitude over a
 * sample of the blocks: sigma = 1.4826 median for a still picture.
 *
 * When the whole picture moves (the tablet in a hand, a pan) the median
 * is movement, not noise, and taking it would call everything still and
 * smear the frames together. So the estimates fall freely but rise by
 * at most 2% a frame: movement cannot inflate them, while a real rise in
 * noise as the light fades is followed within a second or two.
 */
void CamssDenoise::estimateNoise()
{
	std::array<uint32_t, 256> hb{}, hr{}, hc{};
	uint32_t n = 0;

	for (unsigned int by = 0; by < bh_; by += 2) {
		const int16_t *dl = &dLuma_[by * bw_];
		const uint16_t *dr = &dRegion_[by * bw_];
		const uint16_t *dc = &dChroma_[by * bw_];
		for (unsigned int i = 0; i < bw_; i += 2) {
			hb[std::min(std::abs(dl[i]) >> 4, 255)]++;	/* 1/4 level */
			hr[std::min(dr[i] >> 2, 255)]++;		/* 1/16 level */
			hc[std::min(dc[i] >> 2, 255)]++;		/* 1/4 level */
			n++;
		}
	}

	auto median = [n](const std::array<uint32_t, 256> &h) {
		uint32_t sum = 0;
		unsigned int m = 0;
		for (; m < h.size() - 1; m++) {
			sum += h[m];
			if (sum >= n / 2)
				break;
		}
		return 1.4826 * (m + 0.5);
	};

	auto track = [this](double &sigma, double measured) {
		if (!tracked_ || measured <= sigma)
			sigma = measured;
		else
			sigma = std::min(measured, sigma * 1.02);
	};
	track(sigmaBlock_, median(hb) * 16);
	track(sigmaRegion_, median(hr) * 4);
	track(sigmaChroma_, median(hc) * 4);
	tracked_ = true;
}

/* Score each block in sigmas, look up its weight, and spread the weights 3x3 */
void CamssDenoise::blockWeights()
{
	const float fb = 16.0f / sigmaBlock_;
	const float fr = 16.0f / sigmaRegion_;
	const float fc = 16.0f / sigmaChroma_;
	const unsigned int nb = bw_ * bh_;
	const int16_t *__restrict dl = dLuma_.data();
	const uint16_t *__restrict dr = dRegion_.data();
	const uint16_t *__restrict dc = dChroma_.data();
	uint16_t *__restrict wb = blockW_.data();
	uint16_t *__restrict tmp = tmpW_.data();

	for (unsigned int b = 0; b < nb; b++) {
		float s = std::max({ std::abs(dl[b]) * fb, dr[b] * fr, dc[b] * fc });
		wb[b] = weights_[static_cast<unsigned int>(std::min(s, 255.0f))];
	}

	for (unsigned int by = 0; by < bh_; by++) {
		const uint16_t *in = wb + by * bw_;
		uint16_t *out = tmp + by * bw_;
		out[0] = std::max(in[0], in[1]);
		for (unsigned int i = 1; i + 1 < bw_; i++)
			out[i] = std::max({ in[i - 1], in[i], in[i + 1] });
		out[bw_ - 1] = std::max(in[bw_ - 2], in[bw_ - 1]);
	}
	for (unsigned int by = 0; by < bh_; by++) {
		const uint16_t *up = tmp + (by ? by - 1 : 0) * bw_;
		const uint16_t *mid = tmp + by * bw_;
		const uint16_t *down = tmp + std::min(by + 1, bh_ - 1) * bw_;
		uint16_t *out = wb + by * bw_;
		for (unsigned int i = 0; i < bw_; i++)
			out[i] = std::max({ up[i], mid[i], down[i] });
	}
}

void CamssDenoise::process(uint8_t *y, uint8_t *uv)
{
	const unsigned int w = size_.width, h = size_.height;

	if (!primed_) {
		for (unsigned int r = 0; r < h; r++)
			for (unsigned int x = 0; x < w; x++)
				accY_[r * w + x] = y[r * stride_ + x] << 4;
		for (unsigned int r = 0; r < bh_; r++)
			for (unsigned int x = 0; x < w; x++)
				accUV_[r * w + x] = uv[r * stride_ + x] << 4;
		primed_ = true;
		if (params_.chromaRadius)
			blurChroma(uv);
		return;
	}

	measure(y, uv);
	regional();
	estimateNoise();
	blockWeights();

	/* the block's weight serves both its luma rows and the interleaved chroma */
	uint16_t *__restrict pw = pixW_.data();
	for (unsigned int by = 0; by < bh_; by++) {
		const uint16_t *wb = &blockW_[by * bw_];
		for (unsigned int i = 0; i < bw_; i++)
			pw[2 * i] = pw[2 * i + 1] = wb[i];

		blendRow(y + 2 * by * stride_, &accY_[2 * by * w], pw, w);
		blendRow(y + (2 * by + 1) * stride_, &accY_[(2 * by + 1) * w], pw, w);
		blendRow(uv + by * stride_, &accUV_[by * w], pw, w);
	}

	if (params_.chromaRadius)
		blurChroma(uv);
}

/*
 * Box blur of the interleaved chroma plane, separable: vertical running
 * sums into a padded row, then the horizontal taps summed at a stride of
 * two so U and V stay apart, and a reciprocal multiply for the mean.
 */
void CamssDenoise::blurChroma(uint8_t *uv)
{
	const unsigned int r = params_.chromaRadius;
	const unsigned int w = size_.width;
	const unsigned int rows = size_.height / 2;
	const unsigned int k = 2 * r + 1;
	const uint32_t recip = (65536 + k * k / 2) / (k * k);
	const unsigned int pad = 2 * r;
	uint8_t *__restrict src = chromaSrc_.data();
	uint16_t *__restrict rs = rowSums_.data();
	uint32_t *__restrict box = boxSums_.data();

	for (unsigned int row = 0; row < rows; row++)
		std::copy_n(uv + row * stride_, w, src + row * w);

	std::fill(rowSums_.begin(), rowSums_.end(), 0);
	for (int i = -static_cast<int>(r); i <= static_cast<int>(r); i++) {
		const uint8_t *s = src + std::clamp<int>(i, 0, rows - 1) * w;
		for (unsigned int x = 0; x < w; x++)
			rs[pad + x] += s[x];
	}

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int i = 0; i < pad; i++) {
			rs[i] = rs[pad + (i & 1)];
			rs[pad + w + i] = rs[pad + w - 2 + (i & 1)];
		}

		std::fill(boxSums_.begin(), boxSums_.end(), 0);
		for (unsigned int t = 0; t < k; t++) {
			const uint16_t *tap = rs + 2 * t;
			for (unsigned int x = 0; x < w; x++)
				box[x] += tap[x];
		}

		uint8_t *out = uv + row * stride_;
		for (unsigned int x = 0; x < w; x++)
			out[x] = (box[x] * recip + 32768) >> 16;

		const uint8_t *add = src + std::min(row + r + 1, rows - 1) * w;
		const uint8_t *sub = src + (row >= r ? row - r : 0) * w;
		for (unsigned int x = 0; x < w; x++)
			rs[pad + x] += add[x] - sub[x];
	}
}

} /* namespace libcamera */
