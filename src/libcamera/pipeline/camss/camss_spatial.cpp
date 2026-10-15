/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_spatial.cpp - Multi-scale spatial noise reduction on NV12 frames
 */

/* plain float loops over rows, for the vectoriser; see camss_denoise.cpp */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif

#include "camss_spatial.h"

#include <algorithm>
#include <cmath>

namespace libcamera {

void CamssSpatial::allocate(Pyramid &p, unsigned int w, unsigned int h)
{
	for (unsigned int l = 0; l <= kLevels; l++) {
		Plane &pl = p[l];
		pl.w = w;
		pl.h = h;
		size_t n = static_cast<size_t>(w) * h;
		pl.g.assign(n, 0.0f);
		if (l < kLevels) {
			pl.u.assign(n, 0.0f);
			pl.d.assign(n, 0.0f);
			pl.e.assign(n, 0.0f);
		}
		w /= 2;
		h /= 2;
	}
}

CamssSpatial::~CamssSpatial()
{
	stopWorkers();
}

void CamssSpatial::startWorkers()
{
	/*
	 * A new worker starts at the current generation, or it would take
	 * the last job of an earlier configuration (its function long gone)
	 * for a new one.
	 */
	std::lock_guard<std::mutex> l(lock_);
	quit_ = false;
	for (unsigned int i = 1; i < params_.threads; i++)
		workers_.emplace_back(&CamssSpatial::workerMain, this, i, generation_);
}

void CamssSpatial::stopWorkers()
{
	{
		std::lock_guard<std::mutex> l(lock_);
		quit_ = true;
	}
	wake_.notify_all();
	for (std::thread &w : workers_)
		w.join();
	workers_.clear();
}

void CamssSpatial::workerMain(unsigned int index, uint64_t start)
{
	uint64_t seen = start;

	for (;;) {
		const std::function<void(unsigned int, unsigned int)> *job;
		unsigned int rows, bands;
		{
			std::unique_lock<std::mutex> l(lock_);
			wake_.wait(l, [&] { return quit_ || generation_ != seen; });
			if (quit_)
				return;
			seen = generation_;
			job = job_;
			rows = jobRows_;
			bands = jobBands_;
		}

		if (index < bands) {
			unsigned int step = (rows + bands - 1) / bands;
			unsigned int a = index * step, b = std::min(rows, a + step);
			if (a < b)
				(*job)(a, b);
		}

		{
			std::lock_guard<std::mutex> l(lock_);
			if (--pending_ == 0)
				done_.notify_one();
		}
	}
}

void CamssSpatial::configure(const Size &size, unsigned int stride, const Params &params)
{
	stopWorkers();
	params_ = params;
	params_.threads = std::clamp(params_.threads, 1u, 8u);
	startWorkers();
	size_ = size;
	stride_ = stride;
	allocate(luma_, size.width, size.height);
	allocate(cb_, size.width / 2, size.height / 2);
	allocate(cr_, size.width / 2, size.height / 2);
	tmp_.assign(static_cast<size_t>(size.width) * size.height, 0.0f);
}

/* Rows [0, n) split into bands across the worker threads */
void CamssSpatial::parallel(unsigned int n, const std::function<void(unsigned int, unsigned int)> &fn)
{
	unsigned int bands = std::min<unsigned int>(workers_.size() + 1, std::max(n / 16, 1u));
	if (bands <= 1) {
		fn(0, n);
		return;
	}

	{
		std::lock_guard<std::mutex> l(lock_);
		job_ = &fn;
		jobRows_ = n;
		jobBands_ = bands;
		pending_ = workers_.size();
		generation_++;
	}
	wake_.notify_all();

	unsigned int step = (n + bands - 1) / bands;
	fn(0, std::min(n, step));

	std::unique_lock<std::mutex> l(lock_);
	done_.wait(l, [&] { return pending_ == 0; });
}

/*
 * Clip each sample to the range of its eight neighbours widened by a
 * margin: a lone noise spike (or a weak hot or dead pixel) stands out of
 * every neighbourhood, an edge or a line does not. The pyramid's local
 * energy test would otherwise keep a big enough spike as detail.
 */
void CamssSpatial::despeckle(Plane &pl, float margin)
{
	const unsigned int w = pl.w, h = pl.h;
	float *__restrict out = tmp_.data();

	parallel(h, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int y = r0; y < r1; y++) {
			const float *a = &pl.g[(y ? y - 1 : 0) * w];
			const float *b = &pl.g[y * w];
			const float *c = &pl.g[std::min(y + 1, h - 1) * w];
			float *o = out + static_cast<size_t>(y) * w;
			o[0] = b[0];
			o[w - 1] = b[w - 1];
			/* pairwise min and max, which the vectoriser turns into fmin/fmax */
			for (unsigned int x = 1; x + 1 < w; x++) {
				float lo = std::min(std::min(std::min(a[x - 1], a[x]), std::min(a[x + 1], b[x - 1])),
						    std::min(std::min(b[x + 1], c[x - 1]), std::min(c[x], c[x + 1])));
				float hi = std::max(std::max(std::max(a[x - 1], a[x]), std::max(a[x + 1], b[x - 1])),
						    std::max(std::max(b[x + 1], c[x - 1]), std::max(c[x], c[x + 1])));
				o[x] = std::min(std::max(b[x], lo - margin), hi + margin);
			}
		}
	});
	std::copy_n(out, static_cast<size_t>(w) * h, pl.g.begin());
}

/* 2x2 mean; odd last rows or columns are dropped */
void CamssSpatial::down(const Plane &src, Plane &dst)
{
	parallel(dst.h, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int y = r0; y < r1; y++) {
			const float *__restrict a = &src.g[2 * y * src.w];
			const float *__restrict b = a + src.w;
			float *__restrict o = &dst.g[y * dst.w];
			for (unsigned int x = 0; x < dst.w; x++)
				o[x] = 0.25f * (a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1]);
		}
	});
}

/*
 * Bilinear 2x to dst's size (into dst.u), sample centres aligned with the
 * 2x2 mean: each output takes 3/4 of the nearer source sample and 1/4 of
 * the next one out, in each direction, edges replicated.
 */
void CamssSpatial::up(const Plane &src, Plane &dst, std::vector<float> &tmp)
{
	const unsigned int sw = src.w, sh = src.h, dw = dst.w, dh = dst.h;
	float *__restrict h = tmp.data();	/* sh rows of dw */

	parallel(sh, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int y = r0; y < r1; y++) {
			const float *s = &src.g[y * sw];
			float *o = h + static_cast<size_t>(y) * dw;
			for (unsigned int x = 0; x < dw; x++) {
				unsigned int c0 = std::min(x >> 1, sw - 1);
				unsigned int c1 = (x & 1) ? std::min(c0 + 1, sw - 1) : (c0 ? c0 - 1 : 0);
				o[x] = 0.75f * s[c0] + 0.25f * s[c1];
			}
		}
	});

	parallel(dh, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int y = r0; y < r1; y++) {
			unsigned int s0 = std::min(y >> 1, sh - 1);
			unsigned int s1 = (y & 1) ? std::min(s0 + 1, sh - 1) : (s0 ? s0 - 1 : 0);
			const float *__restrict a = h + static_cast<size_t>(s0) * dw;
			const float *__restrict b = h + static_cast<size_t>(s1) * dw;
			float *__restrict o = &dst.u[static_cast<size_t>(y) * dw];
			for (unsigned int x = 0; x < dw; x++)
				o[x] = 0.75f * a[x] + 0.25f * b[x];
		}
	});
}

/* sigma = 1.4826 median |finest detail|, from a sample of the plane */
double CamssSpatial::estimateNoise(const Pyramid &p)
{
	constexpr unsigned int bins = 512;	/* 1/8 level each */
	std::array<uint32_t, bins> hist{};
	uint32_t n = 0;
	const Plane &pl = p[0];

	for (unsigned int y = 1; y < pl.h; y += 3) {
		const float *d = &pl.d[y * pl.w];
		for (unsigned int x = 1; x < pl.w; x += 3) {
			hist[std::min<unsigned int>(std::abs(d[x]) * 8.0f, bins - 1)]++;
			n++;
		}
	}

	uint32_t sum = 0;
	unsigned int m = 0;
	for (; m < bins - 1; m++) {
		sum += hist[m];
		if (sum >= n / 2)
			break;
	}
	return 1.4826 * (m + 0.5) / 8.0;
}

/*
 * Rebuild from the coarsest level: out_l = up(out_l+1) + detail_l * gain,
 * the gain a local Wiener factor 1 - t^2 / E over the 3x3 mean detail
 * energy E, t the level's strength times its noise.
 */
void CamssSpatial::shrink(Pyramid &p, const std::array<double, kLevels> &k,
			  const std::array<double, kLevels> &ratios, double sigma0,
			  const std::array<double, kLevels> &sharpen)
{
	for (int l = kLevels - 1; l >= 0; l--) {
		Plane &pl = p[l];
		const unsigned int w = pl.w, h = pl.h;
		const float t = k[l] * ratios[l] * sigma0;
		const float t2 = t * t;
		const float boost = sharpen[l];

		/* up(out of the coarser level); the coarsest level's out is its g */
		Plane &coarse = p[l + 1];
		if (l + 1 < static_cast<int>(kLevels)) {
			/* the coarser level's output sits in its u; up() reads g */
			std::swap(coarse.g, coarse.u);
			up(coarse, pl, tmp_);
			std::swap(coarse.g, coarse.u);
		} else {
			up(coarse, pl, tmp_);
		}

		if (t2 <= 0.0f && boost == 0.0f) {
			parallel(h, [&](unsigned int r0, unsigned int r1) {
				for (unsigned int y = r0; y < r1; y++) {
					float *__restrict o = &pl.u[y * w];
					const float *__restrict d = &pl.d[y * w];
					for (unsigned int x = 0; x < w; x++)
						o[x] += d[x];
				}
			});
			continue;
		}

		/* horizontal 3-tap sum of d^2 into e */
		parallel(h, [&](unsigned int r0, unsigned int r1) {
			for (unsigned int y = r0; y < r1; y++) {
				const float *__restrict d = &pl.d[y * w];
				float *__restrict e = &pl.e[y * w];
				e[0] = 2 * d[0] * d[0] + d[1] * d[1];
				for (unsigned int x = 1; x + 1 < w; x++)
					e[x] = d[x - 1] * d[x - 1] + d[x] * d[x] + d[x + 1] * d[x + 1];
				e[w - 1] = d[w - 2] * d[w - 2] + 2 * d[w - 1] * d[w - 1];
			}
		});

		/* vertical 3-tap, the gain, and the sum into u */
		parallel(h, [&](unsigned int r0, unsigned int r1) {
			for (unsigned int y = r0; y < r1; y++) {
				const float *__restrict ea = &pl.e[(y ? y - 1 : 0) * w];
				const float *__restrict eb = &pl.e[y * w];
				const float *__restrict ec = &pl.e[std::min(y + 1, h - 1) * w];
				const float *__restrict d = &pl.d[y * w];
				float *__restrict o = &pl.u[y * w];
				for (unsigned int x = 0; x < w; x++) {
					float e = (ea[x] + eb[x] + ec[x]) * (1.0f / 9.0f);
					float g = std::max(1.0f - t2 / std::max(e, 1e-6f), 0.0f);
					/* g^2 keeps the boost off anything near the noise */
					o[x] += d[x] * g * (1.0f + boost * g * g);
				}
			}
		});
	}
}

void CamssSpatial::process(uint8_t *y, uint8_t *uv)
{
	const unsigned int w = size_.width, h = size_.height;

	/* planes in, as float */
	parallel(h, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int r = r0; r < r1; r++) {
			const uint8_t *__restrict s = y + r * stride_;
			float *__restrict o = &luma_[0].g[r * w];
			for (unsigned int x = 0; x < w; x++)
				o[x] = s[x];
		}
	});
	parallel(h / 2, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int r = r0; r < r1; r++) {
			const uint8_t *__restrict s = uv + r * stride_;
			float *__restrict u = &cb_[0].g[r * (w / 2)];
			float *__restrict v = &cr_[0].g[r * (w / 2)];
			for (unsigned int x = 0; x < w / 2; x++) {
				u[x] = s[2 * x];
				v[x] = s[2 * x + 1];
			}
		}
	});

	/* the margin follows the last frame's noise; the first frame gets a wide one */
	if (params_.despeckle > 0.0)
		despeckle(luma_[0], params_.despeckle * (lumaSigma_ > 0.0 ? lumaSigma_ : 8.0));

	/* Gaussian levels, then each level's detail against the next one up */
	for (Pyramid *p : { &luma_, &cb_, &cr_ }) {
		for (unsigned int l = 0; l < kLevels; l++)
			down((*p)[l], (*p)[l + 1]);
		for (unsigned int l = 0; l < kLevels; l++) {
			Plane &pl = (*p)[l];
			up((*p)[l + 1], pl, tmp_);
			parallel(pl.h, [&](unsigned int r0, unsigned int r1) {
				for (unsigned int r = r0; r < r1; r++) {
					const float *__restrict g = &pl.g[r * pl.w];
					const float *__restrict u = &pl.u[r * pl.w];
					float *__restrict d = &pl.d[r * pl.w];
					for (unsigned int x = 0; x < pl.w; x++)
						d[x] = g[x] - u[x];
				}
			});
		}
	}

	lumaSigma_ = estimateNoise(luma_);
	chromaSigma_ = 0.5 * (estimateNoise(cb_) + estimateNoise(cr_));

	/* the coarsest level is its own output */
	for (Pyramid *p : { &luma_, &cb_, &cr_ })
		(*p)[kLevels].u = (*p)[kLevels].g;

	static const std::array<double, kLevels> none{};
	shrink(luma_, params_.luma, params_.lumaRatios, lumaSigma_, params_.sharpen);
	shrink(cb_, params_.chroma, params_.chromaRatios, chromaSigma_, none);
	shrink(cr_, params_.chroma, params_.chromaRatios, chromaSigma_, none);

	/* planes out, rounded and clamped */
	parallel(h, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int r = r0; r < r1; r++) {
			const float *__restrict s = &luma_[0].u[r * w];
			uint8_t *__restrict o = y + r * stride_;
			for (unsigned int x = 0; x < w; x++)
				o[x] = static_cast<uint8_t>(std::clamp(s[x] + 0.5f, 0.0f, 255.0f));
		}
	});
	parallel(h / 2, [&](unsigned int r0, unsigned int r1) {
		for (unsigned int r = r0; r < r1; r++) {
			const float *__restrict u = &cb_[0].u[r * (w / 2)];
			const float *__restrict v = &cr_[0].u[r * (w / 2)];
			uint8_t *__restrict o = uv + r * stride_;
			for (unsigned int x = 0; x < w / 2; x++) {
				o[2 * x] = static_cast<uint8_t>(std::clamp(u[x] + 0.5f, 0.0f, 255.0f));
				o[2 * x + 1] = static_cast<uint8_t>(std::clamp(v[x] + 0.5f, 0.0f, 255.0f));
			}
		}
	});
}

} /* namespace libcamera */
