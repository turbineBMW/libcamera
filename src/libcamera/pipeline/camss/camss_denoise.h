/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_denoise.h - Temporal and chroma noise reduction on NV12 frames
 */

#pragma once

#include <array>
#include <stdint.h>
#include <vector>

#include <libcamera/geometry.h>

namespace libcamera {

/*
 * A recursive temporal filter on NV12, in place: each 2x2 block of luma
 * and its chroma sample move towards the new frame by a weight that grows
 * with how far the picture there has moved beyond the noise, so still
 * parts average over several frames and moving parts follow at once.
 *
 * Movement is judged three ways, each against its own noise level taken
 * from the frame (the median, most of a picture being still): the block's
 * own mean luma change, the mean change over the 5x5 blocks around it
 * (ten times quieter, so it catches a moderate change over an area that a
 * single block cannot tell from noise), and the chroma change. A block
 * near a moving one counts as moving too, so moving areas switch over
 * whole rather than in speckles. A box blur on the output chroma removes
 * the coloured speckle that remains.
 */
class CamssDenoise
{
public:
	struct Params {
		bool enabled = true;
		double minWeight = 0.2;		/* weight of a still frame */
		double threshold = 2.5;		/* noise sigmas before a block counts as moving */
		double ramp = 2.0;		/* further sigmas to full weight */
		unsigned int chromaRadius = 2;	/* box blur radius on the chroma plane, 0 off */
	};

	void configure(const Size &size, unsigned int stride, const Params &params);
	void reset() { primed_ = false; tracked_ = false; }
	bool enabled() const { return params_.enabled; }
	void process(uint8_t *y, uint8_t *uv);
	double sigma() const { return sigmaBlock_ / 64.0; }	/* levels, block mean */

private:
	void measure(const uint8_t *y, const uint8_t *uv);
	void regional();
	void estimateNoise();
	void blockWeights();
	void blurChroma(uint8_t *uv);

	Params params_;
	Size size_;
	unsigned int stride_ = 0;
	unsigned int bw_ = 0, bh_ = 0;	/* blocks */
	bool primed_ = false;

	std::vector<uint16_t> accY_;	/* Q4 */
	std::vector<uint16_t> accUV_;	/* Q4 */

	/* per block: luma change (sum of four Q4 differences, 1/64 level of the mean) */
	std::vector<int16_t> dLuma_;
	std::vector<uint16_t> dChroma_;	/* larger of the U and V changes, Q4 */
	std::vector<uint16_t> dRegion_;	/* mean |change| over 5x5 blocks, 1/64 level */
	std::vector<uint16_t> blockW_;	/* Q8 weights, then dilated */
	std::vector<uint16_t> tmpW_;
	std::vector<int32_t> colSums_;
	std::vector<uint16_t> pixW_;

	double sigmaBlock_ = 256.0, sigmaRegion_ = 64.0, sigmaChroma_ = 64.0;
	bool tracked_ = false;	/* the sigmas hold a measurement */
	std::array<uint16_t, 256> weights_;	/* Q8, by score in sixteenths of a sigma */

	/* chroma blur */
	std::vector<uint16_t> rowSums_;
	std::vector<uint32_t> boxSums_;
	std::vector<uint8_t> chromaSrc_;
};

} /* namespace libcamera */
