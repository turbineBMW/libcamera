/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_spatial.h - Multi-scale spatial noise reduction on NV12 frames
 */

#pragma once

#include <array>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <stdint.h>
#include <vector>

#include <libcamera/geometry.h>

namespace libcamera {

/*
 * Laplacian pyramid shrinkage, in place, on the luma plane and on each
 * chroma plane: every level's detail is kept in proportion to how far its
 * local energy (3x3) stands above that level's noise, so edges and texture
 * stay while flat noise goes, at four scales. The noise at the finest
 * level is measured on each frame (median absolute detail); the coarser
 * levels follow from fixed ratios, a property of the pipeline's noise
 * spectrum measured on this sensor.
 */
class CamssSpatial
{
public:
	static constexpr unsigned int kLevels = 4;

	struct Params {
		bool enabled = true;
		/* strength per level, finest first, in noise sigmas */
		std::array<double, kLevels> luma = { 1.4, 1.2, 0.8, 0.5 };
		std::array<double, kLevels> chroma = { 2.0, 2.0, 2.0, 2.0 };
		/* noise of each level relative to the finest */
		std::array<double, kLevels> lumaRatios = { 1.0, 0.82, 0.53, 0.30 };
		std::array<double, kLevels> chromaRatios = { 1.0, 0.76, 0.43, 0.24 };
		unsigned int threads = 4;
	};

	~CamssSpatial();

	void configure(const Size &size, unsigned int stride, const Params &params);
	bool enabled() const { return params_.enabled; }
	void process(uint8_t *y, uint8_t *uv);
	double lumaSigma() const { return lumaSigma_; }
	double chromaSigma() const { return chromaSigma_; }

private:
	struct Plane {
		unsigned int w, h;
		std::vector<float> g;	/* Gaussian level */
		std::vector<float> u;	/* upsampled coarser level, then this level's output */
		std::vector<float> d;	/* detail */
		std::vector<float> e;	/* local energy */
	};
	using Pyramid = std::array<Plane, kLevels + 1>;

	void allocate(Pyramid &p, unsigned int w, unsigned int h);
	void parallel(unsigned int n, const std::function<void(unsigned int, unsigned int)> &fn);
	void down(const Plane &src, Plane &dst);
	void up(const Plane &src, Plane &dst, std::vector<float> &tmp);
	double estimateNoise(const Pyramid &p);
	void shrink(Pyramid &p, const std::array<double, kLevels> &k,
		    const std::array<double, kLevels> &ratios, double sigma0);

	void startWorkers();
	void stopWorkers();
	void workerMain(unsigned int index);

	/* a small pool: parallel() hands every worker one band of the rows */
	std::vector<std::thread> workers_;
	std::mutex lock_;
	std::condition_variable wake_, done_;
	const std::function<void(unsigned int, unsigned int)> *job_ = nullptr;
	unsigned int jobRows_ = 0, jobBands_ = 0;
	uint64_t generation_ = 0;
	unsigned int pending_ = 0;
	bool quit_ = false;

	Params params_;
	Size size_;
	unsigned int stride_ = 0;
	Pyramid luma_, cb_, cr_;
	std::vector<float> tmp_;
	double lumaSigma_ = 0.0, chromaSigma_ = 0.0;
};

} /* namespace libcamera */
