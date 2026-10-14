/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_isp_inline.h - Inline IFE ISP: NV12 straight from the pixel path
 */

#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "libcamera/internal/v4l2_subdevice.h"
#include "libcamera/internal/v4l2_videodevice.h"

#include "camss_denoise.h"
#include "camss_isp.h"

namespace libcamera {

class CamssCsiCamera;
class CamssFrames;
class CameraSensor;
class ControlInfoMap;
class MappedFrameBuffer;

/*
 * The CAMSS driver's PIX line: the CSID's pixel port feeds the IFE's image
 * pipeline, which writes NV12 to the msm_vfe%d_pix video node. The driver
 * programs the modules; this class owns the video node and the PIX subdev's
 * controls (white balance gains, digital gain, black level) and runs auto
 * exposure and white balance on statistics taken from the output frames.
 */
class CamssIspInline : public CamssIsp
{
public:
	static std::unique_ptr<CamssIspInline> match(CamssCsiCamera *csi,
						     CamssFrames *frameInfos,
						     ControlInfoMap *ispControls);

	CamssIspInline(CamssCsiCamera *csi, CamssFrames *frameInfos);
	~CamssIspInline() override;

	bool isValid() override { return valid_; }
	bool isInline() const override { return true; }
	Size getMargins(PixelFormat inputFormat) override;
	StreamConfiguration generateConfiguration(const StreamConfiguration &raw) const override;
	StreamConfiguration validate(const StreamConfiguration &raw, const StreamConfiguration &req) const override;
	int configure(const StreamConfiguration &inputCfg,
		      const StreamConfiguration &outputCfg) override;
	int allocateBuffers(unsigned int bufferCount) override;
	void freeBuffers() override;
	int exportOutputBuffers(const Stream *stream, unsigned int count,
				std::vector<std::unique_ptr<FrameBuffer>> *buffers) override;
	void queueBuffers(Request *request, FrameBuffer *input) override;
	int queueRequest(Request *request) override;
	void processStats(const uint32_t frame, const uint32_t statsBufferId,
			  const ControlList &sensorControls) override;
	int start() override;
	void stop() override;

private:
	struct Exposure {
		int32_t exposure;
		int32_t gain;
		int32_t vblank;
		int32_t digital;	/* Q10 */
	};

	void loadTuning();
	void bufferReady(FrameBuffer *buffer);
	void runAlgorithms(FrameBuffer *buffer, unsigned int frame);
	MappedFrameBuffer *mapBuffer(FrameBuffer *buffer);
	bool framePlanes(FrameBuffer *buffer, uint8_t **y, uint8_t **c);
	void denoise(FrameBuffer *buffer, unsigned int frame);
	void fillMetadata(ControlList &metadata) const;
	void publishControls(ControlInfoMap *ispControls) const;
	void updateLineDuration();
	int32_t manualExposureMax() const;
	int32_t digitalMax() const;
	void applySensor();
	void applyIsp();
	void applyTables();

	CamssCsiCamera *csi_;
	const CameraSensor *sensor_;
	V4L2Subdevice *pix_;
	V4L2VideoDevice *output_;
	CamssFrames *frameInfos_;
	bool valid_;

	Size size_;
	unsigned int stride_;
	unsigned int bufferCount_;

	/* sensor control ranges */
	int32_t exposureMax_;
	int32_t gainMax_;
	int32_t vblankMin_;
	int32_t vblankMax_;
	int32_t vblankDef_;
	int32_t exposureMargin_;

	Exposure exposure_;
	double lineDurationUs_;
	bool aeEnabled_;
	bool awbEnabled_;
	int32_t wbRed_;		/* Q10 */
	int32_t wbBlue_;
	int32_t channelGainMax_;	/* Q10, the IFE's limit on each channel's gain */
	int32_t blackLevel_;	/* 10-bit units */
	unsigned int frameCount_;

	/* tuning: exposure target, and the tables the PIX line takes */
	double targetLuma_;
	double zoom_;		/* field of view narrowed by this factor */
	double centreWeight_;
	double slowFrame_;	/* longest frame in the dark, in default frame lengths */
	double awbRedBias_;	/* multipliers on the grey-world gains */
	double awbBlueBias_;
	std::optional<std::array<uint16_t, 256>> gammaLut_;
	std::optional<std::array<int32_t, 12>> ccm_;
	std::optional<std::vector<uint16_t>> lscMesh_;
	CamssDenoise::Params denoiseParams_;

	CamssDenoise denoise_;
	std::map<FrameBuffer *, std::unique_ptr<MappedFrameBuffer>> maps_;
};

} /* namespace libcamera */
