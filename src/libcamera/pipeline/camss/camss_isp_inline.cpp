/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_isp_inline.cpp - Inline IFE ISP: NV12 straight from the pixel path
 */

#include "camss_isp_inline.h"

#include <algorithm>
#include <cmath>

#include <linux/media-bus-format.h>
#include <linux/v4l2-controls.h>

#include <libcamera/base/log.h>

#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include "libcamera/internal/bayer_format.h"
#include "libcamera/internal/camera_sensor.h"
#include "libcamera/internal/framebuffer.h"
#include "libcamera/internal/request.h"
#include "libcamera/internal/mapped_framebuffer.h"
#include "libcamera/internal/media_device.h"

#include "camss_csi.h"
#include "camss_frames.h"
#include "camss_util.h"

namespace libcamera {

LOG_DECLARE_CATEGORY(Camss)

namespace {

/* The PIX line's white balance gains and black level, as the driver names them */
constexpr int32_t kWbUnity = 1024;
constexpr int32_t kWbMax = 4096;
constexpr int32_t kDigitalMax = 4096;
/* Loop tuning: target luma, damping, and where to trade frame rate for gain */
constexpr double kTargetLuma = 110.0;
constexpr int32_t kGainKnee = 768;	/* IMX681 code for 4x */
constexpr double kSlowFrame = 2.0;	/* allow frames up to twice as long in the dark */
constexpr unsigned int kStatsStep = 16;
constexpr int32_t kDefaultBlackLevel = 64;

} /* namespace */

std::unique_ptr<CamssIspInline> CamssIspInline::match(CamssCsiCamera *csi,
						      CamssFrames *frameInfos,
						      [[maybe_unused]] ControlInfoMap *ispControls)
{
	if (!csi->hasPix())
		return nullptr;

	/*
	 * The IFE takes its input as RGGB, and the CSID crop turns GBRG into
	 * it; GRBG and BGGR sensors keep the software path until the order
	 * can be set.
	 */
	bool supported = false;
	for (unsigned int code : csi->sensor()->mbusCodes()) {
		const BayerFormat &bf = BayerFormat::fromMbusCode(code);
		if (bf.isValid() && (bf.order == BayerFormat::RGGB || bf.order == BayerFormat::GBRG))
			supported = true;
	}
	if (!supported) {
		LOG(Camss, Info) << "Pixel path not used for " << csi->sensor()->id()
				 << ": Bayer order not handled yet";
		return nullptr;
	}

	auto isp = std::make_unique<CamssIspInline>(csi, frameInfos);
	if (!isp->isValid())
		return nullptr;

	LOG(Camss, Info) << "Using the inline IFE pixel path for "
			 << csi->sensor()->id();
	return isp;
}

CamssIspInline::CamssIspInline(CamssCsiCamera *csi, CamssFrames *frameInfos)
	: csi_(csi), sensor_(csi->sensor()), pix_(csi->pixSubdev()),
	  output_(csi->pixOutput()), frameInfos_(frameInfos), valid_(false),
	  stride_(0), bufferCount_(0), frameCount_(0)
{
	const ControlInfoMap &ctrls = sensor_->controls();
	auto exposure = ctrls.find(V4L2_CID_EXPOSURE);
	auto gain = ctrls.find(V4L2_CID_ANALOGUE_GAIN);
	auto vblank = ctrls.find(V4L2_CID_VBLANK);
	if (exposure == ctrls.end() || gain == ctrls.end() || vblank == ctrls.end()) {
		LOG(Camss, Error) << "Sensor lacks exposure, gain or vblank controls";
		return;
	}

	exposureMax_ = exposure->second.max().get<int32_t>();
	gainMax_ = gain->second.max().get<int32_t>();
	vblankMin_ = vblank->second.min().get<int32_t>();
	vblankMax_ = vblank->second.max().get<int32_t>();
	vblankDef_ = vblank->second.def().get<int32_t>();
	exposureMargin_ = 0;

	exposure_ = { exposure->second.def().get<int32_t>(),
		      gain->second.def().get<int32_t>(), vblankDef_, kWbUnity };
	wbRed_ = kWbUnity;
	wbBlue_ = kWbUnity;
	blackLevel_ = kDefaultBlackLevel;

	output_->bufferReady.connect(this, &CamssIspInline::bufferReady);
	valid_ = true;
}

CamssIspInline::~CamssIspInline() = default;

Size CamssIspInline::getMargins([[maybe_unused]] PixelFormat inputFormat)
{
	return { 0, 0 };
}

StreamConfiguration CamssIspInline::generateConfiguration(const StreamConfiguration &raw) const
{
	std::map<PixelFormat, std::vector<SizeRange>> formats;
	formats[formats::NV12] = { SizeRange(raw.size) };

	StreamConfiguration cfg{ StreamFormats{ formats } };
	cfg.size = raw.size;
	cfg.pixelFormat = formats::NV12;
	cfg.bufferCount = kBufferCount;

	return cfg;
}

StreamConfiguration CamssIspInline::validate(const StreamConfiguration &raw,
					     const StreamConfiguration &req) const
{
	StreamConfiguration cfg;

	/* The IFE's output size is the sensor's; no scaler is exposed yet. */
	cfg.pixelFormat = formats::NV12;
	cfg.size = raw.size;

	V4L2DeviceFormat fmt;
	fmt.fourcc = output_->toV4L2PixelFormat(cfg.pixelFormat);
	fmt.size = cfg.size;
	fmt.planesCount = 1;
	int ret = output_->tryFormat(&fmt);
	if (ret < 0) {
		LOG(Camss, Error) << "PIX output try format failed: " << strerror(-ret);
		return {};
	}
	cfg.stride = fmt.planes[0].bpl;
	cfg.frameSize = fmt.planes[0].size;
	cfg.bufferCount = std::max(kBufferCount, req.bufferCount);
	cfg.setStream(const_cast<Stream *>(&outStream_));

	return cfg;
}

int CamssIspInline::configure(const StreamConfiguration &inputCfg,
			      const StreamConfiguration &outputCfg)
{
	int ret;

	MediaLink *link = csi_->pixLink();
	if (!(link->flags() & MEDIA_LNK_FL_ENABLED)) {
		ret = link->setEnabled(true);
		if (ret)
			return ret;
	}

	/*
	 * The CSID's pixel port reports what it delivers (it may crop a line
	 * or two to turn the Bayer order into the IFE's); take that for the
	 * PIX sink and ask for YUV at the same size on its source.
	 */
	V4L2SubdeviceFormat csidFmt;
	ret = csi_->csidSubdev()->getFormat(4, &csidFmt, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;

	ret = pix_->setFormat(0, &csidFmt, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;

	V4L2SubdeviceFormat pixFmt = csidFmt;
	pixFmt.code = MEDIA_BUS_FMT_YUYV8_1_5X8;
	ret = pix_->setFormat(1, &pixFmt, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;

	V4L2DeviceFormat fmt;
	fmt.fourcc = output_->toV4L2PixelFormat(outputCfg.pixelFormat);
	fmt.size = pixFmt.size;
	fmt.planes[0].bpl = outputCfg.stride;
	fmt.planesCount = 1;
	ret = output_->setFormat(&fmt);
	if (ret < 0) {
		LOG(Camss, Error) << "PIX output set format failed: " << strerror(-ret);
		return ret;
	}
	if (fmt.size != outputCfg.size)
		LOG(Camss, Warning) << "PIX output size " << fmt.size
				    << " differs from the configured " << outputCfg.size;

	size_ = fmt.size;
	stride_ = fmt.planes[0].bpl;
	bufferCount_ = outputCfg.bufferCount;
	exposureMargin_ = inputCfg.size.height + vblankDef_ - exposureMax_;

	return 0;
}

int CamssIspInline::allocateBuffers(unsigned int bufferCount)
{
	int ret = output_->importBuffers(std::max(bufferCount, bufferCount_));
	if (ret < 0)
		return ret;

	frameInfos_->init({}, {}, true);
	return 0;
}

void CamssIspInline::freeBuffers()
{
	if (output_->releaseBuffers())
		LOG(Camss, Error) << "Failed to release PIX buffers";
}

int CamssIspInline::exportOutputBuffers([[maybe_unused]] const Stream *stream,
					unsigned int count,
					std::vector<std::unique_ptr<FrameBuffer>> *buffers)
{
	return output_->exportBuffers(count, buffers);
}

void CamssIspInline::queueBuffers([[maybe_unused]] Request *request,
				  [[maybe_unused]] FrameBuffer *input)
{
	LOG(Camss, Fatal) << "The inline ISP takes requests, not raw buffers";
}

int CamssIspInline::queueRequest(Request *request)
{
	FrameBuffer *buffer = request->findBuffer(&outStream_);
	if (!buffer) {
		LOG(Camss, Error) << "Request without an output buffer";
		return -EINVAL;
	}

	return output_->queueBuffer(buffer);
}

void CamssIspInline::processStats([[maybe_unused]] const uint32_t frame,
				  [[maybe_unused]] const uint32_t statsBufferId,
				  [[maybe_unused]] const ControlList &sensorControls)
{
}

int CamssIspInline::start()
{
	frameCount_ = 0;
	applyIsp();
	applySensor();

	return output_->streamOn();
}

void CamssIspInline::stop()
{
	if (output_->streamOff())
		LOG(Camss, Error) << "PIX stream off failed";
}

/*
 * Set the sensor's frame length, exposure and gain. Exposure and gain go
 * through the pipeline's DelayedControls by way of setSensorControls; the
 * frame length is applied directly, since the delayed controls do not know
 * it and it moves rarely.
 */
void CamssIspInline::applySensor()
{
	CameraSensor *sensor = const_cast<CameraSensor *>(sensor_);
	ControlList vblank(sensor->controls());
	vblank.set(V4L2_CID_VBLANK, exposure_.vblank);
	sensor->setControls(&vblank);

	ControlList ctrls(sensor->controls());
	ctrls.set(V4L2_CID_EXPOSURE, exposure_.exposure);
	ctrls.set(V4L2_CID_ANALOGUE_GAIN, exposure_.gain);
	setSensorControls.emit(ctrls);
}

void CamssIspInline::applyIsp()
{
	ControlList ctrls(pix_->controls());
	ctrls.set(V4L2_CID_BLACK_LEVEL, blackLevel_);
	ctrls.set(V4L2_CID_RED_BALANCE, std::min(kWbMax, wbRed_ * exposure_.digital / kWbUnity));
	ctrls.set(V4L2_CID_BLUE_BALANCE, std::min(kWbMax, wbBlue_ * exposure_.digital / kWbUnity));
	ctrls.set(V4L2_CID_GAIN, exposure_.digital);
	pix_->setControls(&ctrls);
}

/*
 * Statistics from the NV12 output: mean luma, and linear mean R, G, B
 * recovered through BT.601 and the gamma the driver programs. From them,
 * exposure: full frame first, then gain to the knee, then a longer frame,
 * then the rest of the gain, then digital gain in the IFE; and grey-world
 * white balance, applied as a correction to the gains already in place
 * since the statistics are taken after them.
 */
void CamssIspInline::runAlgorithms(FrameBuffer *buffer, unsigned int frame, ControlList &metadata)
{
	MappedFrameBuffer map(buffer, MappedFrameBuffer::MapFlag::Read);
	if (!map.isValid()) {
		LOG(Camss, Warning) << "Cannot map the output buffer for statistics";
		return;
	}

	/* NV12 comes as one V4L2 plane but two logical planes, or as one */
	const std::vector<Span<uint8_t>> &planes = map.planes();
	unsigned int w = size_.width, h = size_.height;
	const uint8_t *y = planes[0].data();
	const uint8_t *c;
	if (planes.size() >= 2) {
		c = planes[1].data();
		if (planes[0].size() < stride_ * h || planes[1].size() < stride_ * (h / 2)) {
			LOG(Camss, Warning) << "Output planes too small for statistics";
			return;
		}
	} else {
		c = y + stride_ * h;
		if (planes[0].size() < stride_ * h * 3 / 2) {
			LOG(Camss, Warning) << "Output plane too small for statistics";
			return;
		}
	}

	double sumY = 0, sumR = 0, sumG = 0, sumB = 0;
	unsigned int n = 0;
	for (unsigned int ry = 0; ry + 1 < h; ry += kStatsStep) {
		for (unsigned int rx = 0; rx + 1 < w; rx += kStatsStep) {
			double Y = y[ry * stride_ + rx];
			const uint8_t *cp = c + (ry / 2) * stride_ + (rx / 2) * 2;
			double cb = cp[0] - 128.0, cr = cp[1] - 128.0;
			double r = std::clamp(Y + 1.402 * cr, 0.0, 255.0);
			double g = std::clamp(Y - 0.344 * cb - 0.714 * cr, 0.0, 255.0);
			double b = std::clamp(Y + 1.772 * cb, 0.0, 255.0);
			sumY += Y;
			sumR += std::pow(r / 255.0, 2.2);
			sumG += std::pow(g / 255.0, 2.2);
			sumB += std::pow(b / 255.0, 2.2);
			n++;
		}
	}
	if (!n)
		return;

	double meanY = sumY / n;
	double R = sumR / n, G = sumG / n, B = sumB / n;

	/* Exposure */
	double k = std::pow(kTargetLuma / 255.0, 2.2) / std::max(std::pow(meanY / 255.0, 2.2), 1e-4);
	k = std::clamp(std::pow(k, 0.6), 0.5, 2.0);

	const double gainMin = 256.0;	/* the IMX681's unity code is 0; use 256 as the floor the loop reasons in */
	double total = (double)exposure_.exposure * std::max<double>(exposure_.gain, gainMin) *
		       exposure_.digital * k / kWbUnity;
	int32_t frameLines = exposureMax_ + exposureMargin_;
	int32_t exposureSlow = (int32_t)(frameLines * kSlowFrame) - exposureMargin_;
	Exposure e;
	if (total <= exposureMax_ * gainMin) {
		e = { std::max<int32_t>(100, std::min<int32_t>(exposureMax_, total / gainMin)), 0, vblankDef_, kWbUnity };
	} else if (total <= exposureMax_ * kGainKnee) {
		e = { exposureMax_, std::min<int32_t>(kGainKnee, total / exposureMax_), vblankDef_, kWbUnity };
	} else if (total <= (double)exposureSlow * kGainKnee) {
		e = { std::min<int32_t>(exposureSlow, total / kGainKnee), kGainKnee, 0, kWbUnity };
	} else if (total <= (double)exposureSlow * gainMax_) {
		e = { exposureSlow, std::min<int32_t>(gainMax_, total / exposureSlow), 0, kWbUnity };
	} else {
		e = { exposureSlow, gainMax_, 0,
		      std::clamp<int32_t>(total / ((double)exposureSlow * gainMax_) * kWbUnity, kWbUnity, kDigitalMax) };
	}
	e.vblank = std::clamp<int32_t>(e.exposure + exposureMargin_ - (frameLines - vblankDef_),
				       vblankDef_, vblankMax_);
	exposure_ = e;

	/* White balance, incremental grey world */
	double kr = G / std::max(R, 1e-4), kb = G / std::max(B, 1e-4);
	wbRed_ = std::clamp<int32_t>(wbRed_ * std::pow(kr, 0.5), 256, kWbMax);
	wbBlue_ = std::clamp<int32_t>(wbBlue_ * std::pow(kb, 0.5), 256, kWbMax);

	applySensor();
	applyIsp();

	if (frame % 30 == 0)
		LOG(Camss, Debug) << "frame " << frame << " Y " << meanY
				  << " exposure " << exposure_.exposure
				  << " vblank " << exposure_.vblank
				  << " gain " << exposure_.gain
				  << " digital " << exposure_.digital
				  << " R " << wbRed_ << " B " << wbBlue_;

	metadata.set(controls::AnalogueGain, 1024.0f / (1024.0f - exposure_.gain));
	metadata.set(controls::ColourGains, Span<const float, 2>({ wbRed_ / 1024.0f, wbBlue_ / 1024.0f }));
	metadata.set(controls::SensorBlackLevels,
		     Span<const int32_t, 4>({ blackLevel_ << 6, blackLevel_ << 6, blackLevel_ << 6, blackLevel_ << 6 }));
}

void CamssIspInline::bufferReady(FrameBuffer *buffer)
{
	CamssFrames::Info *info = frameInfos_->find(buffer);
	if (!info)
		return;

	Request *request = info->request;
	request->_d()->metadata().set(controls::SensorTimestamp,
				      buffer->metadata().timestamp);

	bool ok = buffer->metadata().status == FrameMetadata::FrameSuccess;
	unsigned int frame = info->id;

	/*
	 * The buffer first, the metadata after: the pipeline completes the
	 * request once both are in, and only the buffer carries the frame.
	 * Every fourth frame feeds the loop; the controls land within a frame
	 * or two.
	 */
	outputBufferReady.emit(buffer);

	ControlList metadata(controls::controls);
	if (ok && (frameCount_++ % 4) == 0)
		runAlgorithms(buffer, frame, metadata);
	metadataReady.emit(frame, metadata);
}

} /* namespace libcamera */
