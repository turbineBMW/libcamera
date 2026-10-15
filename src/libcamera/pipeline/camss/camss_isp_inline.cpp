/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * Copyright (C) 2026 Brandon Williams
 *
 * camss_isp_inline.cpp - Inline IFE ISP: NV12 straight from the pixel path
 */

#include "camss_isp_inline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <linux/dma-buf.h>
#include <linux/media-bus-format.h>
#include <linux/qcom-camss-controls.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>

#include <libcamera/base/file.h>
#include <libcamera/base/log.h>

#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/ipa/core_ipa_interface.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include "libcamera/internal/bayer_format.h"
#include "libcamera/internal/camera_sensor.h"
#include "libcamera/internal/framebuffer.h"
#include "libcamera/internal/request.h"
#include "libcamera/internal/mapped_framebuffer.h"
#include "libcamera/internal/media_device.h"
#include "libcamera/internal/yaml_parser.h"

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
constexpr double kTargetLuma = 120.0;
constexpr double kCentreWeight = 3.0;	/* weight of the central quarter of the frame */
constexpr int32_t kGainKnee = 768;	/* IMX681 code for 4x */
constexpr double kSlowFrame = 2.0;	/* allow frames up to this many default lengths in the dark */
constexpr unsigned int kStatsStep = 16;
constexpr int32_t kDefaultBlackLevel = 64;

/*
 * A tone curve with a linear toe: slope \a toe up to \a knee, then a
 * power curve of exponent 1/\a gamma that meets it there and ends at 1.
 * The driver's LUT takes 256 twelve-bit outputs indexed by the top eight
 * bits of the input.
 */
std::array<uint16_t, 256> toneCurve(double toe, double knee, double gamma)
{
	std::array<uint16_t, 256> lut;
	double a = (1.0 - toe * knee) / (1.0 - std::pow(knee, 1.0 / gamma));

	for (unsigned int i = 0; i < lut.size(); i++) {
		double x = i / 255.0;
		double y = x < knee ? toe * x : a * std::pow(x, 1.0 / gamma) - (a - 1.0);
		lut[i] = static_cast<uint16_t>(std::lround(std::clamp(y, 0.0, 1.0) * 4095.0));
	}

	return lut;
}

} /* namespace */

/*
 * Output sizes: the sensor size and a ladder of common ones within what
 * the scalers reach (a sixteenth of the sensor size), widths a multiple
 * of 16 for the write master. Discrete sizes rather than a range: the
 * pipewire plugin encodes a libcamera range as a stepped rectangle choice
 * that it cannot then intersect with a client's fixed size.
 */
static std::vector<Size> outputSizes(const Size &in)
{
	static const Size ladder[] = {
		{ 3840, 2160 }, { 2560, 1440 }, { 1920, 1440 }, { 1920, 1080 },
		{ 1600, 1200 }, { 1280, 960 }, { 1280, 720 }, { 1024, 768 },
		{ 960, 540 }, { 800, 600 }, { 640, 480 }, { 640, 360 }, { 320, 240 },
	};
	constexpr unsigned int kScaleMax = 16;
	std::vector<Size> sizes;

	sizes.push_back(Size(in.width & ~15u, in.height & ~1u));
	for (const Size &sz : ladder) {
		if (sz.width <= in.width && sz.height <= in.height &&
		    sz.width * kScaleMax >= in.width && sz.height * kScaleMax >= in.height &&
		    sz != sizes.front())
			sizes.push_back(sz);
	}
	return sizes;
}

static bool sizeOffered(const Size &in, const Size &req)
{
	for (const Size &sz : outputSizes(in))
		if (sz == req)
			return true;
	return false;
}

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

	isp->publishControls(ispControls);

	LOG(Camss, Info) << "Using the inline IFE pixel path for "
			 << csi->sensor()->id();
	return isp;
}

/*
 * The IMX681 gain code to a multiplier and back; the sensor's unity code is
 * zero and the response is 1024 / (1024 - code).
 */
static float gainFromCode(int32_t code)
{
	return 1024.0f / (1024.0f - std::clamp<int32_t>(code, 0, 1023));
}

static int32_t gainToCode(float gain)
{
	if (gain <= 1.0f)
		return 0;
	return std::clamp<int32_t>(std::lround(1024.0f - 1024.0f / gain), 0, 1023);
}

void CamssIspInline::updateLineDuration()
{
	IPACameraSensorInfo info{};
	if (sensor_->sensorInfo(&info) == 0 && info.pixelRate && info.minLineLength)
		lineDurationUs_ = info.minLineLength * 1e6 / info.pixelRate;
	else
		lineDurationUs_ = 0.0;
}

/* The longest exposure a manual setting may ask for, at the longest frame */
int32_t CamssIspInline::manualExposureMax() const
{
	return exposureMax_ + (vblankMax_ - vblankDef_);
}

/*
 * The digital gain the IFE can still add on top of the colour gains: the
 * channel gains it receives are white balance times digital, and once the
 * strongest of them reaches the block's limit, more digital gain only
 * raises green and the white balance is lost (a green picture in the dark).
 */
int32_t CamssIspInline::digitalMax() const
{
	int32_t wb = std::max({ wbRed_, wbBlue_, kWbUnity });
	return std::clamp<int32_t>(channelGainMax_ * kWbUnity / wb, kWbUnity, kDigitalMax);
}

/*
 * The controls applications see: the auto algorithms on or off, and the
 * manual exposure, gain and colour gains that apply when they are off.
 * Without any, the pipewire libcamera node is dropped by wireplumber
 * (an empty PropInfo enumeration fails its activation).
 */
void CamssIspInline::publishControls(ControlInfoMap *ispControls) const
{
	ControlInfoMap::Map map;
	double line = lineDurationUs_ > 0.0 ? lineDurationUs_ : 1.0;

	map[&controls::AeEnable] = ControlInfo(false, true, true);
	map[&controls::ExposureTime] =
		ControlInfo(static_cast<int32_t>(std::lround(line)),
			    static_cast<int32_t>(std::lround(manualExposureMax() * line)),
			    static_cast<int32_t>(std::lround(exposure_.exposure * line)));
	map[&controls::AnalogueGain] =
		ControlInfo(1.0f, gainFromCode(gainMax_), 1.0f);
	map[&controls::AwbEnable] = ControlInfo(false, true, true);
	map[&controls::ColourGains] = ControlInfo(0.25f, 4.0f, 1.0f);

	*ispControls = ControlInfoMap(std::move(map), controls::controls);
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
	aeEnabled_ = true;
	awbEnabled_ = true;
	updateLineDuration();
	wbRed_ = kWbUnity;
	wbBlue_ = kWbUnity;
	auto wbCtrl = pix_->controls().find(V4L2_CID_RED_BALANCE);
	channelGainMax_ = wbCtrl != pix_->controls().end()
			? wbCtrl->second.max().get<int32_t>() : kWbMax;
	blackLevel_ = kDefaultBlackLevel;
	targetLuma_ = kTargetLuma;
	zoom_ = 1.0;
	spatialMinNoise_ = 1.5;
	spatialActive_ = true;
	centreWeight_ = kCentreWeight;
	slowFrame_ = kSlowFrame;
	sensorDigitalMax_ = 1.0;
	hasSensorDigital_ = ctrls.find(V4L2_CID_DIGITAL_GAIN) != ctrls.end();
	awbWarmth_ = 0.0;
	awbNeutral_ = 1.0;
	awbRedBias_ = 1.0;
	awbBlueBias_ = 1.0;
	loadTuning();

	output_->bufferReady.connect(this, &CamssIspInline::bufferReady);
	valid_ = true;
}

/*
 * Per-sensor tuning, optional: $LIBCAMERA_CAMSS_TUNING or
 * <sysconfdir>/libcamera/camss/<sensor model>.yaml:
 *
 *   version: 1
 *   ae: { target: 120, centre_weight: 3, slow_frame: 2 }   # frame length cap, x default
 *   awb: { red_bias: 1.0, blue_bias: 0.95 }    # leave some of the illuminant in
 *   awb: { warmth: 0.25, neutral_ratio: 1.0 }   # keep part of a warm light's colour
 *   ae: { sensor_digital_max: 6 }               # sensor digital gain after analogue
 *   black_level: 64                             # sensor pedestal, 10-bit units
 *   zoom: 1.3                                   # narrow the field of view (centre crop)
 *   gamma: { toe: 2.6, knee: 0.17, gamma: 2.9 }   # or gamma_lut: [256 x 0..4095]
 *   ccm: [r_r, r_g, r_b, g_r, g_g, g_b, b_r, b_g, b_b]   # row-major, 1.0 unity
 *   ccm_strength: 1.0                                   # blend towards identity
 *   ccm_offsets: [r, g, b]                              # fraction of full scale
 *   lsc: { r: [221 gains], gr: [...], gb: [...], b: [...] }  # 13 rows x 17 columns
 *   lsc_strength: 1.0                                   # blend towards unity, 0 = block off
 *   lsc_response: [a, b]                                # applied = a g^2 + b g + (1 - a - b)
 *   denoise: { enable: true, temporal: 0.2, threshold: 2.5, ramp: 2.0, chroma_radius: 2 }
 *   spatial: { enable: true, luma: [4 per level], chroma: [4], luma_ratios: [4],
 *              chroma_ratios: [4], threads: 4, min_noise: 1.5 }   # runs before denoise
 *
 * Without a file the loop keeps its defaults and the driver its own curve
 * and identity matrix.
 */
void CamssIspInline::loadTuning()
{
	std::string path;
	const char *env = std::getenv("LIBCAMERA_CAMSS_TUNING");
	if (env && *env) {
		path = env;
	} else {
		path = (std::filesystem::path(LIBCAMERA_SYSCONF_DIR) / "camss" /
			(sensor_->model() + ".yaml")).string();
	}

	File file(path);
	if (!file.exists())
		return;
	if (!file.open(File::OpenModeFlag::ReadOnly)) {
		LOG(Camss, Warning) << "Cannot open tuning file " << path;
		return;
	}

	std::unique_ptr<ValueNode> root = YamlParser::parse(file);
	if (!root) {
		LOG(Camss, Warning) << "Cannot parse tuning file " << path;
		return;
	}

	const ValueNode &ae = (*root)["ae"];
	targetLuma_ = ae["target"].get<double>().value_or(kTargetLuma);
	centreWeight_ = ae["centre_weight"].get<double>().value_or(kCentreWeight);
	slowFrame_ = std::max(ae["slow_frame"].get<double>().value_or(kSlowFrame), 1.0);
	sensorDigitalMax_ = std::clamp(ae["sensor_digital_max"].get<double>().value_or(1.0), 1.0, 15.0);
	const ValueNode &awb = (*root)["awb"];
	awbRedBias_ = awb["red_bias"].get<double>().value_or(1.0);
	awbBlueBias_ = awb["blue_bias"].get<double>().value_or(1.0);
	awbWarmth_ = std::clamp(awb["warmth"].get<double>().value_or(0.0), 0.0, 1.0);
	awbNeutral_ = std::max(awb["neutral_ratio"].get<double>().value_or(1.0), 0.1);
	blackLevel_ = (*root)["black_level"].get<int32_t>().value_or(kDefaultBlackLevel);
	zoom_ = (*root)["zoom"].get<double>().value_or(1.0);

	const ValueNode &dn = (*root)["denoise"];
	denoiseParams_.enabled = dn["enable"].get<bool>().value_or(denoiseParams_.enabled);
	denoiseParams_.minWeight = dn["temporal"].get<double>().value_or(denoiseParams_.minWeight);
	denoiseParams_.threshold = dn["threshold"].get<double>().value_or(denoiseParams_.threshold);
	denoiseParams_.ramp = dn["ramp"].get<double>().value_or(denoiseParams_.ramp);
	denoiseParams_.chromaRadius = dn["chroma_radius"].get<uint32_t>().value_or(denoiseParams_.chromaRadius);

	const ValueNode &sp = (*root)["spatial"];
	spatialParams_.enabled = sp["enable"].get<bool>().value_or(spatialParams_.enabled);
	spatialParams_.threads = sp["threads"].get<uint32_t>().value_or(spatialParams_.threads);
	spatialMinNoise_ = sp["min_noise"].get<double>().value_or(spatialMinNoise_);
	auto levels = [&](const char *key, std::array<double, CamssSpatial::kLevels> &out) {
		auto v = sp[key].get<std::vector<double>>();
		if (!v)
			return;
		if (v->size() != out.size()) {
			LOG(Camss, Warning) << "spatial." << key << " needs " << out.size() << " values";
			return;
		}
		std::copy(v->begin(), v->end(), out.begin());
	};
	levels("luma", spatialParams_.luma);
	levels("chroma", spatialParams_.chroma);
	levels("luma_ratios", spatialParams_.lumaRatios);
	levels("chroma_ratios", spatialParams_.chromaRatios);

	if (root->contains("gamma_lut")) {
		auto values = (*root)["gamma_lut"].get<std::vector<uint32_t>>();
		if (values && values->size() == 256) {
			std::array<uint16_t, 256> lut;
			for (unsigned int i = 0; i < 256; i++)
				lut[i] = std::min<uint32_t>((*values)[i], QCOM_CAMSS_GAMMA_LUT_MAX);
			gammaLut_ = lut;
		} else {
			LOG(Camss, Warning) << "gamma_lut needs 256 entries";
		}
	} else if (root->contains("gamma")) {
		const ValueNode &g = (*root)["gamma"];
		gammaLut_ = toneCurve(g["toe"].get<double>().value_or(1.0),
				      g["knee"].get<double>().value_or(0.0),
				      g["gamma"].get<double>().value_or(2.2));
	}

	if (root->contains("ccm")) {
		auto m = (*root)["ccm"].get<std::vector<double>>();
		auto k = (*root)["ccm_offsets"].get<std::vector<double>>();
		double strength = (*root)["ccm_strength"].get<double>().value_or(1.0);
		if (m && m->size() == 9) {
			std::array<int32_t, 12> ccm{};
			for (unsigned int i = 0; i < 9; i++) {
				double identity = i % 4 == 0 ? 1.0 : 0.0;
				double v = identity + strength * ((*m)[i] - identity);
				ccm[i] = std::lround(v * QCOM_CAMSS_CCM_UNITY);
			}
			for (unsigned int i = 0; i < 3; i++)
				ccm[9 + i] = (k && k->size() == 3) ? std::lround((*k)[i] * 4095.0) : 0;
			ccm_ = ccm;
		} else {
			LOG(Camss, Warning) << "ccm needs 9 coefficients";
		}
	}

	if (root->contains("lsc")) {
		const ValueNode &lsc = (*root)["lsc"];
		double strength = (*root)["lsc_strength"].get<double>().value_or(1.0);
		/*
		 * The block's measured gain response: applied = a g^2 + b g + c
		 * for a table value g (c = 1 - a - b); the table is written with
		 * the inverse so the mesh's gains come out as given.
		 */
		auto resp = (*root)["lsc_response"].get<std::vector<double>>();
		double ra = 0.0, rb = 1.0;
		if (resp && resp->size() == 2) {
			ra = (*resp)[0];
			rb = (*resp)[1];
		}
		constexpr unsigned int points = QCOM_CAMSS_LSC_MESH_ROWS * QCOM_CAMSS_LSC_MESH_COLS;
		std::vector<uint16_t> mesh;
		for (const char *ch : { "r", "gr", "gb", "b" }) {
			auto v = lsc[ch].get<std::vector<double>>();
			if (!v || v->size() != points) {
				LOG(Camss, Warning) << "lsc." << ch << " needs " << points << " gains";
				mesh.clear();
				break;
			}
			for (double gain : *v) {
				double g = 1.0 + strength * (gain - 1.0);
				if (ra != 0.0) {
					double rc = 1.0 - ra - rb;
					g = (-rb + std::sqrt(rb * rb - 4.0 * ra * (rc - g))) / (2.0 * ra);
				} else if (rb != 1.0) {
					g = 1.0 + (g - 1.0) / rb;
				}
				mesh.push_back(std::clamp<int>(std::lround(g * QCOM_CAMSS_LSC_UNITY),
							       QCOM_CAMSS_LSC_UNITY, QCOM_CAMSS_LSC_MAX));
			}
		}
		/* strength 0 switches the block off rather than running it at unity */
		if (strength == 0.0)
			std::fill(mesh.begin(), mesh.end(), 0);
		if (!mesh.empty())
			lscMesh_ = std::move(mesh);
	}

	LOG(Camss, Info) << "Tuning from " << path << ": target " << targetLuma_
			 << (gammaLut_ ? ", gamma curve" : "") << (ccm_ ? ", colour matrix" : "")
			 << (lscMesh_ ? ", shading mesh" : "")
			 << (spatialParams_.enabled ? ", spatial denoise" : "")
			 << (denoiseParams_.enabled ? ", temporal denoise" : "");
}

CamssIspInline::~CamssIspInline() = default;

Size CamssIspInline::getMargins([[maybe_unused]] PixelFormat inputFormat)
{
	return { 0, 0 };
}

StreamConfiguration CamssIspInline::generateConfiguration(const StreamConfiguration &raw) const
{
	std::map<PixelFormat, std::vector<SizeRange>> formats;
	for (const Size &sz : outputSizes(raw.size))
		formats[formats::NV12].emplace_back(sz);

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

	cfg.pixelFormat = formats::NV12;
	if (sizeOffered(raw.size, req.size))
		cfg.size = req.size;
	else
		cfg.size = outputSizes(raw.size).front();

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

	/*
	 * Keep the sensor's aspect: scale so the output is covered (the sink
	 * pad's compose rectangle), then crop the centre to it (the source
	 * pad's crop rectangle). Even offsets, so the chroma lands on a
	 * sample. The tuning's zoom scales to that much more than the
	 * output before the crop, as far as the scaler (downscale only)
	 * allows.
	 */
	const Size &in = csidFmt.size;
	const Size &out = outputCfg.size;
	Size scaled;
	if ((uint64_t)out.width * in.height >= (uint64_t)out.height * in.width)
		scaled = Size(out.width, ((uint64_t)in.height * out.width / in.width) & ~1u);
	else
		scaled = Size(((uint64_t)in.width * out.height / in.height) & ~1u, out.height);
	scaled.width = std::max(scaled.width, out.width);
	scaled.height = std::max(scaled.height, out.height);
	double zoom = std::min({ std::max(zoom_, 1.0),
				 static_cast<double>(in.width) / scaled.width,
				 static_cast<double>(in.height) / scaled.height });
	scaled = Size(static_cast<unsigned int>(scaled.width * zoom) & ~1u,
		      static_cast<unsigned int>(scaled.height * zoom) & ~1u);
	scaled.width = std::max(scaled.width, out.width);
	scaled.height = std::max(scaled.height, out.height);
	Rectangle compose(0, 0, scaled);
	ret = pix_->setSelection(0, V4L2_SEL_TGT_COMPOSE, &compose, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;
	Rectangle crop(((compose.width - out.width) / 2) & ~1u,
		       ((compose.height - out.height) / 2) & ~1u, out);
	ret = pix_->setSelection(1, V4L2_SEL_TGT_CROP, &crop, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;
	LOG(Camss, Info) << "PIX scale " << in << " to " << compose.size()
			  << ", crop " << crop;

	V4L2SubdeviceFormat pixFmt = csidFmt;
	pixFmt.code = MEDIA_BUS_FMT_YUYV8_1_5X8;
	pixFmt.size = outputCfg.size;
	ret = pix_->setFormat(1, &pixFmt, V4L2Subdevice::ActiveFormat);
	if (ret)
		return ret;
	if (pixFmt.size != outputCfg.size) {
		LOG(Camss, Error) << "PIX source pad gave " << pixFmt.size
				  << " for " << outputCfg.size;
		return -EINVAL;
	}

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
	denoise_.configure(size_, stride_, denoiseParams_);
	spatial_.configure(size_, stride_, spatialParams_);
	spatialActive_ = true;

	/*
	 * The exposure limit follows the frame length the sensor currently
	 * has, so read both together for the margin, then take the limit at
	 * the default frame length as the full-rate maximum.
	 */
	CameraSensor *sensor = const_cast<CameraSensor *>(sensor_);
	static const uint32_t ids[] = { V4L2_CID_VBLANK };
	ControlList cur = sensor->getControls(ids);
	const ControlInfoMap &ctrls = sensor->controls();
	int32_t vblankCur = cur.contains(V4L2_CID_VBLANK) ? cur.get(V4L2_CID_VBLANK).get<int32_t>() : vblankDef_;
	int32_t exposureMaxCur = ctrls.find(V4L2_CID_EXPOSURE)->second.max().get<int32_t>();
	exposureMargin_ = inputCfg.size.height + vblankCur - exposureMaxCur;
	exposureMax_ = inputCfg.size.height + vblankDef_ - exposureMargin_;
	exposure_.exposure = std::min(exposure_.exposure, exposureMax_);
	exposure_.vblank = vblankDef_;
	updateLineDuration();

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
	maps_.clear();
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

	/*
	 * Manual exposure, gain or colour gains take the matching loop out
	 * of the picture; an explicit AeEnable or AwbEnable in the same
	 * request has the last word.
	 */
	const ControlList &ctrls = request->controls();
	bool sensorChanged = false, ispChanged = false;

	if (const auto &exp = ctrls.get(controls::ExposureTime)) {
		if (*exp > 0 && lineDurationUs_ > 0.0) {
			int32_t frameLines = exposureMax_ + exposureMargin_;
			aeEnabled_ = false;
			exposure_.exposure = std::clamp<int32_t>(std::lround(*exp / lineDurationUs_),
								 1, manualExposureMax());
			exposure_.vblank = std::clamp<int32_t>(exposure_.exposure + exposureMargin_ -
							       (frameLines - vblankDef_),
							       vblankDef_, vblankMax_);
			exposure_.digital = kWbUnity;
			exposure_.sensorDigital = 256;
			sensorChanged = ispChanged = true;
		}
	}
	if (const auto &gain = ctrls.get(controls::AnalogueGain)) {
		aeEnabled_ = false;
		exposure_.gain = std::min(gainToCode(*gain), gainMax_);
		exposure_.digital = kWbUnity;
		exposure_.sensorDigital = 256;
		sensorChanged = ispChanged = true;
	}
	if (const auto &ae = ctrls.get(controls::AeEnable))
		aeEnabled_ = *ae;

	if (const auto &gains = ctrls.get(controls::ColourGains)) {
		awbEnabled_ = false;
		wbRed_ = std::clamp<int32_t>(std::lround((*gains)[0] * kWbUnity), 256, kWbMax);
		wbBlue_ = std::clamp<int32_t>(std::lround((*gains)[1] * kWbUnity), 256, kWbMax);
		ispChanged = true;
	}
	if (const auto &awb = ctrls.get(controls::AwbEnable))
		awbEnabled_ = *awb;

	if (sensorChanged)
		applySensor();
	if (ispChanged)
		applyIsp();

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
	denoise_.reset();
	applyTables();
	applyIsp();
	applySensor();

	return output_->streamOn();
}

void CamssIspInline::stop()
{
	if (output_->streamOff())
		LOG(Camss, Error) << "PIX stream off failed";
	maps_.clear();
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
	if (hasSensorDigital_)
		vblank.set(V4L2_CID_DIGITAL_GAIN, exposure_.sensorDigital);
	sensor->setControls(&vblank);

	ControlList ctrls(sensor->controls());
	ctrls.set(V4L2_CID_EXPOSURE, exposure_.exposure);
	ctrls.set(V4L2_CID_ANALOGUE_GAIN, exposure_.gain);
	setSensorControls.emit(ctrls);
}

/* The gamma curve and colour matrix from the tuning, on drivers that take them */
void CamssIspInline::applyTables()
{
	const ControlInfoMap &info = pix_->controls();
	ControlList ctrls(info);

	if (gammaLut_ && info.find(V4L2_CID_QCOM_CAMSS_GAMMA_LUT) != info.end())
		ctrls.set(V4L2_CID_QCOM_CAMSS_GAMMA_LUT, Span<const uint16_t>(*gammaLut_));
	if (ccm_ && info.find(V4L2_CID_QCOM_CAMSS_CCM) != info.end())
		ctrls.set(V4L2_CID_QCOM_CAMSS_CCM, Span<const int32_t>(*ccm_));
	if (lscMesh_ && info.find(V4L2_CID_QCOM_CAMSS_LSC_MESH) != info.end())
		ctrls.set(V4L2_CID_QCOM_CAMSS_LSC_MESH, Span<const uint16_t>(*lscMesh_));
	if (!ctrls.empty() && pix_->setControls(&ctrls) < 0)
		LOG(Camss, Warning) << "Cannot set the tuning tables";
}

void CamssIspInline::applyIsp()
{
	exposure_.digital = std::min(exposure_.digital, digitalMax());

	ControlList ctrls(pix_->controls());
	ctrls.set(V4L2_CID_BLACK_LEVEL, blackLevel_);
	ctrls.set(V4L2_CID_RED_BALANCE, std::min(channelGainMax_, wbRed_ * exposure_.digital / kWbUnity));
	ctrls.set(V4L2_CID_BLUE_BALANCE, std::min(channelGainMax_, wbBlue_ * exposure_.digital / kWbUnity));
	ctrls.set(V4L2_CID_GAIN, exposure_.digital);
	pix_->setControls(&ctrls);
}

/*
 * The output buffers are the application's and come back request after
 * request; map each once, read-write, and keep it until the stream stops.
 */
MappedFrameBuffer *CamssIspInline::mapBuffer(FrameBuffer *buffer)
{
	auto it = maps_.find(buffer);
	if (it != maps_.end())
		return it->second.get();

	auto map = std::make_unique<MappedFrameBuffer>(buffer, MappedFrameBuffer::MapFlag::ReadWrite);
	if (!map->isValid()) {
		LOG(Camss, Warning) << "Cannot map the output buffer";
		return nullptr;
	}
	return maps_.emplace(buffer, std::move(map)).first->second.get();
}

/* NV12 comes as one V4L2 plane but two logical planes, or as one */
bool CamssIspInline::framePlanes(FrameBuffer *buffer, uint8_t **y, uint8_t **c)
{
	MappedFrameBuffer *map = mapBuffer(buffer);
	if (!map)
		return false;

	const std::vector<Span<uint8_t>> &planes = map->planes();
	unsigned int h = size_.height;
	*y = planes[0].data();
	if (planes.size() >= 2) {
		*c = planes[1].data();
		if (planes[0].size() < stride_ * h || planes[1].size() < stride_ * (h / 2)) {
			LOG(Camss, Warning) << "Output planes too small";
			return false;
		}
	} else {
		*c = *y + stride_ * h;
		if (planes[0].size() < stride_ * h * 3 / 2) {
			LOG(Camss, Warning) << "Output plane too small";
			return false;
		}
	}
	return true;
}

/*
 * Noise reduction in place, before the buffer goes to the application;
 * the CPU's view of the dma-buf is synchronised around it.
 */
void CamssIspInline::denoise(FrameBuffer *buffer, unsigned int frame)
{
	uint8_t *y, *c;
	if (!framePlanes(buffer, &y, &c))
		return;

	int fd = buffer->planes()[0].fd.get();
	struct dma_buf_sync sync = { DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW };
	ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);

	/*
	 * The spatial stage first, so the temporal one sees a cleaner frame
	 * and can tell movement from noise more finely. In good light the
	 * spatial stage has little to remove; it then rests and looks again
	 * every 30 frames.
	 */
	auto t0 = std::chrono::steady_clock::now();
	if (spatial_.enabled() && (spatialActive_ || frame % 30 == 0)) {
		spatial_.process(y, c);
		spatialActive_ = spatial_.lumaSigma() >= spatialMinNoise_;
	}
	auto t1 = std::chrono::steady_clock::now();
	if (denoise_.enabled())
		denoise_.process(y, c);
	auto t2 = std::chrono::steady_clock::now();

	sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
	ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
	using ms = std::chrono::duration<double, std::milli>;
	if (frame % 30 == 0)
		LOG(Camss, Debug) << "denoise " << size_ << ": spatial "
				  << (spatialActive_ ? "" : "resting, ") << ms(t1 - t0).count()
				  << " ms, noise " << spatial_.lumaSigma() << "/" << spatial_.chromaSigma()
				  << "; temporal " << ms(t2 - t1).count() << " ms, noise "
				  << denoise_.sigma();
}

/*
 * Statistics from the NV12 output: mean luma, and linear mean R, G, B
 * recovered through BT.601 and the gamma the driver programs. From them,
 * exposure: full frame first, then gain to the knee, then a longer frame,
 * then the rest of the gain, then digital gain in the IFE; and grey-world
 * white balance, applied as a correction to the gains already in place
 * since the statistics are taken after them.
 */
void CamssIspInline::runAlgorithms(FrameBuffer *buffer, unsigned int frame)
{
	uint8_t *yp, *cp_;
	if (!framePlanes(buffer, &yp, &cp_))
		return;
	unsigned int w = size_.width, h = size_.height;
	const uint8_t *y = yp;
	const uint8_t *c = cp_;

	/*
	 * Centre-weighted: the middle half of the frame in each direction,
	 * where a face usually is, counts centreWeight_ times as much as the
	 * rest. The colour sums stay unweighted; the walls are grey more
	 * often than the face.
	 */
	double sumY = 0, sumR = 0, sumG = 0, sumB = 0, sumW = 0;
	unsigned int n = 0;
	for (unsigned int ry = 0; ry + 1 < h; ry += kStatsStep) {
		for (unsigned int rx = 0; rx + 1 < w; rx += kStatsStep) {
			double Y = y[ry * stride_ + rx];
			bool centre = rx >= w / 4 && rx < w * 3 / 4 && ry >= h / 4 && ry < h * 3 / 4;
			double wgt = centre ? centreWeight_ : 1.0;
			const uint8_t *cp = c + (ry / 2) * stride_ + (rx / 2) * 2;
			double cb = cp[0] - 128.0, cr = cp[1] - 128.0;
			double r = std::clamp(Y + 1.402 * cr, 0.0, 255.0);
			double g = std::clamp(Y - 0.344 * cb - 0.714 * cr, 0.0, 255.0);
			double b = std::clamp(Y + 1.772 * cb, 0.0, 255.0);
			sumY += wgt * Y;
			sumW += wgt;
			sumR += std::pow(r / 255.0, 2.2);
			sumG += std::pow(g / 255.0, 2.2);
			sumB += std::pow(b / 255.0, 2.2);
			n++;
		}
	}
	if (!n)
		return;

	double meanY = sumY / sumW;
	double R = sumR / n, G = sumG / n, B = sumB / n;

	if (!aeEnabled_ && !awbEnabled_)
		return;

	/* Exposure */
	if (aeEnabled_) {
	double k = std::pow(targetLuma_ / 255.0, 2.2) / std::max(std::pow(meanY / 255.0, 2.2), 1e-4);
	k = std::clamp(std::pow(k, 0.6), 0.5, 2.0);

	/*
	 * Work in linear light: exposure lines times the real gain multiplier
	 * (the sensor's code 0 is unity, 1024 / (1024 - code) above it).
	 * Treating the code as the multiplier put a 1.33x step at the point
	 * where the loop moves from exposure to gain, and the picture hunted
	 * across it at a third of a stop.
	 */
	const double knee = gainFromCode(kGainKnee);
	const double gainTop = gainFromCode(gainMax_);
	double total = (double)exposure_.exposure * gainFromCode(exposure_.gain) *
		       exposure_.sensorDigital / 256.0 * exposure_.digital * k / kWbUnity;
	/*
	 * Past the analogue gain, the sensor's digital gain before the IFE's:
	 * it costs the same noise, but gain in the IFE shares its limit with
	 * the white balance and runs out first under warm light.
	 */
	const double sdMax = hasSensorDigital_ ? sensorDigitalMax_ : 1.0;
	int32_t frameLines = exposureMax_ + exposureMargin_;
	int32_t exposureSlow = (int32_t)(frameLines * slowFrame_) - exposureMargin_;
	Exposure e;
	if (total <= exposureMax_) {
		e = { std::max<int32_t>(100, (int32_t)total), 0, vblankDef_, kWbUnity };
	} else if (total <= exposureMax_ * knee) {
		e = { exposureMax_, gainToCode(total / exposureMax_), vblankDef_, kWbUnity };
	} else if (total <= exposureSlow * knee) {
		e = { std::min<int32_t>(exposureSlow, total / knee), kGainKnee, 0, kWbUnity };
	} else if (total <= exposureSlow * gainTop) {
		e = { exposureSlow, std::min(gainToCode(total / exposureSlow), gainMax_), 0, kWbUnity };
	} else if (total <= exposureSlow * gainTop * sdMax) {
		e = { exposureSlow, gainMax_, 0, kWbUnity,
		      std::clamp<int32_t>(std::lround(total / (exposureSlow * gainTop) * 256), 256,
					  std::lround(sdMax * 256)) };
	} else {
		e = { exposureSlow, gainMax_, 0,
		      std::clamp<int32_t>(total / (exposureSlow * gainTop * sdMax) * kWbUnity, kWbUnity, digitalMax()),
		      static_cast<int32_t>(std::lround(sdMax * 256)) };
	}
	e.vblank = std::clamp<int32_t>(e.exposure + exposureMargin_ - (frameLines - vblankDef_),
				       vblankDef_, vblankMax_);
	exposure_ = e;
	}

	/* White balance, incremental grey world */
	if (awbEnabled_) {
	/*
	 * Under a warm light (blue gain well above red) leave part of its
	 * colour in, as a warm room looks warm to the eye: the bias moves
	 * with the light's warmth, by its share 'warmth'.
	 */
	double warm = std::sqrt(std::max(static_cast<double>(wbBlue_) / wbRed_ / awbNeutral_, 1.0));
	double kr = awbRedBias_ * std::pow(warm, awbWarmth_) * G / std::max(R, 1e-4);
	double kb = awbBlueBias_ * std::pow(warm, -awbWarmth_) * G / std::max(B, 1e-4);
	wbRed_ = std::clamp<int32_t>(wbRed_ * std::pow(kr, 0.5), 256, kWbMax);
	wbBlue_ = std::clamp<int32_t>(wbBlue_ * std::pow(kb, 0.5), 256, kWbMax);
	}

	if (aeEnabled_)
		applySensor();
	applyIsp();

	if (frame % 30 == 0)
		LOG(Camss, Debug) << "frame " << frame << " Y " << meanY
				  << " exposure " << exposure_.exposure
				  << " vblank " << exposure_.vblank
				  << " gain " << exposure_.gain
				  << " sensor digital " << exposure_.sensorDigital
				  << " digital " << exposure_.digital
				  << " R " << wbRed_ << " B " << wbBlue_;
}

void CamssIspInline::fillMetadata(ControlList &metadata) const
{
	metadata.set(controls::AeEnable, aeEnabled_);
	metadata.set(controls::ExposureTime,
		     static_cast<int32_t>(std::lround(exposure_.exposure * lineDurationUs_)));
	metadata.set(controls::AnalogueGain, gainFromCode(exposure_.gain));
	metadata.set(controls::DigitalGain, exposure_.digital / static_cast<float>(kWbUnity) *
						    exposure_.sensorDigital / 256.0f);
	metadata.set(controls::AwbEnable, awbEnabled_);
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
	if (ok && (denoise_.enabled() || spatial_.enabled()))
		denoise(buffer, frame);

	outputBufferReady.emit(buffer);

	if (ok && (frameCount_++ % 4) == 0)
		runAlgorithms(buffer, frame);

	ControlList metadata(controls::controls);
	fillMetadata(metadata);
	metadataReady.emit(frame, metadata);
}

} /* namespace libcamera */
