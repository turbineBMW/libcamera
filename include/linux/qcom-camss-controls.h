/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/*
 * Qualcomm CAMSS IFE pixel path controls
 *
 * The image quality blocks the driver programs on the PIX line that take
 * tables or matrices from userspace, as V4L2 array controls on the line's
 * subdevice next to the standard white balance and black level controls.
 */

#ifndef __UAPI_QCOM_CAMSS_CONTROLS_H
#define __UAPI_QCOM_CAMSS_CONTROLS_H

#include <linux/v4l2-controls.h>

/*
 * Gamma: one 256-entry table, applied to all three channels after colour
 * correction. An entry is the 12-bit output for the input value whose top
 * eight bits index it; i << 4 is the identity. Takes effect on the next
 * frame while streaming.
 */
#define V4L2_CID_QCOM_CAMSS_GAMMA_LUT	(V4L2_CID_USER_QCOM_CAMSS_BASE + 0)
#define QCOM_CAMSS_GAMMA_LUT_SIZE	256
#define QCOM_CAMSS_GAMMA_LUT_MAX	4095

/*
 * Colour correction matrix: twelve Q10 signed values, the 3x3 matrix in
 * row-major RGB order followed by the three output offsets:
 *
 *   R' = m[0] R + m[1] G + m[2] B + m[9]
 *   G' = m[3] R + m[4] G + m[5] B + m[10]
 *   B' = m[6] R + m[7] G + m[8] B + m[11]
 *
 * 1024 is unity; the hardware keeps 12 significant bits, so a matrix with
 * a coefficient beyond +-2.0 loses a bit of precision in all of them.
 * The offsets are in 12-bit pixel units.
 */
#define V4L2_CID_QCOM_CAMSS_CCM		(V4L2_CID_USER_QCOM_CAMSS_BASE + 1)
#define QCOM_CAMSS_CCM_SIZE		12
#define QCOM_CAMSS_CCM_UNITY		1024

#endif /* __UAPI_QCOM_CAMSS_CONTROLS_H */
