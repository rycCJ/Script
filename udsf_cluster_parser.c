#ifndef __UDSF_INFOHANFD_CLUSTER_PARSE_C__
#define __UDSF_INFOHANFD_CLUSTER_PARSE_C__

#include "udf/udsf/infohand/normal/udsf_simple_dbscan.c"

static void udsf_target_parser_manager(struct epc *p_epc, struct targetInfo *targets, uint8_t *count, filterClusterConf *filter) __attribute__((unused));
static int udsf_cluster_target(struct targetInfo *info, uint16_t range_res_mm, int vel_res_mm, int vel_scale, int angle_scale, uint8_t dtype, uint32_t rangeIdx, int32_t velIdx, int32_t angleIdx) __attribute__((unused));

/* 解析点云目标，转换实际的距离、角度和速度参数 */
static int udsf_cluster_target(struct targetInfo *info, uint16_t range_res_mm, int vel_res_mm, int vel_scale, int angle_scale, uint8_t dtype, uint32_t rangeIdx, int32_t velIdx, int32_t angleIdx)
{
    if (info == NULL) { return -1; }

    if ((dtype == 0x01) || (dtype == 0x02)) {
        float scale_factor = dtype == 0x01 ? 4096.0f : 409600.0f;
        info->range = (int16_t)(((float)rangeIdx / scale_factor) * (range_res_mm / 10.0f));     /* 除10是将mm转为cm */
        info->velocity = (int16_t)(((float)velIdx / scale_factor) * (vel_res_mm / 10.0f));      /* 除10是将mm转为cm */
        info->angle = (int16_t)(CSTD_ASINF(((float)angleIdx / scale_factor) / (angle_scale / 2)) * MATH_PI_D / MATH_M_PI);
    } else {
        info->range = (int16_t)(((float)rangeIdx * range_res_mm) / 10.0f);                      /* 除10是将mm转为cm */
        if (velIdx < (vel_scale / 2)) {
            info->velocity = (int16_t)(((float)velIdx * vel_res_mm) / 10.0f);                   /* 除10是将mm转为cm */
        } else {
            info->velocity = (int16_t)((((float)velIdx - vel_scale) * vel_res_mm) / 10.0f);     /* 除10是将mm转为cm */
        }

        if (angleIdx < (angle_scale / 2)) {
            info->angle = (int16_t)(CSTD_ASINF((float)angleIdx / (angle_scale / 2.0f)) * MATH_PI_D / MATH_M_PI);
        } else {
            info->angle = (int16_t)(CSTD_ASINF(((float)angleIdx - (float)angle_scale) / ((float)angle_scale / 2.0f)) * MATH_PI_D / MATH_M_PI);
        }
    }

    return 0;
}

/* 解析得到点云目标信息 */
static void udsf_target_parser_manager(struct epc *p_epc, struct targetInfo *targets, uint8_t *count, filterClusterConf *filter)
{
    if ((targets == NULL) || (p_epc == NULL) || (p_epc->p_target == NULL) || (filter == NULL)) {
        return;
    }

    uint8_t targetCnt = 0;
    int16_t angle_scale = 0;
    int16_t vel_res_mm_s = 0;
    uint16_t target_count = 0;
    uint16_t range_res_mm = 0;
    uint8_t velocity_scale = 0;
    struct epc_target *p_epc_target = p_epc->p_target;

    angle_scale = p_epc->angleMax;
    velocity_scale = p_epc->chirpMax;
    vel_res_mm_s = p_epc->vel_res_mm_s;
    range_res_mm = p_epc->range_res_mm;
    target_count = p_epc->target_cnt;

    if (target_count > MAX_CLUSTERS) {
        target_count = MAX_CLUSTERS;
    }

    for (uint8_t i = 0; i < target_count; i++) {
        uint8_t info1 = p_epc_target[i].info1;
        uint8_t info2 = p_epc_target[i].info2;

        uint8_t dtype = (info1 >> 3) & 0x07;
        uint32_t amplitude = p_epc_target[i].pow;
        uint32_t rangeIdx = p_epc_target[i].rIdx;
        int32_t velocityIdx = p_epc_target[i].vIdx;
        int32_t angleIdx = p_epc_target[i].sinPhiIdx;

        struct targetInfo objInfo;
        objInfo.range = 0;
        if (udsf_cluster_target(&objInfo, range_res_mm, vel_res_mm_s, velocity_scale, angle_scale, dtype, rangeIdx, velocityIdx, angleIdx) != 0) {
            continue;
        }

        if ((targetCnt < MAX_CLUSTERS) && (objInfo.range > 0) && (objInfo.angle >= filter->minAngle) && (objInfo.angle <= filter->maxAngle)) {
            uint8_t state = (info1 >> 6) & 0x03;
            if ((filter->rmMovept != 0) && ((state == 0x00) || (state == 0x01))) {
                continue;
            }

            targets[targetCnt].info1 = info1;
            targets[targetCnt].info2 = info2;
            targets[targetCnt].range = objInfo.range;
            if (objInfo.angle > 0) {
                targets[targetCnt].angle = objInfo.angle + filter->offAngP;
            } else if (objInfo.angle < 0) {
                targets[targetCnt].angle = objInfo.angle + filter->offAngN;
            }
            targets[targetCnt].velocity = objInfo.velocity;
            targets[targetCnt].amplitude = amplitude;
            targetCnt++;
        }
    }

    /* 实际点云数 */
    if (count) {
        *count = targetCnt;
    }
}

#endif
