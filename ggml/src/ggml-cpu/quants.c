#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"

#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 ggml_int128_t;
__extension__ typedef unsigned __int128 ggml_uint128_t;
#endif

static int ggml_gptq2_32_qnn_compat_enabled(void) {
    static int enabled = -1;

    int cached = __atomic_load_n(&enabled, __ATOMIC_ACQUIRE);
    if (cached >= 0) {
        return cached;
    }

    const char * const value = getenv("GGML_GPTQ2_QNN_COMPAT");
    const int configured = value != NULL && value[0] != 0 && strcmp(value, "0") != 0;
    int expected = -1;
    if (!__atomic_compare_exchange_n(
            &enabled, &expected, configured, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {
        return expected;
    }

    return configured;
}

int ggml_gptq2_32_gs32_dotprod_enabled(void) {
#if defined(__aarch64__) && defined(__linux__) && defined(__clang__) && \
    defined(HWCAP_ASIMDDP)
    static int enabled = -1;
    int cached = __atomic_load_n(&enabled, __ATOMIC_ACQUIRE);
    if (cached >= 0) {
        return cached;
    }

    const char * const value = getenv("GGML_GPTQ2_GS32_DOTPROD");
    const int disabled =
        value != NULL && (strcmp(value, "0") == 0 ||
            strcmp(value, "off") == 0 || strcmp(value, "false") == 0);
    const int detected =
        !disabled && (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    int expected = -1;
    if (!__atomic_compare_exchange_n(
            &enabled, &expected, detected, false,
            __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {
        return expected;
    }
    return detected;
#else
    return 0;
#endif
}

int ggml_qnn_u16_dotprod_enabled(void) {
#if defined(__aarch64__) && defined(__linux__) && defined(__clang__) && \
    defined(HWCAP_ASIMDDP)
    static int enabled = -1;
    int cached = __atomic_load_n(&enabled, __ATOMIC_ACQUIRE);
    if (cached >= 0) {
        return cached;
    }
    const char * const value = getenv("GGML_QNN_U16_DOTPROD");
    const int requested =
        value != NULL && (strcmp(value, "1") == 0 ||
            strcmp(value, "on") == 0 || strcmp(value, "true") == 0);
    const int detected =
        requested && (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    int expected = -1;
    if (!__atomic_compare_exchange_n(
            &enabled, &expected, detected, false,
            __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {
        return expected;
    }
    return detected;
#else
    return 0;
#endif
}

void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            const int8_t  * GGML_RESTRICT qy   = yb->qs;

            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}


void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

static void ggml_vec_dot_gptq2_f32(
        int n,
        float * GGML_RESTRICT s,
        size_t bs,
        const void * GGML_RESTRICT vx,
        size_t bx,
        const void * GGML_RESTRICT vy,
        size_t by,
        int nrc,
        int group_size) {
    assert(group_size > 0 && group_size % 4 == 0);
    assert(n % group_size == 0);
    assert(nrc == 1);
    assert(n == (int) bx);
    assert(n == (int) by);
    UNUSED(nrc);
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);

    const uint8_t * GGML_RESTRICT x = vx;
    const float   * GGML_RESTRICT y = vy;
    const int code_bytes = group_size / 4;
    const int block_bytes = code_bytes + 4;
    const int nb = n / group_size;
    float sumf = 0.0f;
    const int qnn_compat_enabled = ggml_gptq2_32_qnn_compat_enabled();

    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t * block = x + ib * block_bytes;
        const float raw_scale = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(block + code_bytes));
        const float zero_bias = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(block + code_bytes + 2));

        if (qnn_compat_enabled) {
            const float scale = fmaxf(raw_scale, 1.0e-4f);
            int zero_point = (int) lroundf(zero_bias / scale);
            zero_point = zero_point < 0 ? 0 : (zero_point > 3 ? 3 : zero_point);

            for (int j = 0; j < code_bytes; ++j) {
                const uint8_t packed = block[j];

                sumf += scale * (((packed >> 0) & 0x3) - zero_point) * y[ib * group_size + 4*j + 0];
                sumf += scale * (((packed >> 2) & 0x3) - zero_point) * y[ib * group_size + 4*j + 1];
                sumf += scale * (((packed >> 4) & 0x3) - zero_point) * y[ib * group_size + 4*j + 2];
                sumf += scale * (((packed >> 6) & 0x3) - zero_point) * y[ib * group_size + 4*j + 3];
            }
            continue;
        }

        const float scale = raw_scale;

        for (int j = 0; j < code_bytes; ++j) {
            const uint8_t packed = block[j];

            sumf += (scale * ((packed >> 0) & 0x3) - zero_bias) * y[ib * group_size + 4*j + 0];
            sumf += (scale * ((packed >> 2) & 0x3) - zero_bias) * y[ib * group_size + 4*j + 1];
            sumf += (scale * ((packed >> 4) & 0x3) - zero_bias) * y[ib * group_size + 4*j + 2];
            sumf += (scale * ((packed >> 6) & 0x3) - zero_bias) * y[ib * group_size + 4*j + 3];
        }
    }

    *s = sumf;
}

void ggml_vec_dot_gptq2_32_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_gptq2_f32(n, s, bs, vx, bx, vy, by, nrc, 32);
}

void ggml_vec_dot_gptq2_64_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_gptq2_f32(n, s, bs, vx, bx, vy, by, nrc, 64);
}

void ggml_vec_dot_gptq2_128_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_gptq2_f32(n, s, bs, vx, bx, vy, by, nrc, 128);
}

static inline int64_t ggml_gptq2_32_round_shift_away_from_zero(
        int64_t value,
        int shift) {
    const int64_t half = INT64_C(1) << (shift - 1);
    return value >= 0 ? (value + half) >> shift : -(((-value) + half) >> shift);
}

static inline int64_t ggml_u16_floor_shift(int64_t value, int shift) {
    GGML_ASSERT(shift > 0 && shift < 63);
    if (value >= 0) {
        return value >> shift;
    }
    const uint64_t magnitude = (uint64_t) (-(value + 1)) + 1;
    return -(int64_t) ((magnitude + ((UINT64_C(1) << shift) - 1)) >> shift);
}

static inline int64_t ggml_u16_htp_round_shift(int64_t value, int shift) {
    GGML_ASSERT(shift > 0 && shift < 63);
    return ggml_u16_floor_shift(value + (INT64_C(1) << (shift - 1)), shift);
}

static inline int64_t ggml_u16_floor_shift_with_q7_bias(
        int64_t value, int shift, int32_t bias_q7) {
    GGML_ASSERT(shift > 0 && shift < 63);
    GGML_ASSERT(bias_q7 >= -256 && bias_q7 <= 256);
    if (bias_q7 == 0) {
        return ggml_u16_floor_shift(value, shift);
    }
    const ggml_int128_t denominator = (ggml_int128_t) 1 << (shift + 7);
    const ggml_int128_t adjusted = ((ggml_int128_t) value << 7) +
        ((ggml_int128_t) bias_q7 << shift);
    return adjusted >= 0
        ? (int64_t) (adjusted / denominator)
        : -(int64_t) ((-adjusted + denominator - 1) / denominator);
}

static inline int64_t ggml_gptq2_32_dot_u16_scalar(
        const uint16_t * GGML_RESTRICT activations,
        int32_t activation_zero_point,
        const uint8_t * GGML_RESTRICT packed_codes,
        int32_t weight_zero_point) {
    int64_t dot = 0;
    for (int index = 0; index < 32; ++index) {
        const int32_t activation = (int32_t) activations[index] - activation_zero_point;
        const int32_t code = (packed_codes[index >> 2] >> ((index & 3) * 2)) & 0x3;
        dot += (int64_t) activation * (code - weight_zero_point);
    }
    return dot;
}

static inline int32_t ggml_gptq2_32_weight_sum_scalar(
        const uint8_t * GGML_RESTRICT packed_codes,
        int32_t weight_zero_point) {
    int32_t sum = -32 * weight_zero_point;
    for (int index = 0; index < 8; ++index) {
        const uint8_t packed = packed_codes[index];
        sum += ((packed >> 0) & 0x3) +
            ((packed >> 2) & 0x3) +
            ((packed >> 4) & 0x3) +
            ((packed >> 6) & 0x3);
    }
    return sum;
}

static inline int64_t ggml_qnn_a16s8_reduce_accumulator(
        int64_t centered_dot,
        int64_t expanded_weight_sum,
        int32_t activation_zero_point) {
    const int64_t zero_point_correction =
        (int64_t) activation_zero_point * expanded_weight_sum;
    const int64_t raw_dot = centered_dot + zero_point_correction;
    return (
        ggml_u16_floor_shift(raw_dot, 8) -
        ggml_u16_htp_round_shift(zero_point_correction, 8)) << 8;
}

#if defined(__ARM_NEON)
static inline int32_t ggml_gptq2_32_hsum_s32(int32x4_t value) {
#if defined(__aarch64__)
    return vaddvq_s32(value);
#else
    const int64x2_t pairwise = vpaddlq_s32(value);
    return (int32_t) (vgetq_lane_s64(pairwise, 0) + vgetq_lane_s64(pairwise, 1));
#endif
}

#if defined(__aarch64__)
static inline int32_t ggml_gptq2_32_hsum_pair_s32(
        int32x4_t lo,
        int32x4_t hi) {
    return vaddvq_s32(vaddq_s32(lo, hi));
}
#endif

static inline int64_t ggml_gptq2_32_dot_u16_neon(
        const uint16_t * GGML_RESTRICT activations,
        int32_t activation_zero_point,
        const uint8_t * GGML_RESTRICT packed_codes,
        int32_t weight_zero_point) {
    // vld4q maps activation indices [0, 4, ..., 28] through [3, 7, ..., 31]
    // to the four INT2 positions within each packed source byte.  The code
    // stream is widened from INT2 to QNN INT4 semantics only in registers.
    const uint16x8x4_t activation_lanes = vld4q_u16(activations);
    const uint8x8_t packed = vld1_u8(packed_codes);
    const uint8x8_t mask = vdup_n_u8(0x3);
    const uint8x8_t code_lanes[4] = {
        vand_u8(packed, mask),
        vand_u8(vshr_n_u8(packed, 2), mask),
        vand_u8(vshr_n_u8(packed, 4), mask),
        vand_u8(vshr_n_u8(packed, 6), mask),
    };
#if defined(__aarch64__)
    uint16x8_t activation_min = activation_lanes.val[0];
    uint16x8_t activation_max = activation_lanes.val[0];
    for (int lane = 1; lane < 4; ++lane) {
        activation_min = vminq_u16(activation_min, activation_lanes.val[lane]);
        activation_max = vmaxq_u16(activation_max, activation_lanes.val[lane]);
    }
    const uint16_t lower = activation_zero_point > 32768
        ? (uint16_t) (activation_zero_point - 32768)
        : 0;
    const uint16_t upper = activation_zero_point < 32768
        ? (uint16_t) (activation_zero_point + 32767)
        : UINT16_MAX;
    if (vminvq_u16(activation_min) >= lower &&
        vmaxvq_u16(activation_max) <= upper) {
        const uint16x8_t activation_zero_u16 =
            vdupq_n_u16((uint16_t) activation_zero_point);
        const int16x8_t weight_zero_i16 =
            vdupq_n_s16((int16_t) weight_zero_point);
        int32x4_t accumulator_lo = vdupq_n_s32(0);
        int32x4_t accumulator_hi = vdupq_n_s32(0);
        for (int lane = 0; lane < 4; ++lane) {
            const int16x8_t activation_i16 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[lane], activation_zero_u16));
            const int16x8_t weight_i16 = vsubq_s16(
                vmovl_s8(vreinterpret_s8_u8(code_lanes[lane])),
                weight_zero_i16);
            accumulator_lo = vmlal_s16(
                accumulator_lo,
                vget_low_s16(activation_i16),
                vget_low_s16(weight_i16));
            accumulator_hi = vmlal_s16(
                accumulator_hi,
                vget_high_s16(activation_i16),
                vget_high_s16(weight_i16));
        }
        return
            (int64_t) ggml_gptq2_32_hsum_s32(accumulator_lo) +
            ggml_gptq2_32_hsum_s32(accumulator_hi);
    }
#endif
    const int32x4_t activation_zero = vdupq_n_s32(activation_zero_point);
    const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
    int32x4_t accumulator_lo = vdupq_n_s32(0);
    int32x4_t accumulator_hi = vdupq_n_s32(0);
    for (int lane = 0; lane < 4; ++lane) {
        const int32x4_t activation_lo = vreinterpretq_s32_u32(
            vmovl_u16(vget_low_u16(activation_lanes.val[lane])));
        const int32x4_t activation_hi = vreinterpretq_s32_u32(
            vmovl_u16(vget_high_u16(activation_lanes.val[lane])));
        const int8x8_t code_as_int4 = vreinterpret_s8_u8(code_lanes[lane]);
        const int16x8_t weight_i16 = vmovl_s8(code_as_int4);
        const int32x4_t weight_lo = vmovl_s16(vget_low_s16(weight_i16));
        const int32x4_t weight_hi = vmovl_s16(vget_high_s16(weight_i16));
        accumulator_lo = vmlaq_s32(
            accumulator_lo,
            vsubq_s32(activation_lo, activation_zero),
            vsubq_s32(weight_lo, weight_zero));
        accumulator_hi = vmlaq_s32(
            accumulator_hi,
            vsubq_s32(activation_hi, activation_zero),
            vsubq_s32(weight_hi, weight_zero));
    }
    return
        (int64_t) ggml_gptq2_32_hsum_s32(accumulator_lo) +
        ggml_gptq2_32_hsum_s32(accumulator_hi);
}

static inline void ggml_gptq2_32_dot_s32_neon_4rows_regs(
        const int32x4_t activation_lo[4],
        const int32x4_t activation_hi[4],
        int32_t activation_sum,
        uint8x8_t packed0,
        uint8x8_t packed1,
        uint8x8_t packed2,
        uint8x8_t packed3,
        int32_t weight_zero_point0,
        int32_t weight_zero_point1,
        int32_t weight_zero_point2,
        int32_t weight_zero_point3,
        int64_t * dot0,
        int64_t * dot1,
        int64_t * dot2,
        int64_t * dot3) {
    const uint8x8_t mask = vdup_n_u8(0x3);
    int32x4_t accumulator0_lo = vdupq_n_s32(0);
    int32x4_t accumulator0_hi = vdupq_n_s32(0);
    int32x4_t accumulator1_lo = vdupq_n_s32(0);
    int32x4_t accumulator1_hi = vdupq_n_s32(0);
    int32x4_t accumulator2_lo = vdupq_n_s32(0);
    int32x4_t accumulator2_hi = vdupq_n_s32(0);
    int32x4_t accumulator3_lo = vdupq_n_s32(0);
    int32x4_t accumulator3_hi = vdupq_n_s32(0);
#define GGML_GPTQ2_WIDE_ACCUMULATE(LANE, CODES0, CODES1, CODES2, CODES3) do { \
            const int16x8_t weight0 = vreinterpretq_s16_u16( \
                vmovl_u8((CODES0))); \
            const int16x8_t weight1 = vreinterpretq_s16_u16( \
                vmovl_u8((CODES1))); \
            const int16x8_t weight2 = vreinterpretq_s16_u16( \
                vmovl_u8((CODES2))); \
            const int16x8_t weight3 = vreinterpretq_s16_u16( \
                vmovl_u8((CODES3))); \
            accumulator0_lo = vmlaq_s32( \
                accumulator0_lo, activation_lo[(LANE)], \
                vmovl_s16(vget_low_s16(weight0))); \
            accumulator0_hi = vmlaq_s32( \
                accumulator0_hi, activation_hi[(LANE)], \
                vmovl_s16(vget_high_s16(weight0))); \
            accumulator1_lo = vmlaq_s32( \
                accumulator1_lo, activation_lo[(LANE)], \
                vmovl_s16(vget_low_s16(weight1))); \
            accumulator1_hi = vmlaq_s32( \
                accumulator1_hi, activation_hi[(LANE)], \
                vmovl_s16(vget_high_s16(weight1))); \
            accumulator2_lo = vmlaq_s32( \
                accumulator2_lo, activation_lo[(LANE)], \
                vmovl_s16(vget_low_s16(weight2))); \
            accumulator2_hi = vmlaq_s32( \
                accumulator2_hi, activation_hi[(LANE)], \
                vmovl_s16(vget_high_s16(weight2))); \
            accumulator3_lo = vmlaq_s32( \
                accumulator3_lo, activation_lo[(LANE)], \
                vmovl_s16(vget_low_s16(weight3))); \
            accumulator3_hi = vmlaq_s32( \
                accumulator3_hi, activation_hi[(LANE)], \
                vmovl_s16(vget_high_s16(weight3))); \
        } while (0)
    GGML_GPTQ2_WIDE_ACCUMULATE(
        0,
        vand_u8(packed0, mask), vand_u8(packed1, mask),
        vand_u8(packed2, mask), vand_u8(packed3, mask));
    GGML_GPTQ2_WIDE_ACCUMULATE(
        1,
        vand_u8(vshr_n_u8(packed0, 2), mask),
        vand_u8(vshr_n_u8(packed1, 2), mask),
        vand_u8(vshr_n_u8(packed2, 2), mask),
        vand_u8(vshr_n_u8(packed3, 2), mask));
    GGML_GPTQ2_WIDE_ACCUMULATE(
        2,
        vand_u8(vshr_n_u8(packed0, 4), mask),
        vand_u8(vshr_n_u8(packed1, 4), mask),
        vand_u8(vshr_n_u8(packed2, 4), mask),
        vand_u8(vshr_n_u8(packed3, 4), mask));
    GGML_GPTQ2_WIDE_ACCUMULATE(
        3,
        vshr_n_u8(packed0, 6), vshr_n_u8(packed1, 6),
        vshr_n_u8(packed2, 6), vshr_n_u8(packed3, 6));
#undef GGML_GPTQ2_WIDE_ACCUMULATE
    *dot0 =
        (int64_t) ggml_gptq2_32_hsum_pair_s32(
            accumulator0_lo, accumulator0_hi) -
        (int64_t) weight_zero_point0 * activation_sum;
    *dot1 =
        (int64_t) ggml_gptq2_32_hsum_pair_s32(
            accumulator1_lo, accumulator1_hi) -
        (int64_t) weight_zero_point1 * activation_sum;
    *dot2 =
        (int64_t) ggml_gptq2_32_hsum_pair_s32(
            accumulator2_lo, accumulator2_hi) -
        (int64_t) weight_zero_point2 * activation_sum;
    *dot3 =
        (int64_t) ggml_gptq2_32_hsum_pair_s32(
            accumulator3_lo, accumulator3_hi) -
        (int64_t) weight_zero_point3 * activation_sum;
}

static __attribute__((noinline)) void ggml_gptq2_32_dot_u16_neon_4rows_wide(
        const uint16_t * GGML_RESTRICT activations,
        int32_t activation_zero_point,
        int32_t activation_sum,
        const uint8_t * packed_codes0,
        const uint8_t * packed_codes1,
        const uint8_t * packed_codes2,
        const uint8_t * packed_codes3,
        int32_t weight_zero_point0,
        int32_t weight_zero_point1,
        int32_t weight_zero_point2,
        int32_t weight_zero_point3,
        int64_t * dot0,
        int64_t * dot1,
        int64_t * dot2,
        int64_t * dot3) {
    const uint16x8x4_t lanes = vld4q_u16(activations);
    const int32x4_t zero = vdupq_n_s32(activation_zero_point);
    int32x4_t activation_lo[4];
    int32x4_t activation_hi[4];
    for (int lane = 0; lane < 4; ++lane) {
        activation_lo[lane] = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lanes.val[lane]))),
            zero);
        activation_hi[lane] = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lanes.val[lane]))),
            zero);
    }
    ggml_gptq2_32_dot_s32_neon_4rows_regs(
        activation_lo, activation_hi, activation_sum,
        vld1_u8(packed_codes0), vld1_u8(packed_codes1),
        vld1_u8(packed_codes2), vld1_u8(packed_codes3),
        weight_zero_point0, weight_zero_point1,
        weight_zero_point2, weight_zero_point3,
        dot0, dot1, dot2, dot3);
}

#if defined(__aarch64__)
static inline void ggml_gptq2_32_dot_s16_neon_4rows_regs(
        int16x8_t activation0,
        int16x8_t activation1,
        int16x8_t activation2,
        int16x8_t activation3,
        int32_t activation_sum,
        uint8x8_t packed0,
        uint8x8_t packed1,
        uint8x8_t packed2,
        uint8x8_t packed3,
        int32_t weight_zero_point0,
        int32_t weight_zero_point1,
        int32_t weight_zero_point2,
        int32_t weight_zero_point3,
        int64_t * dot0,
        int64_t * dot1,
        int64_t * dot2,
        int64_t * dot3) {
    const uint8x8_t mask = vdup_n_u8(0x3);
    int32x4_t accumulator0_lo = vdupq_n_s32(0);
    int32x4_t accumulator0_hi = vdupq_n_s32(0);
    int32x4_t accumulator1_lo = vdupq_n_s32(0);
    int32x4_t accumulator1_hi = vdupq_n_s32(0);
    int32x4_t accumulator2_lo = vdupq_n_s32(0);
    int32x4_t accumulator2_hi = vdupq_n_s32(0);
    int32x4_t accumulator3_lo = vdupq_n_s32(0);
    int32x4_t accumulator3_hi = vdupq_n_s32(0);
#define GGML_GPTQ2_ACCUMULATE_4ROWS(ACTIVATION, CODES0, CODES1, CODES2, CODES3) do { \
            const int16x8_t weight0 = vmovl_s8(vreinterpret_s8_u8((CODES0))); \
            const int16x8_t weight1 = vmovl_s8(vreinterpret_s8_u8((CODES1))); \
            const int16x8_t weight2 = vmovl_s8(vreinterpret_s8_u8((CODES2))); \
            const int16x8_t weight3 = vmovl_s8(vreinterpret_s8_u8((CODES3))); \
            accumulator0_lo = vmlal_s16(accumulator0_lo, \
                vget_low_s16((ACTIVATION)), vget_low_s16(weight0)); \
            accumulator0_hi = vmlal_s16(accumulator0_hi, \
                vget_high_s16((ACTIVATION)), vget_high_s16(weight0)); \
            accumulator1_lo = vmlal_s16(accumulator1_lo, \
                vget_low_s16((ACTIVATION)), vget_low_s16(weight1)); \
            accumulator1_hi = vmlal_s16(accumulator1_hi, \
                vget_high_s16((ACTIVATION)), vget_high_s16(weight1)); \
            accumulator2_lo = vmlal_s16(accumulator2_lo, \
                vget_low_s16((ACTIVATION)), vget_low_s16(weight2)); \
            accumulator2_hi = vmlal_s16(accumulator2_hi, \
                vget_high_s16((ACTIVATION)), vget_high_s16(weight2)); \
            accumulator3_lo = vmlal_s16(accumulator3_lo, \
                vget_low_s16((ACTIVATION)), vget_low_s16(weight3)); \
            accumulator3_hi = vmlal_s16(accumulator3_hi, \
                vget_high_s16((ACTIVATION)), vget_high_s16(weight3)); \
        } while (0)
    GGML_GPTQ2_ACCUMULATE_4ROWS(
        activation0,
        vand_u8(packed0, mask), vand_u8(packed1, mask),
        vand_u8(packed2, mask), vand_u8(packed3, mask));
    GGML_GPTQ2_ACCUMULATE_4ROWS(
        activation1,
        vand_u8(vshr_n_u8(packed0, 2), mask),
        vand_u8(vshr_n_u8(packed1, 2), mask),
        vand_u8(vshr_n_u8(packed2, 2), mask),
        vand_u8(vshr_n_u8(packed3, 2), mask));
    GGML_GPTQ2_ACCUMULATE_4ROWS(
        activation2,
        vand_u8(vshr_n_u8(packed0, 4), mask),
        vand_u8(vshr_n_u8(packed1, 4), mask),
        vand_u8(vshr_n_u8(packed2, 4), mask),
        vand_u8(vshr_n_u8(packed3, 4), mask));
    GGML_GPTQ2_ACCUMULATE_4ROWS(
        activation3,
        vand_u8(vshr_n_u8(packed0, 6), mask),
        vand_u8(vshr_n_u8(packed1, 6), mask),
        vand_u8(vshr_n_u8(packed2, 6), mask),
        vand_u8(vshr_n_u8(packed3, 6), mask));
#undef GGML_GPTQ2_ACCUMULATE_4ROWS
    *dot0 = (int64_t) ggml_gptq2_32_hsum_pair_s32(
        accumulator0_lo, accumulator0_hi) -
        (int64_t) weight_zero_point0 * activation_sum;
    *dot1 = (int64_t) ggml_gptq2_32_hsum_pair_s32(
        accumulator1_lo, accumulator1_hi) -
        (int64_t) weight_zero_point1 * activation_sum;
    *dot2 = (int64_t) ggml_gptq2_32_hsum_pair_s32(
        accumulator2_lo, accumulator2_hi) -
        (int64_t) weight_zero_point2 * activation_sum;
    *dot3 = (int64_t) ggml_gptq2_32_hsum_pair_s32(
        accumulator3_lo, accumulator3_hi) -
        (int64_t) weight_zero_point3 * activation_sum;
}

static inline void ggml_gptq2_32_dot_s16_neon_4rows(
        int16x8_t activation0,
        int16x8_t activation1,
        int16x8_t activation2,
        int16x8_t activation3,
        int32_t activation_sum,
        const uint8_t * packed_codes0,
        const uint8_t * packed_codes1,
        const uint8_t * packed_codes2,
        const uint8_t * packed_codes3,
        int32_t weight_zero_point0,
        int32_t weight_zero_point1,
        int32_t weight_zero_point2,
        int32_t weight_zero_point3,
        int64_t * dot0,
        int64_t * dot1,
        int64_t * dot2,
        int64_t * dot3) {
    ggml_gptq2_32_dot_s16_neon_4rows_regs(
        activation0, activation1, activation2, activation3, activation_sum,
        vld1_u8(packed_codes0), vld1_u8(packed_codes1),
        vld1_u8(packed_codes2), vld1_u8(packed_codes3),
        weight_zero_point0, weight_zero_point1,
        weight_zero_point2, weight_zero_point3,
        dot0, dot1, dot2, dot3);
}

// Consume one native gs32_source_v1 qcode group without reconstructing eight
// row-major 8-byte code streams. Each packed pair contains two adjacent
// qbytes for all eight rows. Keeping that pair-major order lets one unpack
// feed all rows and reduces the live accumulator set from 16 vectors to four.
static inline void ggml_gptq2_32_dot_s16_neon_gs32_8rows(
        int16x8_t activation0,
        int16x8_t activation1,
        int16x8_t activation2,
        int16x8_t activation3,
        int32_t activation_sum,
        uint8x16_t packed_pair0,
        uint8x16_t packed_pair1,
        uint8x16_t packed_pair2,
        uint8x16_t packed_pair3,
        const int32_t weight_zero_points[8],
        int64_t dots[8]) {
    const uint8x16_t mask = vdupq_n_u8(0x3);
    int32x4_t accumulator01 = vdupq_n_s32(0);
    int32x4_t accumulator23 = vdupq_n_s32(0);
    int32x4_t accumulator45 = vdupq_n_s32(0);
    int32x4_t accumulator67 = vdupq_n_s32(0);

#define GGML_GPTQ2_GS32_ACCUMULATE(CODES, ACTIVATION, PAIR) do { \
            const int16x8_t codes_lo = vmovl_s8( \
                vreinterpret_s8_u8(vget_low_u8((CODES)))); \
            const int16x8_t codes_hi = vmovl_s8( \
                vreinterpret_s8_u8(vget_high_u8((CODES)))); \
            const int16x8_t activation_pair = vreinterpretq_s16_s32( \
                vdupq_laneq_s32( \
                    vreinterpretq_s32_s16((ACTIVATION)), (PAIR))); \
            accumulator01 = vmlal_s16( \
                accumulator01, vget_low_s16(codes_lo), \
                vget_low_s16(activation_pair)); \
            accumulator23 = vmlal_s16( \
                accumulator23, vget_high_s16(codes_lo), \
                vget_high_s16(activation_pair)); \
            accumulator45 = vmlal_s16( \
                accumulator45, vget_low_s16(codes_hi), \
                vget_low_s16(activation_pair)); \
            accumulator67 = vmlal_s16( \
                accumulator67, vget_high_s16(codes_hi), \
                vget_high_s16(activation_pair)); \
        } while (0)
#define GGML_GPTQ2_GS32_ACCUMULATE_PAIR(PACKED, PAIR) do { \
            GGML_GPTQ2_GS32_ACCUMULATE( \
                vandq_u8((PACKED), mask), activation0, (PAIR)); \
            GGML_GPTQ2_GS32_ACCUMULATE( \
                vandq_u8(vshrq_n_u8((PACKED), 2), mask), \
                activation1, (PAIR)); \
            GGML_GPTQ2_GS32_ACCUMULATE( \
                vandq_u8(vshrq_n_u8((PACKED), 4), mask), \
                activation2, (PAIR)); \
            GGML_GPTQ2_GS32_ACCUMULATE( \
                vshrq_n_u8((PACKED), 6), activation3, (PAIR)); \
        } while (0)
    GGML_GPTQ2_GS32_ACCUMULATE_PAIR(packed_pair0, 0);
    GGML_GPTQ2_GS32_ACCUMULATE_PAIR(packed_pair1, 1);
    GGML_GPTQ2_GS32_ACCUMULATE_PAIR(packed_pair2, 2);
    GGML_GPTQ2_GS32_ACCUMULATE_PAIR(packed_pair3, 3);
#undef GGML_GPTQ2_GS32_ACCUMULATE_PAIR
#undef GGML_GPTQ2_GS32_ACCUMULATE

    const int32x4_t dot03 = vpaddq_s32(accumulator01, accumulator23);
    const int32x4_t dot47 = vpaddq_s32(accumulator45, accumulator67);
#define GGML_GPTQ2_GS32_FINISH(ROW, VECTOR, LANE) \
    dots[(ROW)] = (int64_t) vgetq_lane_s32((VECTOR), (LANE)) - \
        (int64_t) weight_zero_points[(ROW)] * activation_sum
    GGML_GPTQ2_GS32_FINISH(0, dot03, 0);
    GGML_GPTQ2_GS32_FINISH(1, dot03, 1);
    GGML_GPTQ2_GS32_FINISH(2, dot03, 2);
    GGML_GPTQ2_GS32_FINISH(3, dot03, 3);
    GGML_GPTQ2_GS32_FINISH(4, dot47, 0);
    GGML_GPTQ2_GS32_FINISH(5, dot47, 1);
    GGML_GPTQ2_GS32_FINISH(6, dot47, 2);
    GGML_GPTQ2_GS32_FINISH(7, dot47, 3);
#undef GGML_GPTQ2_GS32_FINISH
}

#if defined(__clang__)
__attribute__((target("dotprod")))
static inline void ggml_gptq2_32_dot_s16_dotprod_gs32_8rows(
        uint8x8_t activation_low0,
        uint8x8_t activation_low1,
        uint8x8_t activation_low2,
        uint8x8_t activation_low3,
        int8x8_t activation_high0,
        int8x8_t activation_high1,
        int8x8_t activation_high2,
        int8x8_t activation_high3,
        uint8x16_t packed_pair0,
        uint8x16_t packed_pair1,
        uint8x16_t packed_pair2,
        uint8x16_t packed_pair3,
        int32x4_t * dot03,
        int32x4_t * dot47) {
    const uint8x16_t mask = vdupq_n_u8(0x3);
    uint32x4_t low03 = vdupq_n_u32(0);
    uint32x4_t low47 = vdupq_n_u32(0);
    int32x4_t high03 = vdupq_n_s32(0);
    int32x4_t high47 = vdupq_n_s32(0);

#define GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE( \
        CODE0, CODE1, CODE2, CODE3, ACTIVATION_LOW, ACTIVATION_HIGH) do { \
            const uint16x8x2_t codes01 = vzipq_u16( \
                vreinterpretq_u16_u8((CODE0)), \
                vreinterpretq_u16_u8((CODE1))); \
            const uint16x8x2_t codes23 = vzipq_u16( \
                vreinterpretq_u16_u8((CODE2)), \
                vreinterpretq_u16_u8((CODE3))); \
            low03 = vdotq_lane_u32( \
                low03, vreinterpretq_u8_u16(codes01.val[0]), \
                (ACTIVATION_LOW), 0); \
            low47 = vdotq_lane_u32( \
                low47, vreinterpretq_u8_u16(codes01.val[1]), \
                (ACTIVATION_LOW), 0); \
            high03 = vdotq_lane_s32( \
                high03, vreinterpretq_s8_u16(codes01.val[0]), \
                (ACTIVATION_HIGH), 0); \
            high47 = vdotq_lane_s32( \
                high47, vreinterpretq_s8_u16(codes01.val[1]), \
                (ACTIVATION_HIGH), 0); \
            low03 = vdotq_lane_u32( \
                low03, vreinterpretq_u8_u16(codes23.val[0]), \
                (ACTIVATION_LOW), 1); \
            low47 = vdotq_lane_u32( \
                low47, vreinterpretq_u8_u16(codes23.val[1]), \
                (ACTIVATION_LOW), 1); \
            high03 = vdotq_lane_s32( \
                high03, vreinterpretq_s8_u16(codes23.val[0]), \
                (ACTIVATION_HIGH), 1); \
            high47 = vdotq_lane_s32( \
                high47, vreinterpretq_s8_u16(codes23.val[1]), \
                (ACTIVATION_HIGH), 1); \
        } while (0)
    GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE(
        vandq_u8(packed_pair0, mask),
        vandq_u8(packed_pair1, mask),
        vandq_u8(packed_pair2, mask),
        vandq_u8(packed_pair3, mask),
        activation_low0, activation_high0);
    GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE(
        vandq_u8(vshrq_n_u8(packed_pair0, 2), mask),
        vandq_u8(vshrq_n_u8(packed_pair1, 2), mask),
        vandq_u8(vshrq_n_u8(packed_pair2, 2), mask),
        vandq_u8(vshrq_n_u8(packed_pair3, 2), mask),
        activation_low1, activation_high1);
    GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE(
        vandq_u8(vshrq_n_u8(packed_pair0, 4), mask),
        vandq_u8(vshrq_n_u8(packed_pair1, 4), mask),
        vandq_u8(vshrq_n_u8(packed_pair2, 4), mask),
        vandq_u8(vshrq_n_u8(packed_pair3, 4), mask),
        activation_low2, activation_high2);
    GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE(
        vshrq_n_u8(packed_pair0, 6),
        vshrq_n_u8(packed_pair1, 6),
        vshrq_n_u8(packed_pair2, 6),
        vshrq_n_u8(packed_pair3, 6),
        activation_low3, activation_high3);
#undef GGML_GPTQ2_GS32_DOTPROD_ACCUMULATE

    *dot03 = vaddq_s32(
        vreinterpretq_s32_u32(low03), vshlq_n_s32(high03, 8));
    *dot47 = vaddq_s32(
        vreinterpretq_s32_u32(low47), vshlq_n_s32(high47, 8));
}

__attribute__((target("dotprod")))
static void ggml_gptq2_32_gs32_dotprod_accumulate_blocks(
        int blocks,
        const uint8_t * GGML_RESTRICT tensor_source,
        size_t row_outer,
        size_t row_middle,
        const uint8_t * GGML_RESTRICT activation_low,
        const int8_t * GGML_RESTRICT activation_high,
        const int32_t * GGML_RESTRICT activation_block_sums,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        int32_t activation_zero_point,
        int64_t centered_dots[8]) {
    const size_t pair0_offset =
        (((row_outer * 4 + 0) * 4 + row_middle) * 8) * 2;
    const size_t pair1_offset =
        (((row_outer * 4 + 1) * 4 + row_middle) * 8) * 2;
    const size_t pair2_offset =
        (((row_outer * 4 + 2) * 4 + row_middle) * 8) * 2;
    const size_t pair3_offset =
        (((row_outer * 4 + 3) * 4 + row_middle) * 8) * 2;
    (void) activation_zero_point;

#pragma clang loop unroll_count(2)
    for (int block = 0; block < blocks; ++block) {
        const uint8_t * source_group =
            tensor_source + (size_t) block * 768;
        int32x4_t dot03;
        int32x4_t dot47;
        const uint8_t * const activation_low_block =
            activation_low + block * 32;
        const int8_t * const activation_high_block =
            activation_high + block * 32;
        const uint8x16_t packed_pair0 =
            vld1q_u8(source_group + pair0_offset);
        const uint8x16_t packed_pair1 =
            vld1q_u8(source_group + pair1_offset);
        const uint8x16_t packed_pair2 =
            vld1q_u8(source_group + pair2_offset);
        const uint8x16_t packed_pair3 =
            vld1q_u8(source_group + pair3_offset);
        ggml_gptq2_32_dot_s16_dotprod_gs32_8rows(
            vld1_u8(activation_low_block + 0),
            vld1_u8(activation_low_block + 8),
            vld1_u8(activation_low_block + 16),
            vld1_u8(activation_low_block + 24),
            vld1_s8(activation_high_block + 0),
            vld1_s8(activation_high_block + 8),
            vld1_s8(activation_high_block + 16),
            vld1_s8(activation_high_block + 24),
            packed_pair0, packed_pair1, packed_pair2, packed_pair3,
            &dot03, &dot47);

        const int32_t activation_sum = activation_block_sums[block];
#define GGML_GPTQ2_GS32_DOTPROD_FINISH(ROW, VECTOR, LANE) do { \
            const uint8_t prepared = prepared_block_codes[ \
                prepared_row_stride != 0 ? \
                    (size_t) (ROW) * prepared_row_stride + block : \
                    (size_t) block * 8 + (ROW)]; \
            centered_dots[(ROW)] += \
                ((int64_t) vgetq_lane_s32((VECTOR), (LANE)) - \
                    (int64_t) ((prepared >> 5) & 0x3) * activation_sum) * \
                (prepared & 0x1f); \
        } while (0)
        GGML_GPTQ2_GS32_DOTPROD_FINISH(0, dot03, 0);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(1, dot03, 1);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(2, dot03, 2);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(3, dot03, 3);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(4, dot47, 0);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(5, dot47, 1);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(6, dot47, 2);
        GGML_GPTQ2_GS32_DOTPROD_FINISH(7, dot47, 3);
#undef GGML_GPTQ2_GS32_DOTPROD_FINISH
    }
}
#endif
#endif

static inline void ggml_gptq2_32_dot_u16_neon_4rows(
        const uint16_t * GGML_RESTRICT activations,
        int32_t activation_sum,
        int activations_fit_i16,
        int32_t activation_zero_point,
        const uint8_t * packed_codes0,
        const uint8_t * packed_codes1,
        const uint8_t * packed_codes2,
        const uint8_t * packed_codes3,
        int32_t weight_zero_point0,
        int32_t weight_zero_point1,
        int32_t weight_zero_point2,
        int32_t weight_zero_point3,
        int64_t * dot0,
        int64_t * dot1,
        int64_t * dot2,
        int64_t * dot3) {
#if defined(__aarch64__)
    if (activations_fit_i16) {
        const uint16x8x4_t activation_lanes = vld4q_u16(activations);
        const uint16x8_t activation_zero_u16 =
            vdupq_n_u16((uint16_t) activation_zero_point);
        ggml_gptq2_32_dot_s16_neon_4rows_regs(
            vreinterpretq_s16_u16(vsubq_u16(
                activation_lanes.val[0], activation_zero_u16)),
            vreinterpretq_s16_u16(vsubq_u16(
                activation_lanes.val[1], activation_zero_u16)),
            vreinterpretq_s16_u16(vsubq_u16(
                activation_lanes.val[2], activation_zero_u16)),
            vreinterpretq_s16_u16(vsubq_u16(
                activation_lanes.val[3], activation_zero_u16)),
            activation_sum,
            vld1_u8(packed_codes0), vld1_u8(packed_codes1),
            vld1_u8(packed_codes2), vld1_u8(packed_codes3),
            weight_zero_point0, weight_zero_point1,
            weight_zero_point2, weight_zero_point3,
            dot0, dot1, dot2, dot3);
        return;
    }
#endif
    ggml_gptq2_32_dot_u16_neon_4rows_wide(
        activations, activation_zero_point, activation_sum,
        packed_codes0, packed_codes1, packed_codes2, packed_codes3,
        weight_zero_point0, weight_zero_point1,
        weight_zero_point2, weight_zero_point3,
        dot0, dot1, dot2, dot3);
}
#endif

int ggml_gptq2_32_prepare_u16_activation(
        int n,
        const uint16_t * GGML_RESTRICT activations,
        int32_t activation_zero_point,
        int32_t * GGML_RESTRICT activation_block_sums,
        uint8_t * GGML_RESTRICT activation_low,
        int8_t * GGML_RESTRICT activation_high) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(activations != NULL);
    GGML_ASSERT(activation_block_sums != NULL);
    GGML_ASSERT((activation_low == NULL) == (activation_high == NULL));

#if defined(__ARM_NEON) && defined(__aarch64__)
    uint16x8_t activation_min = vdupq_n_u16(UINT16_MAX);
    uint16x8_t activation_max = vdupq_n_u16(0);
    const uint16x8_t zero =
        vdupq_n_u16((uint16_t) activation_zero_point);
    for (int block = 0; block < n / 32; ++block) {
        const uint16x8x4_t lanes = vld4q_u16(activations + block * 32);
        uint32_t raw_sum = 0;
        for (int group = 0; group < 4; ++group) {
            activation_min = vminq_u16(activation_min, lanes.val[group]);
            activation_max = vmaxq_u16(activation_max, lanes.val[group]);
            raw_sum += vaddlvq_u16(lanes.val[group]);
            if (activation_low != NULL) {
                const int16x8_t centered = vreinterpretq_s16_u16(
                    vsubq_u16(lanes.val[group], zero));
                vst1_u8(
                    activation_low + block * 32 + group * 8,
                    vmovn_u16(vreinterpretq_u16_s16(centered)));
                vst1_s8(
                    activation_high + block * 32 + group * 8,
                    vshrn_n_s16(centered, 8));
            }
        }
        activation_block_sums[block] =
            (int32_t) raw_sum - activation_zero_point * 32;
    }
    const uint16_t lower = activation_zero_point > 32768
        ? (uint16_t) (activation_zero_point - 32768)
        : 0;
    const uint16_t upper = activation_zero_point < 32768
        ? (uint16_t) (activation_zero_point + 32767)
        : UINT16_MAX;
    return vminvq_u16(activation_min) >= lower &&
        vmaxvq_u16(activation_max) <= upper;
#else
    int activations_fit_i16 = 1;
    for (int block = 0; block < n / 32; ++block) {
        int32_t sum = 0;
        for (int lane = 0; lane < 32; ++lane) {
            const int32_t centered =
                (int32_t) activations[block * 32 + lane] -
                activation_zero_point;
            sum += centered;
            activations_fit_i16 = activations_fit_i16 &&
                centered >= INT16_MIN && centered <= INT16_MAX;
            if (activation_low != NULL) {
                const int16_t centered_i16 = (int16_t) centered;
                const int group = lane % 4;
                const int group_lane = lane / 4;
                activation_low[block * 32 + group * 8 + group_lane] =
                    (uint8_t) centered_i16;
                activation_high[block * 32 + group * 8 + group_lane] =
                    (int8_t) (centered_i16 >> 8);
            }
        }
        activation_block_sums[block] = sum;
    }
    return activations_fit_i16;
#endif
}

static void ggml_vec_dot_gptq2_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point,
        int group_size) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(activation_scale > 0.0f);
    GGML_ASSERT(output_scale > 0.0f);

    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int code_bytes = group_size / 4;
    const int block_bytes = code_bytes + 4;
    const int groups = n / group_size;
    const int requant_shift = 20;
    int64_t scaled_accumulator = 0;

    for (int group_index = 0; group_index < groups; ++group_index) {
        const uint8_t * block = weights + group_index * block_bytes;
        const float raw_scale = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *) (block + code_bytes));
        const float zero_bias = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *) (block + code_bytes + 2));
        const float weight_scale = fmaxf(raw_scale, 1.0e-4f);
        int32_t weight_zero_point = (int32_t) lroundf(zero_bias / weight_scale);
        weight_zero_point = MAX(0, MIN(3, weight_zero_point));
        const double ratio = (double) activation_scale * (double) weight_scale / (double) output_scale;
        const int64_t multiplier = (int64_t) llround(ldexp(ratio, requant_shift));
        int64_t group_dot = 0;
        for (int chunk = 0; chunk < group_size / 32; ++chunk) {
            const uint16_t * chunk_activations =
                activations + group_index * group_size + chunk * 32;
            const uint8_t * chunk_codes = block + chunk * 8;
#if defined(__ARM_NEON)
            group_dot += ggml_gptq2_32_dot_u16_neon(
                chunk_activations, activation_zero_point, chunk_codes, weight_zero_point);
#else
            group_dot += ggml_gptq2_32_dot_u16_scalar(
                chunk_activations, activation_zero_point, chunk_codes, weight_zero_point);
#endif
        }
        scaled_accumulator += group_dot * multiplier;
    }

    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        scaled_accumulator,
        requant_shift) + output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_dot_gptq2_32_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn(
        n, output, packed_weights, activations, activation_scale,
        activation_zero_point, output_scale, output_zero_point, 32);
}

void ggml_vec_dot_gptq2_64_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn(
        n, output, packed_weights, activations, activation_scale,
        activation_zero_point, output_scale, output_zero_point, 64);
}

void ggml_vec_dot_gptq2_128_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn(
        n, output, packed_weights, activations, activation_scale,
        activation_zero_point, output_scale, output_zero_point, 128);
}

#define GGML_U16_Q20_SHIFT 20
#define GGML_U16_MIN_WEIGHT_SCALE_Q20 105

static inline int64_t ggml_gptq2_32_fp16_to_q20(ggml_fp16_t value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const int sign = (bits >> 15) & 1;
    const int exponent = (bits >> 10) & 0x1f;
    const uint64_t mantissa = bits & 0x03ff;
    uint64_t magnitude;
    if (exponent == 0) {
        magnitude = (mantissa + 8) >> 4;
    } else if (exponent == 31) {
        magnitude = INT64_MAX;
    } else {
        const uint64_t significand = 1024 + mantissa;
        const int shift = exponent - 5;
        if (shift >= 0) {
            magnitude = significand > ((uint64_t) INT64_MAX >> shift) ? INT64_MAX : significand << shift;
        } else {
            const int right_shift = -shift;
            magnitude = (significand + ((uint64_t) 1 << (right_shift - 1))) >> right_shift;
        }
    }
    if (!sign) return (int64_t) magnitude;
    return magnitude >= (uint64_t) INT64_MAX ? INT64_MIN : -(int64_t) magnitude;
}

static inline int32_t ggml_gptq2_source_zero_point_q20(
        int64_t source_scale_q20,
        int64_t zero_bias_q20) {
    GGML_ASSERT(source_scale_q20 > 0);
    if (zero_bias_q20 <= 0) {
        return 0;
    }
    const int64_t rounded_numerator = zero_bias_q20 + source_scale_q20 / 2;
    if (rounded_numerator < source_scale_q20) {
        return 0;
    }
    if (rounded_numerator < 2 * source_scale_q20) {
        return 1;
    }
    if (rounded_numerator < 3 * source_scale_q20) {
        return 2;
    }
    return 3;
}

static void ggml_vec_dot_gptq2_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(activation_to_output_q20 > 0);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int code_bytes = group_size / 4;
    const int block_bytes = code_bytes + 4;
    int64_t scaled_accumulator = 0;
    for (int group_index = 0; group_index < n / group_size; ++group_index) {
        const uint8_t * block = weights + group_index * block_bytes;
        ggml_fp16_t raw_scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        memcpy(&raw_scale_fp16, block + code_bytes, sizeof(raw_scale_fp16));
        memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        int64_t weight_scale_q20 = ggml_gptq2_32_fp16_to_q20(raw_scale_fp16);
        weight_scale_q20 = MAX(weight_scale_q20, GGML_U16_MIN_WEIGHT_SCALE_Q20);
        const int64_t zero_bias_q20 = ggml_gptq2_32_fp16_to_q20(zero_bias_fp16);
        const int32_t weight_zero_point =
            ggml_gptq2_source_zero_point_q20(weight_scale_q20, zero_bias_q20);
        const int64_t multiplier = ggml_gptq2_32_round_shift_away_from_zero(
            activation_to_output_q20 * weight_scale_q20, GGML_U16_Q20_SHIFT);
        int64_t group_dot = 0;
        for (int chunk = 0; chunk < group_size / 32; ++chunk) {
            const uint16_t * chunk_activations =
                activations + group_index * group_size + chunk * 32;
            const uint8_t * chunk_codes = block + chunk * 8;
#if defined(__ARM_NEON)
            group_dot += ggml_gptq2_32_dot_u16_neon(
                chunk_activations, activation_zero_point, chunk_codes, weight_zero_point);
#else
            group_dot += ggml_gptq2_32_dot_u16_scalar(
                chunk_activations, activation_zero_point, chunk_codes, weight_zero_point);
#endif
        }
        scaled_accumulator += group_dot * multiplier;
    }
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        scaled_accumulator, GGML_U16_Q20_SHIFT) + output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_dot_gptq2_32_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_fixed(
        n, output, packed_weights, activations, activation_to_output_q20,
        activation_zero_point, output_zero_point, 32);
}

void ggml_vec_dot_gptq2_64_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_fixed(
        n, output, packed_weights, activations, activation_to_output_q20,
        activation_zero_point, output_zero_point, 64);
}

void ggml_vec_dot_gptq2_128_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_fixed(
        n, output, packed_weights, activations, activation_to_output_q20,
        activation_zero_point, output_zero_point, 128);
}

static void ggml_vec_dot_gptq2_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(block_scale_codes != NULL && channel_scale_to_output_q31 > 0);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    const int qnn_blocks = n / 32;
    int64_t scaled_accumulator = 0;
    int fractional_bits = 31;
    int intermediate_shift = 0;
    if (fractional_constant > 0) {
        const int multiplier_exponent =
            63 - __builtin_clzll((unsigned long long) channel_scale_to_output_q31);
        fractional_bits = MAX(1, MIN(20, fractional_constant - multiplier_exponent));
        intermediate_shift = 31 - fractional_bits;
    }

    for (int qnn_block = 0; qnn_block < qnn_blocks; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * block = weights + source_group * source_block_bytes;
        ggml_fp16_t raw_scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        memcpy(&raw_scale_fp16, block + code_bytes, sizeof(raw_scale_fp16));
        memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        int64_t source_scale_q20 = ggml_gptq2_32_fp16_to_q20(raw_scale_fp16);
        source_scale_q20 = MAX(source_scale_q20, GGML_U16_MIN_WEIGHT_SCALE_Q20);
        const int64_t zero_bias_q20 = ggml_gptq2_32_fp16_to_q20(zero_bias_fp16);
        const int32_t weight_zero_point =
            ggml_gptq2_source_zero_point_q20(source_scale_q20, zero_bias_q20);
        const uint8_t * chunk_codes = block + chunk * 8;
        const uint16_t * chunk_activations = activations + qnn_block * 32;
#if defined(__ARM_NEON)
        const int64_t dot = ggml_gptq2_32_dot_u16_neon(
            chunk_activations, activation_zero_point, chunk_codes,
            weight_zero_point);
#else
        const int64_t dot = ggml_gptq2_32_dot_u16_scalar(
            chunk_activations, activation_zero_point, chunk_codes,
            (int32_t) weight_zero_point);
#endif
        const int64_t multiplier =
            channel_scale_to_output_q31 * block_scale_codes[qnn_block];
        GGML_ASSERT(multiplier >= 0 &&
            (dot == 0 || llabs(dot) <= INT64_MAX / MAX(INT64_C(1), multiplier)));
        const int64_t block_accumulator = dot * multiplier;
        scaled_accumulator += fractional_constant > 0
            ? ggml_u16_floor_shift(block_accumulator, intermediate_shift)
            : block_accumulator;
    }
    GGML_ASSERT(output_bias_q7 == 0 || !final_round_to_nearest);
    const int64_t shifted = final_round_to_nearest
        ? ggml_u16_htp_round_shift(scaled_accumulator, fractional_bits)
        : ggml_u16_floor_shift_with_q7_bias(
            scaled_accumulator, fractional_bits, output_bias_q7);
    const int64_t quantized = shifted + output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_fixed(
        n, output, packed_weights, activations, block_scale_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 32, 0, 0, 0);
}

void ggml_vec_dot_gptq2_64_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_fixed(
        n, output, packed_weights, activations, block_scale_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 64, 0, 0, 0);
}

void ggml_vec_dot_gptq2_128_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_fixed(
        n, output, packed_weights, activations, block_scale_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 128, 0, 0, 0);
}

void ggml_gptq2_prepare_qnn_block_codes(
        int n,
        uint8_t * prepared_block_codes,
        const void * GGML_RESTRICT packed_weights,
        const uint8_t * block_scale_codes,
        int group_size) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(prepared_block_codes != NULL);
    GGML_ASSERT(packed_weights != NULL);
    GGML_ASSERT(block_scale_codes != NULL);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int source_code_bytes = group_size / 4;
    const int source_block_bytes = source_code_bytes + 4;
    const int qnn_blocks_per_source_group = group_size / 32;
    for (int source_group = 0; source_group < n / group_size; ++source_group) {
        const uint8_t * block = weights + source_group * source_block_bytes;
        ggml_fp16_t raw_scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        memcpy(&raw_scale_fp16, block + source_code_bytes, sizeof(raw_scale_fp16));
        memcpy(
            &zero_bias_fp16,
            block + source_code_bytes + sizeof(raw_scale_fp16),
            sizeof(zero_bias_fp16));
        int64_t source_scale_q20 = ggml_gptq2_32_fp16_to_q20(raw_scale_fp16);
        source_scale_q20 = MAX(source_scale_q20, GGML_U16_MIN_WEIGHT_SCALE_Q20);
        const int64_t zero_bias_q20 =
            ggml_gptq2_32_fp16_to_q20(zero_bias_fp16);
        const int32_t weight_zero_point =
            ggml_gptq2_source_zero_point_q20(source_scale_q20, zero_bias_q20);
        for (int chunk = 0; chunk < qnn_blocks_per_source_group; ++chunk) {
            const int block_index =
                source_group * qnn_blocks_per_source_group + chunk;
            prepared_block_codes[block_index] =
                (block_scale_codes[block_index] & 0x1f) |
                (uint8_t) (weight_zero_point << 5);
        }
    }
}

void ggml_gptq2_32_gs32_restore_rows(
        int n,
        const void * GGML_RESTRICT gs32_weights,
        int64_t first_row,
        int row_count,
        void * GGML_RESTRICT row_major_weights,
        size_t row_major_stride) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(gs32_weights != NULL && row_major_weights != NULL);
    GGML_ASSERT(first_row >= 0 && row_count > 0 && row_count <= 8);

    const int groups = n / 32;
    const size_t row_bytes = (size_t) groups * 12;
    const size_t row_block_bytes = 64 * row_bytes;
    const uint8_t * source = (const uint8_t *) gs32_weights;
    uint8_t * destination = (uint8_t *) row_major_weights;
    GGML_ASSERT(row_major_stride >= row_bytes);

    // Decode's 8-row micro-kernel always requests an aligned row tile. The
    // source layout stores each two-byte qcode pair for those eight rows as
    // one contiguous 16-byte run, so consume that native tiling directly.
    if (row_count == 8 && first_row % 8 == 0 &&
        first_row / 64 == (first_row + 7) / 64) {
        const size_t row_block = (size_t) first_row / 64;
        const size_t first = (size_t) first_row % 64;
        const size_t row_outer = first / 32;
        const size_t row_middle = (first % 32) / 8;
        for (int group = 0; group < groups; ++group) {
            const uint8_t * source_group =
                source + row_block * row_block_bytes + (size_t) group * 768;
            for (size_t pair = 0; pair < 4; ++pair) {
                const size_t source_offset =
                    (((row_outer * 4 + pair) * 4 + row_middle) * 8) * 2;
                for (size_t row = 0; row < 8; ++row) {
                    memcpy(
                        destination + row * row_major_stride +
                            (size_t) group * 12 + pair * 2,
                        source_group + source_offset + row * 2,
                        2);
                }
            }
            for (size_t row = 0; row < 8; ++row) {
                memcpy(
                    destination + row * row_major_stride +
                        (size_t) group * 12 + 8,
                    source_group + 512 + (first + row) * 4,
                    4);
            }
        }
        return;
    }

    for (int output_row = 0; output_row < row_count; ++output_row) {
        const int64_t absolute_row = first_row + output_row;
        const size_t row_block = (size_t) absolute_row / 64;
        const size_t row = (size_t) absolute_row % 64;
        const size_t row_outer = row / 32;
        const size_t row_middle = (row % 32) / 8;
        const size_t row_inner = row % 8;
        uint8_t * row_destination =
            destination + (size_t) output_row * row_major_stride;

        for (int group = 0; group < groups; ++group) {
            const uint8_t * source_group =
                source + row_block * row_block_bytes + (size_t) group * 768;
            uint8_t * block_destination =
                row_destination + (size_t) group * 12;
            for (size_t qbyte = 0; qbyte < 8; ++qbyte) {
                const size_t byte_outer = qbyte / 2;
                const size_t byte_inner = qbyte % 2;
                const size_t source_offset =
                    (((((row_outer * 4 + byte_outer) * 4 + row_middle) * 8 +
                        row_inner) * 2) + byte_inner);
                block_destination[qbyte] = source_group[source_offset];
            }
            memcpy(block_destination + 8, source_group + 512 + row * 4, 4);
        }
    }
}

int64_t ggml_gptq2_32_qnn_prepared_weight_sum(
        int n,
        const void * GGML_RESTRICT packed_weights,
        const uint8_t * GGML_RESTRICT prepared_block_codes) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(packed_weights != NULL && prepared_block_codes != NULL);
    const uint8_t * const weights = packed_weights;
    int64_t result = 0;
    for (int block = 0; block < n / 32; ++block) {
        const uint8_t prepared = prepared_block_codes[block];
        result +=
            (int64_t) ggml_gptq2_32_weight_sum_scalar(
                weights + block * 12, (prepared >> 5) & 0x3) *
            (prepared & 0x1f);
    }
    return result;
}

static void ggml_vec_dot_gptq2_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(prepared_block_codes != NULL && channel_scale_to_output_q31 > 0);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int source_code_bytes = group_size / 4;
    const int source_block_bytes = source_code_bytes + 4;
    const int qnn_blocks_per_source_group = group_size / 32;
    int64_t scaled_accumulator = 0;
    int fractional_bits = 31;
    int intermediate_shift = 0;
    if (fractional_constant > 0) {
        const int multiplier_exponent =
            63 - __builtin_clzll((unsigned long long) channel_scale_to_output_q31);
        fractional_bits = MAX(1, MIN(20, fractional_constant - multiplier_exponent));
        intermediate_shift = 31 - fractional_bits;
    }

    for (int block_index = 0; block_index < n / 32; ++block_index) {
        const uint8_t prepared = prepared_block_codes[block_index];
        const int32_t weight_zero_point = (prepared >> 5) & 0x3;
        const int source_group = block_index / qnn_blocks_per_source_group;
        const int chunk = block_index % qnn_blocks_per_source_group;
#if defined(__ARM_NEON)
        const int64_t dot = ggml_gptq2_32_dot_u16_neon(
            activations + block_index * 32,
            activation_zero_point,
            weights + source_group * source_block_bytes + chunk * 8,
            weight_zero_point);
#else
        const int64_t dot = ggml_gptq2_32_dot_u16_scalar(
            activations + block_index * 32,
            activation_zero_point,
            weights + source_group * source_block_bytes + chunk * 8,
            weight_zero_point);
#endif
        const int64_t multiplier =
            channel_scale_to_output_q31 * (prepared & 0x1f);
        GGML_ASSERT(multiplier >= 0 &&
            (dot == 0 || llabs(dot) <= INT64_MAX / MAX(INT64_C(1), multiplier)));
        const int64_t block_accumulator = dot * multiplier;
        scaled_accumulator += fractional_constant > 0
            ? ggml_u16_floor_shift(block_accumulator, intermediate_shift)
            : block_accumulator;
    }

    GGML_ASSERT(output_bias_q7 == 0 || !final_round_to_nearest);
    const int64_t shifted = final_round_to_nearest
        ? ggml_u16_htp_round_shift(scaled_accumulator, fractional_bits)
        : ggml_u16_floor_shift_with_q7_bias(
            scaled_accumulator, fractional_bits, output_bias_q7);
    const int64_t quantized = shifted + output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static void ggml_vec_dot_gptq2_u16_qnn_blockwise_affine_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(prepared_block_codes != NULL && channel_scale_to_output_q31 > 0);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int source_code_bytes = group_size / 4;
    const int source_block_bytes = source_code_bytes + 4;
    const int qnn_blocks_per_source_group = group_size / 32;
    int64_t centered_dot = 0;
    int64_t expanded_weight_sum = 0;

    for (int block_index = 0; block_index < n / 32; ++block_index) {
        const uint8_t prepared = prepared_block_codes[block_index];
        const int32_t weight_zero_point = (prepared >> 5) & 0x3;
        const int32_t block_scale_code = prepared & 0x1f;
        const int source_group = block_index / qnn_blocks_per_source_group;
        const int chunk = block_index % qnn_blocks_per_source_group;
        const uint8_t * packed_codes =
            weights + source_group * source_block_bytes + chunk * 8;
#if defined(__ARM_NEON)
        const int64_t block_dot = ggml_gptq2_32_dot_u16_neon(
            activations + block_index * 32,
            activation_zero_point,
            packed_codes,
            weight_zero_point);
#else
        const int64_t block_dot = ggml_gptq2_32_dot_u16_scalar(
            activations + block_index * 32,
            activation_zero_point,
            packed_codes,
            weight_zero_point);
#endif
        centered_dot += block_dot * block_scale_code;
        expanded_weight_sum +=
            (int64_t) ggml_gptq2_32_weight_sum_scalar(
                packed_codes, weight_zero_point) *
            block_scale_code;
    }

    const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
        centered_dot, expanded_weight_sum, activation_zero_point);
    const int64_t quantized =
        ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31, 31) +
        output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static void ggml_vec_dot_gptq2_u16_qnn_blockwise_affine_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size) {
    GGML_ASSERT(group_size >= 32 && group_size % 32 == 0);
    GGML_ASSERT(n > 0 && n % group_size == 0);
    GGML_ASSERT(block_scale_codes != NULL && channel_scale_to_output_q31 > 0);
    const uint8_t * GGML_RESTRICT weights = packed_weights;
    const int source_code_bytes = group_size / 4;
    const int source_block_bytes = source_code_bytes + 4;
    const int qnn_blocks_per_source_group = group_size / 32;
    int64_t centered_dot = 0;
    int64_t expanded_weight_sum = 0;

    for (int block_index = 0; block_index < n / 32; ++block_index) {
        const int source_group = block_index / qnn_blocks_per_source_group;
        const int chunk = block_index % qnn_blocks_per_source_group;
        const uint8_t * block = weights + source_group * source_block_bytes;
        ggml_fp16_t raw_scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        memcpy(&raw_scale_fp16, block + source_code_bytes, sizeof(raw_scale_fp16));
        memcpy(
            &zero_bias_fp16,
            block + source_code_bytes + sizeof(raw_scale_fp16),
            sizeof(zero_bias_fp16));
        int64_t source_scale_q20 = ggml_gptq2_32_fp16_to_q20(raw_scale_fp16);
        source_scale_q20 = MAX(source_scale_q20, GGML_U16_MIN_WEIGHT_SCALE_Q20);
        const int64_t zero_bias_q20 =
            ggml_gptq2_32_fp16_to_q20(zero_bias_fp16);
        const int32_t weight_zero_point =
            ggml_gptq2_source_zero_point_q20(source_scale_q20, zero_bias_q20);
        const int32_t block_scale_code = block_scale_codes[block_index];
        const uint8_t * packed_codes = block + chunk * 8;
#if defined(__ARM_NEON)
        const int64_t block_dot = ggml_gptq2_32_dot_u16_neon(
            activations + block_index * 32,
            activation_zero_point,
            packed_codes,
            weight_zero_point);
#else
        const int64_t block_dot = ggml_gptq2_32_dot_u16_scalar(
            activations + block_index * 32,
            activation_zero_point,
            packed_codes,
            weight_zero_point);
#endif
        centered_dot += block_dot * block_scale_code;
        expanded_weight_sum +=
            (int64_t) ggml_gptq2_32_weight_sum_scalar(
                packed_codes, weight_zero_point) *
            block_scale_code;
    }

    const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
        centered_dot, expanded_weight_sum, activation_zero_point);
    const int64_t quantized =
        ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31, 31) +
        output_zero_point;
    *output = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_prepared(
        n, output, packed_weights, activations, prepared_block_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 32, 0, 0, 0);
}

void ggml_vec_dot_gptq2_64_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_prepared(
        n, output, packed_weights, activations, prepared_block_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 64, 0, 0, 0);
}

void ggml_vec_dot_gptq2_128_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point) {
    ggml_vec_dot_gptq2_u16_qnn_blockwise_prepared(
        n, output, packed_weights, activations, prepared_block_codes,
        channel_scale_to_output_q31, activation_zero_point, output_zero_point, 128, 0, 0, 0);
}

void ggml_vec_dot_gptq2_u16_qnn_blockwise_requant(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int metadata_prepared,
        int fractional_constant) {
    GGML_ASSERT(fractional_constant > 0);
    if (metadata_prepared) {
        ggml_vec_dot_gptq2_u16_qnn_blockwise_prepared(
            n, output, packed_weights, activations, block_scale_codes,
            channel_scale_to_output_q31, activation_zero_point, output_zero_point,
            group_size, fractional_constant, 0, 0);
    } else {
        ggml_vec_dot_gptq2_u16_qnn_blockwise_fixed(
            n, output, packed_weights, activations, block_scale_codes,
            channel_scale_to_output_q31, activation_zero_point, output_zero_point,
            group_size, fractional_constant, 0, 0);
    }
}

void ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int metadata_prepared,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(fractional_constant >= 0);
    (void) fractional_constant;
    (void) final_round_to_nearest;
    (void) output_bias_q7;
    if (metadata_prepared) {
        ggml_vec_dot_gptq2_u16_qnn_blockwise_affine_prepared(
            n, output, packed_weights, activations, block_scale_codes,
            channel_scale_to_output_q31, activation_zero_point, output_zero_point,
            group_size);
    } else {
        ggml_vec_dot_gptq2_u16_qnn_blockwise_affine_fixed(
            n, output, packed_weights, activations, block_scale_codes,
            channel_scale_to_output_q31, activation_zero_point, output_zero_point,
            group_size);
    }
}

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_4rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(outputs != NULL && packed_weights != NULL);
    GGML_ASSERT(activations != NULL && activation_block_sums != NULL);
    GGML_ASSERT(prepared_block_codes != NULL);
    GGML_ASSERT(channel_scale_to_output_q31 != NULL);
    GGML_ASSERT(prepared_weight_sums != NULL);
#if !defined(__ARM_NEON)
    (void) activations_fit_i16;
#endif

    const uint8_t * const weights = packed_weights;
    const int blocks = n / 32;
    (void) fractional_constant;
    (void) final_round_to_nearest;
    (void) output_bias_q7;
    int64_t centered_dots[4] = { 0, 0, 0, 0 };

    for (int block = 0; block < blocks; ++block) {
        const uint8_t prepared0 = prepared_block_codes[block];
        const uint8_t prepared1 = prepared_block_codes[prepared_row_stride + block];
        const uint8_t prepared2 = prepared_block_codes[2 * prepared_row_stride + block];
        const uint8_t prepared3 = prepared_block_codes[3 * prepared_row_stride + block];
        const uint8_t * codes0 = weights + block * 12;
        const uint8_t * codes1 = weights + weight_row_stride + block * 12;
        const uint8_t * codes2 = weights + 2 * weight_row_stride + block * 12;
        const uint8_t * codes3 = weights + 3 * weight_row_stride + block * 12;
        int64_t dot0;
        int64_t dot1;
        int64_t dot2;
        int64_t dot3;
#if defined(__ARM_NEON)
        ggml_gptq2_32_dot_u16_neon_4rows(
            activations + block * 32, activation_block_sums[block],
            activations_fit_i16, activation_zero_point,
            codes0, codes1, codes2, codes3,
            (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
            (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
            &dot0, &dot1, &dot2, &dot3);
#else
        dot0 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            codes0, (prepared0 >> 5) & 0x3);
        dot1 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            codes1, (prepared1 >> 5) & 0x3);
        dot2 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            codes2, (prepared2 >> 5) & 0x3);
        dot3 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            codes3, (prepared3 >> 5) & 0x3);
#endif
        const int32_t block_scale0 = prepared0 & 0x1f;
        const int32_t block_scale1 = prepared1 & 0x1f;
        const int32_t block_scale2 = prepared2 & 0x1f;
        const int32_t block_scale3 = prepared3 & 0x1f;
        centered_dots[0] += dot0 * block_scale0;
        centered_dots[1] += dot1 * block_scale1;
        centered_dots[2] += dot2 * block_scale2;
        centered_dots[3] += dot3 * block_scale3;
    }

    for (int row = 0; row < 4; ++row) {
        const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
            centered_dots[row],
            prepared_weight_sums[row],
            activation_zero_point);
        const int64_t shifted = ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31[row], 31);
        outputs[row] = (uint16_t) MAX(
            0, MIN(UINT16_MAX, shifted + output_zero_point));
    }
}

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(outputs != NULL && packed_weights != NULL);
    GGML_ASSERT(activations != NULL && activation_block_sums != NULL);
    GGML_ASSERT(prepared_block_codes != NULL);
    GGML_ASSERT(channel_scale_to_output_q31 != NULL);
    GGML_ASSERT(prepared_weight_sums != NULL);
#if !defined(__ARM_NEON)
    (void) activations_fit_i16;
#endif

    const uint8_t * const weights = packed_weights;
    const int blocks = n / 32;
    (void) fractional_constant;
    (void) final_round_to_nearest;
    (void) output_bias_q7;
    int64_t centered_dots[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

#if defined(__aarch64__) && defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
    for (int block = 0; block < blocks; ++block) {
        const uint8_t prepared0 = prepared_block_codes[block];
        const uint8_t prepared1 = prepared_block_codes[prepared_row_stride + block];
        const uint8_t prepared2 = prepared_block_codes[2 * prepared_row_stride + block];
        const uint8_t prepared3 = prepared_block_codes[3 * prepared_row_stride + block];
        const uint8_t prepared4 = prepared_block_codes[4 * prepared_row_stride + block];
        const uint8_t prepared5 = prepared_block_codes[5 * prepared_row_stride + block];
        const uint8_t prepared6 = prepared_block_codes[6 * prepared_row_stride + block];
        const uint8_t prepared7 = prepared_block_codes[7 * prepared_row_stride + block];
        const uint8_t * const block_weights = weights + block * 12;
        int64_t dot0;
        int64_t dot1;
        int64_t dot2;
        int64_t dot3;
        int64_t dot4;
        int64_t dot5;
        int64_t dot6;
        int64_t dot7;

#if defined(__ARM_NEON) && defined(__aarch64__)
        if (activations_fit_i16) {
            const uint16x8x4_t activation_lanes =
                vld4q_u16(activations + block * 32);
            const uint16x8_t activation_zero_u16 =
                vdupq_n_u16((uint16_t) activation_zero_point);
            const int16x8_t activation0 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[0], activation_zero_u16));
            const int16x8_t activation1 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[1], activation_zero_u16));
            const int16x8_t activation2 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[2], activation_zero_u16));
            const int16x8_t activation3 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[3], activation_zero_u16));
            ggml_gptq2_32_dot_s16_neon_4rows(
                activation0, activation1, activation2, activation3,
                activation_block_sums[block],
                block_weights,
                block_weights + weight_row_stride,
                block_weights + 2 * weight_row_stride,
                block_weights + 3 * weight_row_stride,
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
                &dot0, &dot1, &dot2, &dot3);
            centered_dots[0] += dot0 * (prepared0 & 0x1f);
            centered_dots[1] += dot1 * (prepared1 & 0x1f);
            centered_dots[2] += dot2 * (prepared2 & 0x1f);
            centered_dots[3] += dot3 * (prepared3 & 0x1f);
            ggml_gptq2_32_dot_s16_neon_4rows(
                activation0, activation1, activation2, activation3,
                activation_block_sums[block],
                block_weights + 4 * weight_row_stride,
                block_weights + 5 * weight_row_stride,
                block_weights + 6 * weight_row_stride,
                block_weights + 7 * weight_row_stride,
                (prepared4 >> 5) & 0x3, (prepared5 >> 5) & 0x3,
                (prepared6 >> 5) & 0x3, (prepared7 >> 5) & 0x3,
                &dot4, &dot5, &dot6, &dot7);
            centered_dots[4] += dot4 * (prepared4 & 0x1f);
            centered_dots[5] += dot5 * (prepared5 & 0x1f);
            centered_dots[6] += dot6 * (prepared6 & 0x1f);
            centered_dots[7] += dot7 * (prepared7 & 0x1f);
        } else
#endif
#if defined(__ARM_NEON)
        {
            ggml_gptq2_32_dot_u16_neon_4rows(
                activations + block * 32, activation_block_sums[block],
                activations_fit_i16, activation_zero_point,
                block_weights,
                block_weights + weight_row_stride,
                block_weights + 2 * weight_row_stride,
                block_weights + 3 * weight_row_stride,
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
                &dot0, &dot1, &dot2, &dot3);
            centered_dots[0] += dot0 * (prepared0 & 0x1f);
            centered_dots[1] += dot1 * (prepared1 & 0x1f);
            centered_dots[2] += dot2 * (prepared2 & 0x1f);
            centered_dots[3] += dot3 * (prepared3 & 0x1f);
            ggml_gptq2_32_dot_u16_neon_4rows(
                activations + block * 32, activation_block_sums[block],
                activations_fit_i16, activation_zero_point,
                block_weights + 4 * weight_row_stride,
                block_weights + 5 * weight_row_stride,
                block_weights + 6 * weight_row_stride,
                block_weights + 7 * weight_row_stride,
                (prepared4 >> 5) & 0x3, (prepared5 >> 5) & 0x3,
                (prepared6 >> 5) & 0x3, (prepared7 >> 5) & 0x3,
                &dot4, &dot5, &dot6, &dot7);
            centered_dots[4] += dot4 * (prepared4 & 0x1f);
            centered_dots[5] += dot5 * (prepared5 & 0x1f);
            centered_dots[6] += dot6 * (prepared6 & 0x1f);
            centered_dots[7] += dot7 * (prepared7 & 0x1f);
        }
#else
        dot0 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights, (prepared0 >> 5) & 0x3);
        dot1 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + weight_row_stride, (prepared1 >> 5) & 0x3);
        dot2 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 2 * weight_row_stride, (prepared2 >> 5) & 0x3);
        dot3 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 3 * weight_row_stride, (prepared3 >> 5) & 0x3);
        centered_dots[0] += dot0 * (prepared0 & 0x1f);
        centered_dots[1] += dot1 * (prepared1 & 0x1f);
        centered_dots[2] += dot2 * (prepared2 & 0x1f);
        centered_dots[3] += dot3 * (prepared3 & 0x1f);
        dot4 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 4 * weight_row_stride, (prepared4 >> 5) & 0x3);
        dot5 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 5 * weight_row_stride, (prepared5 >> 5) & 0x3);
        dot6 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 6 * weight_row_stride, (prepared6 >> 5) & 0x3);
        dot7 = ggml_gptq2_32_dot_u16_scalar(
            activations + block * 32, activation_zero_point,
            block_weights + 7 * weight_row_stride, (prepared7 >> 5) & 0x3);
        centered_dots[4] += dot4 * (prepared4 & 0x1f);
        centered_dots[5] += dot5 * (prepared5 & 0x1f);
        centered_dots[6] += dot6 * (prepared6 & 0x1f);
        centered_dots[7] += dot7 * (prepared7 & 0x1f);
#endif
    }

    for (int row = 0; row < 8; ++row) {
        const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
            centered_dots[row],
            prepared_weight_sums[row],
            activation_zero_point);
        const int64_t shifted = ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31[row], 31);
        outputs[row] = (uint16_t) MAX(
            0, MIN(UINT16_MAX, shifted + output_zero_point));
    }
}

void ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_8rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT gs32_weights,
        int64_t first_row,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT activation_low,
        const int8_t * GGML_RESTRICT activation_high,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(outputs != NULL && gs32_weights != NULL);
    GGML_ASSERT(activations != NULL && activation_block_sums != NULL);
    GGML_ASSERT(prepared_block_codes != NULL);
    GGML_ASSERT(channel_scale_to_output_q31 != NULL);
    GGML_ASSERT(prepared_weight_sums != NULL);
    GGML_ASSERT(first_row >= 0 && first_row % 8 == 0);
    GGML_ASSERT(first_row / 64 == (first_row + 7) / 64);
#if !defined(__ARM_NEON)
    (void) activations_fit_i16;
    (void) activation_low;
    (void) activation_high;
#endif
    (void) fractional_constant;
    (void) final_round_to_nearest;
    (void) output_bias_q7;

    const int blocks = n / 32;
    const size_t row_block_bytes = (size_t) blocks * 768;
    const size_t row_block = (size_t) first_row / 64;
    const size_t first = (size_t) first_row % 64;
    const size_t row_outer = first / 32;
    const size_t row_middle = (first % 32) / 8;
    const uint8_t * tensor_source =
        (const uint8_t *) gs32_weights + row_block * row_block_bytes;
    int64_t centered_dots[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int first_generic_block = 0;

#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__clang__)
    if (activations_fit_i16 && ggml_gptq2_32_gs32_dotprod_enabled()) {
        GGML_ASSERT(activation_low != NULL && activation_high != NULL);
        ggml_gptq2_32_gs32_dotprod_accumulate_blocks(
            blocks, tensor_source, row_outer, row_middle,
            activation_low, activation_high, activation_block_sums,
            prepared_block_codes, prepared_row_stride,
            activation_zero_point,
            centered_dots);
        first_generic_block = blocks;
    }
#endif

#if defined(__aarch64__) && defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
    for (int block = first_generic_block; block < blocks; ++block) {
        const uint8_t * source_group =
            tensor_source + (size_t) block * 768;
#if defined(__ARM_NEON)
        const size_t pair0_offset =
            (((row_outer * 4 + 0) * 4 + row_middle) * 8) * 2;
        const size_t pair1_offset =
            (((row_outer * 4 + 1) * 4 + row_middle) * 8) * 2;
        const size_t pair2_offset =
            (((row_outer * 4 + 2) * 4 + row_middle) * 8) * 2;
        const size_t pair3_offset =
            (((row_outer * 4 + 3) * 4 + row_middle) * 8) * 2;
        const uint8x16_t packed_pair0 =
            vld1q_u8(source_group + pair0_offset);
        const uint8x16_t packed_pair1 =
            vld1q_u8(source_group + pair1_offset);
        const uint8x16_t packed_pair2 =
            vld1q_u8(source_group + pair2_offset);
        const uint8x16_t packed_pair3 =
            vld1q_u8(source_group + pair3_offset);
#else
        uint8_t row_codes[8][8];
        for (size_t pair = 0; pair < 4; ++pair) {
            const size_t source_offset =
                (((row_outer * 4 + pair) * 4 + row_middle) * 8) * 2;
            const uint8_t * pair_source = source_group + source_offset;
            for (size_t row = 0; row < 8; ++row) {
                memcpy(row_codes[row] + pair * 2, pair_source + row * 2, 2);
            }
        }
#endif

        const uint8_t * prepared_lanes = prepared_row_stride == 0
            ? prepared_block_codes + (size_t) block * 8 : NULL;
        const uint8_t prepared0 = prepared_lanes != NULL
            ? prepared_lanes[0] : prepared_block_codes[block];
        const uint8_t prepared1 = prepared_lanes != NULL
            ? prepared_lanes[1] : prepared_block_codes[prepared_row_stride + block];
        const uint8_t prepared2 = prepared_lanes != NULL
            ? prepared_lanes[2] : prepared_block_codes[2 * prepared_row_stride + block];
        const uint8_t prepared3 = prepared_lanes != NULL
            ? prepared_lanes[3] : prepared_block_codes[3 * prepared_row_stride + block];
        const uint8_t prepared4 = prepared_lanes != NULL
            ? prepared_lanes[4] : prepared_block_codes[4 * prepared_row_stride + block];
        const uint8_t prepared5 = prepared_lanes != NULL
            ? prepared_lanes[5] : prepared_block_codes[5 * prepared_row_stride + block];
        const uint8_t prepared6 = prepared_lanes != NULL
            ? prepared_lanes[6] : prepared_block_codes[6 * prepared_row_stride + block];
        const uint8_t prepared7 = prepared_lanes != NULL
            ? prepared_lanes[7] : prepared_block_codes[7 * prepared_row_stride + block];
        int64_t dots[8];

#if defined(__ARM_NEON) && defined(__aarch64__)
        if (activations_fit_i16) {
            const uint16x8x4_t activation_lanes =
                vld4q_u16(activations + block * 32);
            const uint16x8_t activation_zero_u16 =
                vdupq_n_u16((uint16_t) activation_zero_point);
            const int16x8_t activation0 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[0], activation_zero_u16));
            const int16x8_t activation1 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[1], activation_zero_u16));
            const int16x8_t activation2 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[2], activation_zero_u16));
            const int16x8_t activation3 = vreinterpretq_s16_u16(
                vsubq_u16(activation_lanes.val[3], activation_zero_u16));
            const int32_t weight_zero_points[8] = {
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
                (prepared4 >> 5) & 0x3, (prepared5 >> 5) & 0x3,
                (prepared6 >> 5) & 0x3, (prepared7 >> 5) & 0x3,
            };
            ggml_gptq2_32_dot_s16_neon_gs32_8rows(
                activation0, activation1, activation2, activation3,
                activation_block_sums[block],
                packed_pair0, packed_pair1, packed_pair2, packed_pair3,
                weight_zero_points, dots);
        } else
#endif
#if defined(__ARM_NEON)
        {
            const uint16x8x2_t zip01 = vzipq_u16(
                vreinterpretq_u16_u8(packed_pair0),
                vreinterpretq_u16_u8(packed_pair1));
            const uint16x8x2_t zip23 = vzipq_u16(
                vreinterpretq_u16_u8(packed_pair2),
                vreinterpretq_u16_u8(packed_pair3));
            const uint32x4x2_t rows03 = vzipq_u32(
                vreinterpretq_u32_u16(zip01.val[0]),
                vreinterpretq_u32_u16(zip23.val[0]));
            const uint32x4x2_t rows47 = vzipq_u32(
                vreinterpretq_u32_u16(zip01.val[1]),
                vreinterpretq_u32_u16(zip23.val[1]));
            const uint8x16_t row01 =
                vreinterpretq_u8_u32(rows03.val[0]);
            const uint8x16_t row23 =
                vreinterpretq_u8_u32(rows03.val[1]);
            const uint8x16_t row45 =
                vreinterpretq_u8_u32(rows47.val[0]);
            const uint8x16_t row67 =
                vreinterpretq_u8_u32(rows47.val[1]);
#if defined(__aarch64__)
            const uint16x8x4_t activation_lanes =
                vld4q_u16(activations + block * 32);
            const int32x4_t activation_zero =
                vdupq_n_s32(activation_zero_point);
            int32x4_t activation_lo[4];
            int32x4_t activation_hi[4];
            for (int lane = 0; lane < 4; ++lane) {
                activation_lo[lane] = vsubq_s32(
                    vreinterpretq_s32_u32(vmovl_u16(
                        vget_low_u16(activation_lanes.val[lane]))),
                    activation_zero);
                activation_hi[lane] = vsubq_s32(
                    vreinterpretq_s32_u32(vmovl_u16(
                        vget_high_u16(activation_lanes.val[lane]))),
                    activation_zero);
            }
            ggml_gptq2_32_dot_s32_neon_4rows_regs(
                activation_lo, activation_hi, activation_block_sums[block],
                vget_low_u8(row01), vget_high_u8(row01),
                vget_low_u8(row23), vget_high_u8(row23),
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
                &dots[0], &dots[1], &dots[2], &dots[3]);
            ggml_gptq2_32_dot_s32_neon_4rows_regs(
                activation_lo, activation_hi, activation_block_sums[block],
                vget_low_u8(row45), vget_high_u8(row45),
                vget_low_u8(row67), vget_high_u8(row67),
                (prepared4 >> 5) & 0x3, (prepared5 >> 5) & 0x3,
                (prepared6 >> 5) & 0x3, (prepared7 >> 5) & 0x3,
                &dots[4], &dots[5], &dots[6], &dots[7]);
#else
            uint8_t row_codes[8][8];
            vst1q_u8(row_codes[0], row01);
            vst1q_u8(row_codes[2], row23);
            vst1q_u8(row_codes[4], row45);
            vst1q_u8(row_codes[6], row67);
            ggml_gptq2_32_dot_u16_neon_4rows(
                activations + block * 32, activation_block_sums[block],
                activations_fit_i16, activation_zero_point,
                row_codes[0], row_codes[1], row_codes[2], row_codes[3],
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3,
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3,
                &dots[0], &dots[1], &dots[2], &dots[3]);
            ggml_gptq2_32_dot_u16_neon_4rows(
                activations + block * 32, activation_block_sums[block],
                activations_fit_i16, activation_zero_point,
                row_codes[4], row_codes[5], row_codes[6], row_codes[7],
                (prepared4 >> 5) & 0x3, (prepared5 >> 5) & 0x3,
                (prepared6 >> 5) & 0x3, (prepared7 >> 5) & 0x3,
                &dots[4], &dots[5], &dots[6], &dots[7]);
#endif
        }
#else
        const uint8_t prepared_scalar[8] = {
            prepared0, prepared1, prepared2, prepared3,
            prepared4, prepared5, prepared6, prepared7,
        };
        for (int row = 0; row < 8; ++row) {
            dots[row] = ggml_gptq2_32_dot_u16_scalar(
                activations + block * 32, activation_zero_point,
                row_codes[row], (prepared_scalar[row] >> 5) & 0x3);
        }
#endif
        const uint8_t prepared[8] = {
            prepared0, prepared1, prepared2, prepared3,
            prepared4, prepared5, prepared6, prepared7,
        };
        for (int row = 0; row < 8; ++row) {
            centered_dots[row] += dots[row] * (prepared[row] & 0x1f);
        }
    }

    for (int row = 0; row < 8; ++row) {
        const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
            centered_dots[row], prepared_weight_sums[row],
            activation_zero_point);
        const int64_t shifted = ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31[row], 31);
        outputs[row] = (uint16_t) MAX(
            0, MIN(UINT16_MAX, shifted + output_zero_point));
    }
}

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_16rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7) {
#if !defined(__ARM_NEON) || !defined(__aarch64__)
    ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
        n, outputs, packed_weights, weight_row_stride, activations,
        activation_block_sums, activations_fit_i16, prepared_block_codes,
        prepared_row_stride, channel_scale_to_output_q31, prepared_weight_sums,
        activation_zero_point, output_zero_point, fractional_constant,
        final_round_to_nearest, output_bias_q7);
    ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
        n, outputs + 8,
        (const uint8_t *) packed_weights + 8 * weight_row_stride,
        weight_row_stride, activations, activation_block_sums,
        activations_fit_i16,
        prepared_block_codes + 8 * prepared_row_stride,
        prepared_row_stride, channel_scale_to_output_q31 + 8,
        prepared_weight_sums + 8, activation_zero_point, output_zero_point,
        fractional_constant, final_round_to_nearest, output_bias_q7);
#else
    if (!activations_fit_i16) {
        ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
            n, outputs, packed_weights, weight_row_stride, activations,
            activation_block_sums, activations_fit_i16, prepared_block_codes,
            prepared_row_stride, channel_scale_to_output_q31, prepared_weight_sums,
            activation_zero_point, output_zero_point, fractional_constant,
            final_round_to_nearest, output_bias_q7);
        ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
            n, outputs + 8,
            (const uint8_t *) packed_weights + 8 * weight_row_stride,
            weight_row_stride, activations, activation_block_sums,
            activations_fit_i16,
            prepared_block_codes + 8 * prepared_row_stride,
            prepared_row_stride, channel_scale_to_output_q31 + 8,
            prepared_weight_sums + 8, activation_zero_point, output_zero_point,
            fractional_constant, final_round_to_nearest, output_bias_q7);
        return;
    }

    GGML_ASSERT(n > 0 && n % 32 == 0);
    GGML_ASSERT(outputs != NULL && packed_weights != NULL);
    GGML_ASSERT(activations != NULL && activation_block_sums != NULL);
    GGML_ASSERT(prepared_block_codes != NULL);
    GGML_ASSERT(channel_scale_to_output_q31 != NULL);
    GGML_ASSERT(prepared_weight_sums != NULL);
    const uint8_t * const weights = packed_weights;
    const int blocks = n / 32;
    int64_t centered_dots[16] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };

    for (int block = 0; block < blocks; ++block) {
        const uint16x8x4_t activation_lanes =
            vld4q_u16(activations + block * 32);
        const uint16x8_t activation_zero_u16 =
            vdupq_n_u16((uint16_t) activation_zero_point);
        const int16x8_t activation0 = vreinterpretq_s16_u16(
            vsubq_u16(activation_lanes.val[0], activation_zero_u16));
        const int16x8_t activation1 = vreinterpretq_s16_u16(
            vsubq_u16(activation_lanes.val[1], activation_zero_u16));
        const int16x8_t activation2 = vreinterpretq_s16_u16(
            vsubq_u16(activation_lanes.val[2], activation_zero_u16));
        const int16x8_t activation3 = vreinterpretq_s16_u16(
            vsubq_u16(activation_lanes.val[3], activation_zero_u16));
        const uint8_t * const block_weights = weights + block * 12;
#define GGML_GPTQ2_PROCESS_4ROW_GROUP(BASE) do { \
            const uint8_t prepared0 = prepared_block_codes[((BASE) + 0) * prepared_row_stride + block]; \
            const uint8_t prepared1 = prepared_block_codes[((BASE) + 1) * prepared_row_stride + block]; \
            const uint8_t prepared2 = prepared_block_codes[((BASE) + 2) * prepared_row_stride + block]; \
            const uint8_t prepared3 = prepared_block_codes[((BASE) + 3) * prepared_row_stride + block]; \
            int64_t dot0; \
            int64_t dot1; \
            int64_t dot2; \
            int64_t dot3; \
            ggml_gptq2_32_dot_s16_neon_4rows( \
                activation0, activation1, activation2, activation3, \
                activation_block_sums[block], \
                block_weights + ((BASE) + 0) * weight_row_stride, \
                block_weights + ((BASE) + 1) * weight_row_stride, \
                block_weights + ((BASE) + 2) * weight_row_stride, \
                block_weights + ((BASE) + 3) * weight_row_stride, \
                (prepared0 >> 5) & 0x3, (prepared1 >> 5) & 0x3, \
                (prepared2 >> 5) & 0x3, (prepared3 >> 5) & 0x3, \
                &dot0, &dot1, &dot2, &dot3); \
            centered_dots[(BASE) + 0] += dot0 * (prepared0 & 0x1f); \
            centered_dots[(BASE) + 1] += dot1 * (prepared1 & 0x1f); \
            centered_dots[(BASE) + 2] += dot2 * (prepared2 & 0x1f); \
            centered_dots[(BASE) + 3] += dot3 * (prepared3 & 0x1f); \
        } while (0)
        GGML_GPTQ2_PROCESS_4ROW_GROUP(0);
        GGML_GPTQ2_PROCESS_4ROW_GROUP(4);
        GGML_GPTQ2_PROCESS_4ROW_GROUP(8);
        GGML_GPTQ2_PROCESS_4ROW_GROUP(12);
#undef GGML_GPTQ2_PROCESS_4ROW_GROUP
    }

    for (int row = 0; row < 16; ++row) {
        const int64_t reduced_dot = ggml_qnn_a16s8_reduce_accumulator(
            centered_dots[row], prepared_weight_sums[row],
            activation_zero_point);
        const int64_t shifted = ggml_u16_htp_round_shift(
            reduced_dot * channel_scale_to_output_q31[row], 31);
        outputs[row] = (uint16_t) MAX(
            0, MIN(UINT16_MAX, shifted + output_zero_point));
    }
#endif
}

void ggml_vec_add_affine_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        float lhs_scale,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        float rhs_scale,
        int32_t rhs_zero_point,
        float output_scale,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(lhs_scale > 0.0f);
    GGML_ASSERT(rhs_scale > 0.0f);
    GGML_ASSERT(output_scale > 0.0f);

    const int requant_shift = 20;
    const int64_t lhs_multiplier = (int64_t) llround(ldexp(
        (double) lhs_scale / (double) output_scale, requant_shift));
    const int64_t rhs_multiplier = (int64_t) llround(ldexp(
        (double) rhs_scale / (double) output_scale, requant_shift));
    ggml_vec_add_affine_u16_qnn_fixed(
        n, output, lhs, lhs_multiplier, lhs_zero_point,
        rhs, rhs_multiplier, rhs_zero_point, output_zero_point);
}

void ggml_vec_add_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_multiplier,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int64_t rhs_multiplier,
        int32_t rhs_zero_point,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    const int requant_shift = GGML_U16_Q20_SHIFT;
    int index = 0;

#if defined(__ARM_NEON)
    if (lhs_multiplier >= INT32_MIN && lhs_multiplier <= INT32_MAX &&
        rhs_multiplier >= INT32_MIN && rhs_multiplier <= INT32_MAX) {
        const int32x4_t lhs_zero = vdupq_n_s32(lhs_zero_point);
        const int32x4_t rhs_zero = vdupq_n_s32(rhs_zero_point);
        const int32x2_t lhs_mul = vdup_n_s32((int32_t) lhs_multiplier);
        const int32x2_t rhs_mul = vdup_n_s32((int32_t) rhs_multiplier);

        for (; index + 4 <= n; index += 4) {
            const int32x4_t lhs_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(lhs + index))), lhs_zero);
            const int32x4_t rhs_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(rhs + index))), rhs_zero);
            const int64x2_t sum_lo = vaddq_s64(
                vmull_s32(vget_low_s32(lhs_values), lhs_mul),
                vmull_s32(vget_low_s32(rhs_values), rhs_mul));
            const int64x2_t sum_hi = vaddq_s64(
                vmull_s32(vget_high_s32(lhs_values), lhs_mul),
                vmull_s32(vget_high_s32(rhs_values), rhs_mul));
            const int64_t sum0 = vgetq_lane_s64(sum_lo, 0);
            const int64_t sum1 = vgetq_lane_s64(sum_lo, 1);
            const int64_t sum2 = vgetq_lane_s64(sum_hi, 0);
            const int64_t sum3 = vgetq_lane_s64(sum_hi, 1);
            output[index + 0] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(sum0, requant_shift) + output_zero_point));
            output[index + 1] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(sum1, requant_shift) + output_zero_point));
            output[index + 2] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(sum2, requant_shift) + output_zero_point));
            output[index + 3] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(sum3, requant_shift) + output_zero_point));
        }
    }
#endif

    for (; index < n; ++index) {
        const int64_t lhs_value = (int64_t) lhs[index] - lhs_zero_point;
        const int64_t rhs_value = (int64_t) rhs[index] - rhs_zero_point;
        const int64_t scaled_sum = lhs_value * lhs_multiplier + rhs_value * rhs_multiplier;
        const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
            scaled_sum, requant_shift) + output_zero_point;
        output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
    }
}

void ggml_vec_add_affine_u16_qnn_q15(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_multiplier,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int64_t rhs_multiplier,
        int32_t rhs_zero_point,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    for (int index = 0; index < n; ++index) {
        const int64_t lhs_value = (int64_t) lhs[index] - lhs_zero_point;
        const int64_t rhs_value = (int64_t) rhs[index] - rhs_zero_point;
        const int64_t scaled_sum =
            lhs_value * lhs_multiplier + rhs_value * rhs_multiplier;
        const int64_t quantized =
            ggml_u16_htp_round_shift(scaled_sum, 15) + output_zero_point;
        output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
    }
}

static inline uint16_t ggml_u16_mul_requant_fixed(
        int64_t lhs,
        int64_t rhs,
        int64_t multiplier,
        int32_t output_zero_point) {
    const int64_t product = lhs * rhs;
    GGML_ASSERT(multiplier >= 0);
    GGML_ASSERT(product == 0 || multiplier <= INT64_MAX / llabs(product));
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        product * multiplier, GGML_U16_Q20_SHIFT) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_mul_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int32_t rhs_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(product_to_output_q20 >= 0);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t lhs_zero = vdupq_n_s32(lhs_zero_point);
    const int32x4_t rhs_zero = vdupq_n_s32(rhs_zero_point);
    for (; index + 4 <= n; index += 4) {
        const int32x4_t lhs_values = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(lhs + index))), lhs_zero);
        const int32x4_t rhs_values = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(rhs + index))), rhs_zero);
        const int64x2_t product_lo = vmull_s32(
            vget_low_s32(lhs_values), vget_low_s32(rhs_values));
        const int64x2_t product_hi = vmull_s32(
            vget_high_s32(lhs_values), vget_high_s32(rhs_values));
        output[index + 0] = ggml_u16_mul_requant_fixed(
            vgetq_lane_s64(product_lo, 0), 1, product_to_output_q20, output_zero_point);
        output[index + 1] = ggml_u16_mul_requant_fixed(
            vgetq_lane_s64(product_lo, 1), 1, product_to_output_q20, output_zero_point);
        output[index + 2] = ggml_u16_mul_requant_fixed(
            vgetq_lane_s64(product_hi, 0), 1, product_to_output_q20, output_zero_point);
        output[index + 3] = ggml_u16_mul_requant_fixed(
            vgetq_lane_s64(product_hi, 1), 1, product_to_output_q20, output_zero_point);
    }
#endif
    for (; index < n; ++index) {
        output[index] = ggml_u16_mul_requant_fixed(
            (int32_t) lhs[index] - lhs_zero_point,
            (int32_t) rhs[index] - rhs_zero_point,
            product_to_output_q20,
            output_zero_point);
    }
}

static inline uint16_t ggml_u16_add_requant_fixed(
        int64_t lhs,
        int64_t lhs_multiplier,
        int64_t rhs,
        int64_t rhs_multiplier,
        int32_t output_zero_point);

void ggml_vec_add_affine_u16_qnn_fixed_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_multiplier,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int64_t rhs_multiplier,
        int32_t rhs_zero_point,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    const int64_t rhs_value = (int32_t) rhs_code - rhs_zero_point;
    for (int index = 0; index < n; ++index) {
        const int64_t lhs_value = (int32_t) lhs[index] - lhs_zero_point;
        output[index] = ggml_u16_add_requant_fixed(
            lhs_value, lhs_multiplier, rhs_value, rhs_multiplier,
            output_zero_point);
    }
}

void ggml_vec_mul_affine_u16_qnn_fixed_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int32_t rhs_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    const int64_t rhs_value = (int32_t) rhs_code - rhs_zero_point;
    for (int index = 0; index < n; ++index) {
        output[index] = ggml_u16_mul_requant_fixed(
            (int32_t) lhs[index] - lhs_zero_point,
            rhs_value, product_to_output_q20, output_zero_point);
    }
}

static inline uint16_t ggml_u16_add_requant_fixed(
        int64_t lhs,
        int64_t lhs_multiplier,
        int64_t rhs,
        int64_t rhs_multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(lhs == 0 || llabs(lhs_multiplier) <= INT64_MAX / llabs(lhs));
    GGML_ASSERT(rhs == 0 || llabs(rhs_multiplier) <= INT64_MAX / llabs(rhs));
    const int64_t lhs_scaled = lhs * lhs_multiplier;
    const int64_t rhs_scaled = rhs * rhs_multiplier;
    GGML_ASSERT((rhs_scaled <= 0 || lhs_scaled <= INT64_MAX - rhs_scaled) &&
                (rhs_scaled >= 0 || lhs_scaled >= INT64_MIN - rhs_scaled));
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        lhs_scaled + rhs_scaled, GGML_U16_Q20_SHIFT) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_requant_q31(
        int64_t value,
        int64_t multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(value == 0 || llabs(multiplier) <= INT64_MAX / llabs(value));
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        value * multiplier, 31) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_mul_product_requant_q31(
        int64_t value,
        int64_t multiplier,
        int64_t nudge,
        int32_t output_zero_point) {
    GGML_ASSERT(value == 0 || llabs(multiplier) <= INT64_MAX / llabs(value));
    // QNN HTP ElementWiseMultiply switches to the upper output code below
    // the conventional half-way point. Preserve that requantization behavior
    // in the integer path instead of correcting the output afterwards.
    const int64_t quantized =
        ggml_u16_floor_shift(value * multiplier + nudge, 31) +
        output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_swiglu_qnn_q31_one(
        uint16_t gate,
        uint16_t up,
        const uint16_t * GGML_RESTRICT sigmoid_lut,
        int32_t silu_lhs_zero_point,
        int32_t silu_rhs_zero_point,
        int64_t silu_product_to_output_q31,
        int64_t silu_product_requant_nudge_q31,
        int32_t silu_output_zero_point,
        int32_t product_lhs_zero_point,
        int32_t product_rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t product_output_zero_point) {
    const int64_t gate_value = (int32_t) gate - silu_lhs_zero_point;
    const int64_t sigmoid_value =
        (int32_t) sigmoid_lut[gate] - silu_rhs_zero_point;
    const uint16_t silu_code = ggml_u16_mul_product_requant_q31(
        gate_value * sigmoid_value, silu_product_to_output_q31,
        silu_product_requant_nudge_q31, silu_output_zero_point);
    const int64_t silu_value = (int32_t) silu_code - product_lhs_zero_point;
    const int64_t up_value = (int32_t) up - product_rhs_zero_point;
    return ggml_u16_mul_product_requant_q31(
        silu_value * up_value, product_to_output_q31,
        product_requant_nudge_q31, product_output_zero_point);
}

static inline uint16_t ggml_u16_mul_requant_q31(
        int64_t lhs,
        int64_t rhs,
        int64_t multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(lhs == 0 || llabs(rhs) <= INT64_MAX / llabs(lhs));
    return ggml_u16_requant_q31(lhs * rhs, multiplier, output_zero_point);
}

static inline uint16_t ggml_u16_mul_requant_htp_shift3(
        int64_t lhs,
        int64_t rhs,
        int64_t multiplier_q31_shift3,
        int32_t output_zero_point) {
    GGML_ASSERT(lhs == 0 || llabs(rhs) <= INT64_MAX / llabs(lhs));
    const int64_t product = lhs * rhs;
    GGML_ASSERT(product == 0 ||
        llabs(multiplier_q31_shift3) <= INT64_MAX / llabs(product));
    const int64_t q31 = ggml_u16_htp_round_shift(
        product * multiplier_q31_shift3, 31);
    const int64_t quantized =
        ggml_u16_htp_round_shift(q31, 3) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_binary_requant_htp_q15(
        uint16_t lhs,
        int32_t lhs_multiplier_q15,
        uint16_t rhs,
        int32_t rhs_multiplier_q15,
        int32_t shift,
        int64_t bias,
        bool subtract) {
    GGML_ASSERT(lhs_multiplier_q15 >= 0 && lhs_multiplier_q15 <= INT16_MAX);
    GGML_ASSERT(rhs_multiplier_q15 >= 0 && rhs_multiplier_q15 <= INT16_MAX);
    GGML_ASSERT(shift > 0 && shift < 31);
    const int64_t lhs_scaled = (int64_t) lhs * lhs_multiplier_q15;
    const int64_t rhs_scaled = (int64_t) rhs * rhs_multiplier_q15;
    const int64_t average = ggml_u16_floor_shift(
        lhs_scaled + (subtract ? -rhs_scaled : rhs_scaled), 1);
    const int64_t quantized = ggml_u16_htp_round_shift(
        average + bias, shift);
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_mul_affine_u16_qnn_q31(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int32_t rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(product_to_output_q31 >= 0);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t lhs_zero = vdupq_n_s32(lhs_zero_point);
    const int32x4_t rhs_zero = vdupq_n_s32(rhs_zero_point);
    for (; index + 4 <= n; index += 4) {
        const int32x4_t lhs_values = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(lhs + index))), lhs_zero);
        const int32x4_t rhs_values = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(rhs + index))), rhs_zero);
        const int64x2_t product_lo = vmull_s32(
            vget_low_s32(lhs_values), vget_low_s32(rhs_values));
        const int64x2_t product_hi = vmull_s32(
            vget_high_s32(lhs_values), vget_high_s32(rhs_values));
        output[index + 0] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_lo, 0), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 1] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_lo, 1), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 2] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_hi, 0), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 3] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_hi, 1), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
    }
#endif
    for (; index < n; ++index) {
        const int64_t lhs_value = (int32_t) lhs[index] - lhs_zero_point;
        const int64_t rhs_value = (int32_t) rhs[index] - rhs_zero_point;
        GGML_ASSERT(lhs_value == 0 ||
            llabs(rhs_value) <= INT64_MAX / llabs(lhs_value));
        output[index] = ggml_u16_mul_product_requant_q31(
            lhs_value * rhs_value, product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
    }
}

void ggml_vec_swiglu_u16_qnn_q31(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT gate,
        const uint16_t * GGML_RESTRICT up,
        const uint16_t * GGML_RESTRICT sigmoid_lut,
        int32_t silu_lhs_zero_point,
        int32_t silu_rhs_zero_point,
        int64_t silu_product_to_output_q31,
        int64_t silu_product_requant_nudge_q31,
        int32_t silu_output_zero_point,
        int32_t product_lhs_zero_point,
        int32_t product_rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t product_output_zero_point) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(output != NULL && gate != NULL && up != NULL && sigmoid_lut != NULL);
    GGML_ASSERT(silu_product_to_output_q31 >= 0 && product_to_output_q31 >= 0);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t silu_lhs_zero = vdupq_n_s32(silu_lhs_zero_point);
    const int32x4_t silu_rhs_zero = vdupq_n_s32(silu_rhs_zero_point);
    const int32x4_t product_lhs_zero = vdupq_n_s32(product_lhs_zero_point);
    const int32x4_t product_rhs_zero = vdupq_n_s32(product_rhs_zero_point);
    for (; index + 8 <= n; index += 8) {
        const uint16x8_t gate_codes = vld1q_u16(gate + index);
        const uint16_t sigmoid_codes_data[8] = {
            sigmoid_lut[gate[index + 0]], sigmoid_lut[gate[index + 1]],
            sigmoid_lut[gate[index + 2]], sigmoid_lut[gate[index + 3]],
            sigmoid_lut[gate[index + 4]], sigmoid_lut[gate[index + 5]],
            sigmoid_lut[gate[index + 6]], sigmoid_lut[gate[index + 7]],
        };
        const uint16x8_t sigmoid_codes = vld1q_u16(sigmoid_codes_data);
        const int32x4_t gate_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(gate_codes))),
            silu_lhs_zero);
        const int32x4_t gate_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(gate_codes))),
            silu_lhs_zero);
        const int32x4_t sigmoid_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sigmoid_codes))),
            silu_rhs_zero);
        const int32x4_t sigmoid_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sigmoid_codes))),
            silu_rhs_zero);
        const int64x2_t silu_product_0 =
            vmull_s32(vget_low_s32(gate_lo), vget_low_s32(sigmoid_lo));
        const int64x2_t silu_product_1 =
            vmull_s32(vget_high_s32(gate_lo), vget_high_s32(sigmoid_lo));
        const int64x2_t silu_product_2 =
            vmull_s32(vget_low_s32(gate_hi), vget_low_s32(sigmoid_hi));
        const int64x2_t silu_product_3 =
            vmull_s32(vget_high_s32(gate_hi), vget_high_s32(sigmoid_hi));
        uint16_t silu_codes_data[8];
        silu_codes_data[0] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_0, 0), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[1] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_0, 1), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[2] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_1, 0), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[3] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_1, 1), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[4] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_2, 0), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[5] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_2, 1), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[6] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_3, 0), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        silu_codes_data[7] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(silu_product_3, 1), silu_product_to_output_q31,
            silu_product_requant_nudge_q31, silu_output_zero_point);
        const uint16x8_t silu_codes = vld1q_u16(silu_codes_data);
        const uint16x8_t up_codes = vld1q_u16(up + index);
        const int32x4_t silu_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(silu_codes))),
            product_lhs_zero);
        const int32x4_t silu_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(silu_codes))),
            product_lhs_zero);
        const int32x4_t up_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(up_codes))),
            product_rhs_zero);
        const int32x4_t up_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(up_codes))),
            product_rhs_zero);
        const int64x2_t output_product_0 =
            vmull_s32(vget_low_s32(silu_lo), vget_low_s32(up_lo));
        const int64x2_t output_product_1 =
            vmull_s32(vget_high_s32(silu_lo), vget_high_s32(up_lo));
        const int64x2_t output_product_2 =
            vmull_s32(vget_low_s32(silu_hi), vget_low_s32(up_hi));
        const int64x2_t output_product_3 =
            vmull_s32(vget_high_s32(silu_hi), vget_high_s32(up_hi));
        output[index + 0] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_0, 0), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 1] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_0, 1), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 2] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_1, 0), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 3] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_1, 1), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 4] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_2, 0), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 5] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_2, 1), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 6] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_3, 0), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 7] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(output_product_3, 1), product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
    }
#endif
    for (; index + 8 <= n; index += 8) {
        output[index + 0] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 0], up[index + 0], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 1] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 1], up[index + 1], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 2] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 2], up[index + 2], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 3] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 3], up[index + 3], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 4] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 4], up[index + 4], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 5] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 5], up[index + 5], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 6] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 6], up[index + 6], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
        output[index + 7] = ggml_u16_swiglu_qnn_q31_one(
            gate[index + 7], up[index + 7], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
    }
    for (; index < n; ++index) {
        output[index] = ggml_u16_swiglu_qnn_q31_one(
            gate[index], up[index], sigmoid_lut,
            silu_lhs_zero_point, silu_rhs_zero_point,
            silu_product_to_output_q31, silu_product_requant_nudge_q31,
            silu_output_zero_point, product_lhs_zero_point,
            product_rhs_zero_point, product_to_output_q31,
            product_requant_nudge_q31, product_output_zero_point);
    }
}

void ggml_vec_mul_affine_u16_qnn_q31_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int32_t rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    const int32_t rhs_value = (int32_t) rhs_code - rhs_zero_point;
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t lhs_zero = vdupq_n_s32(lhs_zero_point);
    const int32x2_t rhs_pair = vdup_n_s32(rhs_value);
    for (; index + 4 <= n; index += 4) {
        const int32x4_t lhs_values = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(lhs + index))), lhs_zero);
        const int64x2_t product_lo = vmull_s32(
            vget_low_s32(lhs_values), rhs_pair);
        const int64x2_t product_hi = vmull_s32(
            vget_high_s32(lhs_values), rhs_pair);
        output[index + 0] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_lo, 0), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 1] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_lo, 1), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 2] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_hi, 0), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
        output[index + 3] = ggml_u16_mul_product_requant_q31(
            vgetq_lane_s64(product_hi, 1), product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
    }
#endif
    for (; index < n; ++index) {
        const int64_t lhs_value = (int32_t) lhs[index] - lhs_zero_point;
        GGML_ASSERT(lhs_value == 0 ||
            llabs(rhs_value) <= INT64_MAX / llabs(lhs_value));
        output[index] = ggml_u16_mul_product_requant_q31(
            lhs_value * rhs_value, product_to_output_q31,
            product_requant_nudge_q31, output_zero_point);
    }
}

void ggml_vec_mul_static_affine_u16_qnn_q15(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        int32_t multiplier_q15,
        int32_t right_shift,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0);
    GGML_ASSERT(multiplier_q15 >= INT16_MIN &&
        multiplier_q15 <= INT16_MAX);
    GGML_ASSERT(right_shift > 0 && right_shift < 31);
    for (int index = 0; index < n; ++index) {
        const int64_t centered =
            (int32_t) input[index] - input_zero_point;
        const int64_t quantized = ggml_u16_htp_round_shift(
            centered * multiplier_q15, right_shift) + output_zero_point;
        output[index] = (uint16_t) MAX(
            0, MIN(UINT16_MAX, quantized));
    }
}

static inline void ggml_u16_rope_requant_pair_storage_tables(
        uint16_t * GGML_RESTRICT output_first,
        uint16_t * GGML_RESTRICT output_second,
        int64_t input_first,
        int64_t input_second,
        uint16_t table_code0,
        uint16_t table_code1,
        uint16_t table_code2,
        uint16_t table_code3,
        const struct ggml_u16_rope_qnn_fixed_params * params) {
    const uint16_t split_first = (uint16_t) MAX(0, MIN(UINT16_MAX,
        ggml_gptq2_32_round_shift_away_from_zero(
            (input_first - params->split_input_zero_point) *
                params->split_to_output_q31[0],
            31) + params->split_output_zero_points[0]));
    const uint16_t split_second = (uint16_t) MAX(0, MIN(UINT16_MAX,
        ggml_gptq2_32_round_shift_away_from_zero(
            (input_second - params->split_input_zero_point) *
                params->split_to_output_q31[1],
            31) + params->split_output_zero_points[1]));
    const uint16_t table_codes[4] = {
        table_code0,
        table_code1,
        table_code2,
        table_code3,
    };
    const uint16_t products[4] = {
        ggml_u16_mul_requant_htp_shift3(
            (int64_t) split_first - params->lhs_zero_points[0],
            (int64_t) table_codes[0] - params->table_zero_points[0],
            params->product_to_output_q31_shift3[0],
            params->product_output_zero_points[0]),
        ggml_u16_mul_requant_htp_shift3(
            (int64_t) split_second - params->lhs_zero_points[1],
            (int64_t) table_codes[1] - params->table_zero_points[1],
            params->product_to_output_q31_shift3[1],
            params->product_output_zero_points[1]),
        ggml_u16_mul_requant_htp_shift3(
            (int64_t) split_first - params->lhs_zero_points[2],
            (int64_t) table_codes[2] - params->table_zero_points[2],
            params->product_to_output_q31_shift3[2],
            params->product_output_zero_points[2]),
        ggml_u16_mul_requant_htp_shift3(
            (int64_t) split_second - params->lhs_zero_points[3],
            (int64_t) table_codes[3] - params->table_zero_points[3],
            params->product_to_output_q31_shift3[3],
            params->product_output_zero_points[3]),
    };
    *output_first = ggml_u16_binary_requant_htp_q15(
        products[0], params->combine_multipliers_q15[0],
        products[1], params->combine_multipliers_q15[1],
        params->combine_shifts[0], params->combine_biases[0], true);
    *output_second = ggml_u16_binary_requant_htp_q15(
        products[2], params->combine_multipliers_q15[2],
        products[3], params->combine_multipliers_q15[3],
        params->combine_shifts[1], params->combine_biases[1], false);
}

#if defined(__ARM_NEON)
static inline int64_t ggml_u16_max_centered_code(int32_t zero_point) {
    return MAX((int64_t) zero_point, (int64_t) UINT16_MAX - zero_point);
}

static inline int32x4_t ggml_u16_requant_i32x4_q31(
        int32x4_t values,
        int32_t multiplier,
        int32_t output_zero_point) {
    const int32x2_t multiplier_pair = vdup_n_s32(multiplier);
    const int64x2_t products_lo = vmull_s32(
        vget_low_s32(values), multiplier_pair);
    const int64x2_t products_hi = vmull_s32(
        vget_high_s32(values), multiplier_pair);
    int32x4_t result = vdupq_n_s32(0);
    result = vsetq_lane_s32(ggml_u16_requant_q31(
        vgetq_lane_s64(products_lo, 0), 1, output_zero_point), result, 0);
    result = vsetq_lane_s32(ggml_u16_requant_q31(
        vgetq_lane_s64(products_lo, 1), 1, output_zero_point), result, 1);
    result = vsetq_lane_s32(ggml_u16_requant_q31(
        vgetq_lane_s64(products_hi, 0), 1, output_zero_point), result, 2);
    result = vsetq_lane_s32(ggml_u16_requant_q31(
        vgetq_lane_s64(products_hi, 1), 1, output_zero_point), result, 3);
    return result;
}

static inline int32x4_t ggml_u16_requant_i32x4_htp_shift3(
        int32x4_t values,
        int32_t multiplier_q31_shift3,
        int32_t output_zero_point) {
    int32x4_t result = vdupq_n_s32(0);
    result = vsetq_lane_s32(ggml_u16_mul_requant_htp_shift3(
        vgetq_lane_s32(values, 0), 1, multiplier_q31_shift3,
        output_zero_point), result, 0);
    result = vsetq_lane_s32(ggml_u16_mul_requant_htp_shift3(
        vgetq_lane_s32(values, 1), 1, multiplier_q31_shift3,
        output_zero_point), result, 1);
    result = vsetq_lane_s32(ggml_u16_mul_requant_htp_shift3(
        vgetq_lane_s32(values, 2), 1, multiplier_q31_shift3,
        output_zero_point), result, 2);
    result = vsetq_lane_s32(ggml_u16_mul_requant_htp_shift3(
        vgetq_lane_s32(values, 3), 1, multiplier_q31_shift3,
        output_zero_point), result, 3);
    return result;
}

static inline int32x4_t ggml_u16_binary_requant_i32x4_htp_q15(
        int32x4_t lhs,
        int32_t lhs_multiplier_q15,
        int32x4_t rhs,
        int32_t rhs_multiplier_q15,
        int32_t shift,
        int64_t bias,
        bool subtract) {
    int32x4_t result = vdupq_n_s32(0);
    result = vsetq_lane_s32(ggml_u16_binary_requant_htp_q15(
        (uint16_t) vgetq_lane_s32(lhs, 0), lhs_multiplier_q15,
        (uint16_t) vgetq_lane_s32(rhs, 0), rhs_multiplier_q15,
        shift, bias, subtract), result, 0);
    result = vsetq_lane_s32(ggml_u16_binary_requant_htp_q15(
        (uint16_t) vgetq_lane_s32(lhs, 1), lhs_multiplier_q15,
        (uint16_t) vgetq_lane_s32(rhs, 1), rhs_multiplier_q15,
        shift, bias, subtract), result, 1);
    result = vsetq_lane_s32(ggml_u16_binary_requant_htp_q15(
        (uint16_t) vgetq_lane_s32(lhs, 2), lhs_multiplier_q15,
        (uint16_t) vgetq_lane_s32(rhs, 2), rhs_multiplier_q15,
        shift, bias, subtract), result, 2);
    result = vsetq_lane_s32(ggml_u16_binary_requant_htp_q15(
        (uint16_t) vgetq_lane_s32(lhs, 3), lhs_multiplier_q15,
        (uint16_t) vgetq_lane_s32(rhs, 3), rhs_multiplier_q15,
        shift, bias, subtract), result, 3);
    return result;
}

static inline bool ggml_u16_rope_direct_neon_is_safe(
        const struct ggml_u16_rope_qnn_fixed_params * params) {
    if (params->split_to_output_q31[0] != (INT64_C(1) << 31) ||
        params->split_input_zero_point != params->split_output_zero_points[0]) {
        return false;
    }
    for (int operation = 0; operation < 4; ++operation) {
        if (params->product_to_output_q31_shift3[operation] > INT32_MAX) {
            return false;
        }
        const int64_t lhs_max =
            ggml_u16_max_centered_code(params->lhs_zero_points[operation]);
        const int64_t table_max =
            ggml_u16_max_centered_code(params->table_zero_points[operation]);
        if (lhs_max * table_max > INT32_MAX) {
            return false;
        }
    }
    return true;
}

static inline void ggml_u16_rope_requant_four_direct_neon(
        uint16_t * GGML_RESTRICT output_first,
        uint16_t * GGML_RESTRICT output_second,
        const uint16_t * GGML_RESTRICT input_first,
        const uint16_t * GGML_RESTRICT input_second,
        const uint16_t * GGML_RESTRICT cos_codes,
        const uint16_t * GGML_RESTRICT sin_codes,
        const struct ggml_u16_rope_qnn_fixed_params * params) {
    const int32x4_t first = vreinterpretq_s32_u32(
        vmovl_u16(vld1_u16(input_first)));
    const uint16x4_t second_codes = vld1_u16(input_second);
    int32x4_t second = vdupq_n_s32(0);
    second = vsetq_lane_s32(ggml_u16_requant_q31(
        (int64_t) vget_lane_u16(second_codes, 0) - params->split_input_zero_point,
        params->split_to_output_q31[1],
        params->split_output_zero_points[1]), second, 0);
    second = vsetq_lane_s32(ggml_u16_requant_q31(
        (int64_t) vget_lane_u16(second_codes, 1) - params->split_input_zero_point,
        params->split_to_output_q31[1],
        params->split_output_zero_points[1]), second, 1);
    second = vsetq_lane_s32(ggml_u16_requant_q31(
        (int64_t) vget_lane_u16(second_codes, 2) - params->split_input_zero_point,
        params->split_to_output_q31[1],
        params->split_output_zero_points[1]), second, 2);
    second = vsetq_lane_s32(ggml_u16_requant_q31(
        (int64_t) vget_lane_u16(second_codes, 3) - params->split_input_zero_point,
        params->split_to_output_q31[1],
        params->split_output_zero_points[1]), second, 3);
    const int32x4_t cos_values = vreinterpretq_s32_u32(
        vmovl_u16(vld1_u16(cos_codes)));
    const int32x4_t sin_values = vreinterpretq_s32_u32(
        vmovl_u16(vld1_u16(sin_codes)));
    const int32x4_t products[4] = {
        ggml_u16_requant_i32x4_htp_shift3(
            vmulq_s32(
                vsubq_s32(first, vdupq_n_s32(params->lhs_zero_points[0])),
                vsubq_s32(cos_values, vdupq_n_s32(params->table_zero_points[0]))),
            (int32_t) params->product_to_output_q31_shift3[0],
            params->product_output_zero_points[0]),
        ggml_u16_requant_i32x4_htp_shift3(
            vmulq_s32(
                vsubq_s32(second, vdupq_n_s32(params->lhs_zero_points[1])),
                vsubq_s32(sin_values, vdupq_n_s32(params->table_zero_points[1]))),
            (int32_t) params->product_to_output_q31_shift3[1],
            params->product_output_zero_points[1]),
        ggml_u16_requant_i32x4_htp_shift3(
            vmulq_s32(
                vsubq_s32(first, vdupq_n_s32(params->lhs_zero_points[2])),
                vsubq_s32(sin_values, vdupq_n_s32(params->table_zero_points[2]))),
            (int32_t) params->product_to_output_q31_shift3[2],
            params->product_output_zero_points[2]),
        ggml_u16_requant_i32x4_htp_shift3(
            vmulq_s32(
                vsubq_s32(second, vdupq_n_s32(params->lhs_zero_points[3])),
                vsubq_s32(cos_values, vdupq_n_s32(params->table_zero_points[3]))),
            (int32_t) params->product_to_output_q31_shift3[3],
            params->product_output_zero_points[3]),
    };
    const int32x4_t combined_first =
        ggml_u16_binary_requant_i32x4_htp_q15(
            products[0], params->combine_multipliers_q15[0],
            products[1], params->combine_multipliers_q15[1],
            params->combine_shifts[0], params->combine_biases[0], true);
    const int32x4_t combined_second =
        ggml_u16_binary_requant_i32x4_htp_q15(
            products[2], params->combine_multipliers_q15[2],
            products[3], params->combine_multipliers_q15[3],
            params->combine_shifts[1], params->combine_biases[1], false);
    vst1_u16(output_first, vqmovun_s32(combined_first));
    vst1_u16(output_second, vqmovun_s32(combined_second));
}
#endif

static inline void ggml_u16_rope_requant_pair(
        uint16_t * GGML_RESTRICT output_first,
        uint16_t * GGML_RESTRICT output_second,
        int64_t input_first,
        int64_t input_second,
        int64_t cos_code,
        int64_t sin_code,
        const struct ggml_u16_rope_qnn_fixed_params * params) {
    const uint16_t table_codes[4] = {
        (uint16_t) MAX(0, MIN(UINT16_MAX,
            ggml_gptq2_32_round_shift_away_from_zero(
                (cos_code - params->table_source_zero_points[0]) *
                    params->table_source_to_storage_q31[0],
                31) + params->table_zero_points[0])),
        (uint16_t) MAX(0, MIN(UINT16_MAX,
            ggml_gptq2_32_round_shift_away_from_zero(
                (sin_code - params->table_source_zero_points[1]) *
                    params->table_source_to_storage_q31[1],
                31) + params->table_zero_points[1])),
        (uint16_t) MAX(0, MIN(UINT16_MAX,
            ggml_gptq2_32_round_shift_away_from_zero(
                (sin_code - params->table_source_zero_points[2]) *
                    params->table_source_to_storage_q31[2],
                31) + params->table_zero_points[2])),
        (uint16_t) MAX(0, MIN(UINT16_MAX,
            ggml_gptq2_32_round_shift_away_from_zero(
                (cos_code - params->table_source_zero_points[3]) *
                    params->table_source_to_storage_q31[3],
                31) + params->table_zero_points[3])),
    };
    ggml_u16_rope_requant_pair_storage_tables(
        output_first,
        output_second,
        input_first,
        input_second,
        table_codes[0],
        table_codes[1],
        table_codes[2],
        table_codes[3],
        params);
}

void ggml_vec_rope_affine_u16_qnn_fixed(
        int half_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint16_t * GGML_RESTRICT cos_codes,
        const uint16_t * GGML_RESTRICT sin_codes,
        const struct ggml_u16_rope_qnn_fixed_params * params) {
    GGML_ASSERT(half_dimension > 0);
    GGML_ASSERT(output != NULL && input != NULL);
    GGML_ASSERT(cos_codes != NULL && sin_codes != NULL && params != NULL);
    for (int parameter = 0; parameter < 4; ++parameter) {
        GGML_ASSERT(params->table_source_to_storage_q31[parameter] >= 0);
        GGML_ASSERT(params->product_to_output_q31_shift3[parameter] >= 0);
        GGML_ASSERT(params->combine_multipliers_q15[parameter] >= 0);
        GGML_ASSERT(params->combine_multipliers_q15[parameter] <= INT16_MAX);
    }
    for (int output_index = 0; output_index < 2; ++output_index) {
        GGML_ASSERT(params->combine_shifts[output_index] > 0);
        GGML_ASSERT(params->combine_shifts[output_index] < 31);
    }

    int index = 0;
    bool direct_tables = true;
    for (int parameter = 0; parameter < 4; ++parameter) {
        direct_tables =
            direct_tables &&
            params->table_source_to_storage_q31[parameter] == (INT64_C(1) << 31) &&
            params->table_source_zero_points[parameter] ==
                params->table_zero_points[parameter];
    }
    if (direct_tables) {
#if defined(__ARM_NEON)
        if (ggml_u16_rope_direct_neon_is_safe(params)) {
            for (; index + 4 <= half_dimension; index += 4) {
                ggml_u16_rope_requant_four_direct_neon(
                    output + index,
                    output + half_dimension + index,
                    input + index,
                    input + half_dimension + index,
                    cos_codes + index,
                    sin_codes + index,
                    params);
            }
        }
#endif
        for (; index < half_dimension; ++index) {
            ggml_u16_rope_requant_pair_storage_tables(
                output + index,
                output + half_dimension + index,
                input[index],
                input[half_dimension + index],
                cos_codes[index],
                sin_codes[index],
                sin_codes[index],
                cos_codes[index],
                params);
        }
        return;
    }
#if defined(__ARM_NEON)
    for (; index + 4 <= half_dimension; index += 4) {
        const uint16x4_t first = vld1_u16(input + index);
        const uint16x4_t second = vld1_u16(input + half_dimension + index);
        const uint16x4_t cos_values = vld1_u16(cos_codes + index);
        const uint16x4_t sin_values = vld1_u16(sin_codes + index);
#define GGML_U16_ROPE_LANE(LANE)                                      \
        ggml_u16_rope_requant_pair(                                   \
            output + index + (LANE),                                  \
            output + half_dimension + index + (LANE),                 \
            vget_lane_u16(first, (LANE)),                              \
            vget_lane_u16(second, (LANE)),                             \
            vget_lane_u16(cos_values, (LANE)),                         \
            vget_lane_u16(sin_values, (LANE)),                         \
            params)
        GGML_U16_ROPE_LANE(0);
        GGML_U16_ROPE_LANE(1);
        GGML_U16_ROPE_LANE(2);
        GGML_U16_ROPE_LANE(3);
#undef GGML_U16_ROPE_LANE
    }
#endif
    for (; index < half_dimension; ++index) {
        ggml_u16_rope_requant_pair(
            output + index,
            output + half_dimension + index,
            input[index],
            input[half_dimension + index],
            cos_codes[index],
            sin_codes[index],
            params);
    }
}

static inline uint16_t ggml_u16_matmul_requant_q31(
        int64_t accumulator,
        int64_t multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(multiplier >= 0);
    GGML_ASSERT(accumulator == 0 ||
        multiplier <= INT64_MAX / llabs(accumulator));
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        accumulator * multiplier, 31) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_htp_matmul_requant_q31(
        int64_t accumulator,
        int64_t multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(multiplier >= 0);
    GGML_ASSERT(accumulator == 0 ||
        multiplier <= INT64_MAX / llabs(accumulator));
    const int64_t quantized = ggml_u16_floor_shift(
        accumulator * multiplier, 31) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

static inline uint16_t ggml_u16_htp_matmul_requant_symmetric_u16(
        int64_t accumulator,
        int64_t multiplier,
        int32_t output_zero_point) {
    GGML_ASSERT(multiplier >= 0);
    GGML_ASSERT(accumulator == 0 ||
        multiplier <= INT64_MAX / llabs(accumulator));
    const int64_t scaled = ggml_u16_htp_round_shift(
        accumulator * multiplier, 31);
    const int64_t symmetric_s16 = MAX(INT16_MIN, MIN(INT16_MAX, scaled));
    const int64_t public_code = symmetric_s16 + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, public_code));
}

static void ggml_vec_hadamard_128_u16_s16_qnn_fixed(
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        int64_t product_to_output_q24,
        int32_t output_zero_point) {
    int32_t transformed[128];
    int64_t input_sum = 0;
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
    for (; index + 8 <= 128; index += 8) {
        const uint16x8_t packed = vld1q_u16(input + index);
        const int32x4_t low = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(packed))), input_zero);
        const int32x4_t high = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(packed))), input_zero);
        vst1q_s32(transformed + index, low);
        vst1q_s32(transformed + index + 4, high);
        input_sum += vaddvq_s32(low);
        input_sum += vaddvq_s32(high);
    }
#endif
    for (; index < 128; ++index) {
        transformed[index] = (int32_t) input[index] - input_zero_point;
        input_sum += transformed[index];
    }

    for (int width = 1; width < 128; width <<= 1) {
        for (int base = 0; base < 128; base += width << 1) {
            int lane = 0;
#if defined(__ARM_NEON)
            for (; lane + 4 <= width; lane += 4) {
                const int32x4_t first = vld1q_s32(transformed + base + lane);
                const int32x4_t second =
                    vld1q_s32(transformed + base + width + lane);
                vst1q_s32(transformed + base + lane, vaddq_s32(first, second));
                vst1q_s32(
                    transformed + base + width + lane, vsubq_s32(first, second));
            }
#endif
            for (; lane < width; ++lane) {
                const int32_t first = transformed[base + lane];
                const int32_t second = transformed[base + width + lane];
                transformed[base + lane] = first + second;
                transformed[base + width + lane] = first - second;
            }
        }
    }

    for (int column = 0; column < 128; ++column) {
        // ConvLayer_s1.opt reconstructs +32767 as (127, 127) = 32639,
        // while -32767 remains (1, -128) = -32767. For Hadamard sign h,
        // the resulting coefficient is 32703*h - 64.
        const int64_t centered =
            INT64_C(32703) * transformed[column] - INT64_C(64) * input_sum;
        const int64_t weight_sum =
            column == 0 ? INT64_C(4177792) : -8192;
        const int64_t correction =
            (int64_t) input_zero_point * weight_sum;

        // HTP reduces the unsigned raw dot and affine correction separately
        // to S16 before subtracting. The final scale is prepared as Q24 for
        // this reduced domain, preserving all runtime intermediates as ints.
        const int64_t reduced =
            ggml_u16_floor_shift(centered + correction, 16)
            - ggml_u16_htp_round_shift(correction, 16);
        GGML_ASSERT(reduced == 0 ||
            product_to_output_q24 <= INT64_MAX / llabs(reduced));
        const int64_t scaled = ggml_u16_htp_round_shift(
            reduced * product_to_output_q24, 24);
        const int64_t public_code = scaled + output_zero_point;
        output[column] =
            (uint16_t) MAX(0, MIN(UINT16_MAX, public_code));
    }
}

void ggml_vec_matmul_u16_s16_qnn_fixed(
        int input_dimension,
        int output_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const int16_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point,
        int weight_right_shift,
        int accumulator_reduction_shift) {
    GGML_ASSERT(input_dimension > 0 && output_dimension > 0);
    GGML_ASSERT(output != NULL && input != NULL && weights != NULL);
    GGML_ASSERT(product_to_output_q20 >= 0);
    GGML_ASSERT(weight_right_shift == 0 || weight_right_shift == 7);
    GGML_ASSERT(accumulator_reduction_shift >= 0 &&
        accumulator_reduction_shift < 31);
    GGML_ASSERT(accumulator_reduction_shift == 0 ||
        (weight_right_shift == 7 &&
         input_dimension == 128 && output_dimension == 128));
    GGML_ASSERT(weight_right_shift == 0 || weight_zero_point == 0);
    const bool use_hadamard_fast_path =
        weight_right_shift == 7 &&
        input_dimension == 128 && output_dimension == 128;

    if (use_hadamard_fast_path) {
        GGML_ASSERT(accumulator_reduction_shift == 16);
        ggml_vec_hadamard_128_u16_s16_qnn_fixed(
            output, input, input_zero_point, product_to_output_q20,
            output_zero_point);
        return;
    }

    GGML_ASSERT(product_to_output_q20 <= (INT64_MAX >> weight_right_shift));
    const int64_t effective_multiplier =
        product_to_output_q20 << weight_right_shift;

    int column = 0;
#if defined(__ARM_NEON)
    const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
    for (; column + 8 <= output_dimension; column += 8) {
        int64x2_t accumulators[4] = {
            vdupq_n_s64(0), vdupq_n_s64(0),
            vdupq_n_s64(0), vdupq_n_s64(0),
        };
        for (int row = 0; row < input_dimension; ++row) {
            const int32_t input_value = (int32_t) input[row] - input_zero_point;
            const int32x2_t input_pair = vdup_n_s32(input_value);
            int16x8_t packed = vld1q_s16(
                weights + (size_t) row * output_dimension + column);
            if (weight_right_shift == 7) {
                // QNN HTP narrows this S16 Hadamard operand to signed 9-bit
                // lanes before multiplication: +32767/-32767 -> +255/-256.
                packed = vshrq_n_s16(packed, 7);
            }
            const int32x4_t low = vsubq_s32(
                vmovl_s16(vget_low_s16(packed)), weight_zero);
            const int32x4_t high = vsubq_s32(
                vmovl_s16(vget_high_s16(packed)), weight_zero);
            accumulators[0] = vmlal_s32(
                accumulators[0], vget_low_s32(low), input_pair);
            accumulators[1] = vmlal_s32(
                accumulators[1], vget_high_s32(low), input_pair);
            accumulators[2] = vmlal_s32(
                accumulators[2], vget_low_s32(high), input_pair);
            accumulators[3] = vmlal_s32(
                accumulators[3], vget_high_s32(high), input_pair);
        }
#define GGML_U16_S16_STORE(LANE, ACCUMULATOR, ACCUMULATOR_LANE)                  \
        output[column + (LANE)] = weight_right_shift == 0                        \
            ? ggml_u16_matmul_requant_q31(                                       \
                vgetq_lane_s64((ACCUMULATOR), (ACCUMULATOR_LANE)),               \
                effective_multiplier, output_zero_point)                         \
            : ggml_u16_htp_matmul_requant_q31(                                   \
                vgetq_lane_s64((ACCUMULATOR), (ACCUMULATOR_LANE)),               \
                effective_multiplier, output_zero_point)
        GGML_U16_S16_STORE(0, accumulators[0], 0);
        GGML_U16_S16_STORE(1, accumulators[0], 1);
        GGML_U16_S16_STORE(2, accumulators[1], 0);
        GGML_U16_S16_STORE(3, accumulators[1], 1);
        GGML_U16_S16_STORE(4, accumulators[2], 0);
        GGML_U16_S16_STORE(5, accumulators[2], 1);
        GGML_U16_S16_STORE(6, accumulators[3], 0);
        GGML_U16_S16_STORE(7, accumulators[3], 1);
#undef GGML_U16_S16_STORE
    }
#endif
    for (; column < output_dimension; ++column) {
        int64_t accumulator = 0;
        for (int row = 0; row < input_dimension; ++row) {
            const int64_t input_value = (int32_t) input[row] - input_zero_point;
            int64_t weight_value =
                (int32_t) weights[(size_t) row * output_dimension + column] -
                weight_zero_point;
            if (weight_right_shift == 7) {
                weight_value = weight_value >= 0
                    ? weight_value >> 7
                    : -(((-weight_value) + 127) >> 7);
            }
            accumulator += input_value * weight_value;
        }
        output[column] = weight_right_shift == 0
            ? ggml_u16_matmul_requant_q31(
                accumulator, effective_multiplier, output_zero_point)
            : ggml_u16_htp_matmul_requant_q31(
                accumulator, effective_multiplier, output_zero_point);
    }
}

static inline uint8_t ggml_u16_to_u8_requant_q31(
        int64_t centered,
        int64_t multiplier,
        int32_t output_zero_point) {
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        centered * multiplier, 31) + output_zero_point;
    return (uint8_t) MAX(0, MIN(UINT8_MAX, quantized));
}

void ggml_vec_convert_u16_u8_qnn_fixed(
        int n,
        uint8_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        int64_t input_to_output_q31,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0 && output != NULL && input != NULL);
    GGML_ASSERT(input_to_output_q31 >= 0 && input_to_output_q31 <= INT32_MAX);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
    const int32x2_t multiplier = vdup_n_s32((int32_t) input_to_output_q31);
    for (; index + 4 <= n; index += 4) {
        const int32x4_t centered = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + index))),
            input_zero);
        const int64x2_t product_lo = vmull_s32(
            vget_low_s32(centered), multiplier);
        const int64x2_t product_hi = vmull_s32(
            vget_high_s32(centered), multiplier);
        output[index + 0] = ggml_u16_to_u8_requant_q31(
            vgetq_lane_s64(product_lo, 0), 1, output_zero_point);
        output[index + 1] = ggml_u16_to_u8_requant_q31(
            vgetq_lane_s64(product_lo, 1), 1, output_zero_point);
        output[index + 2] = ggml_u16_to_u8_requant_q31(
            vgetq_lane_s64(product_hi, 0), 1, output_zero_point);
        output[index + 3] = ggml_u16_to_u8_requant_q31(
            vgetq_lane_s64(product_hi, 1), 1, output_zero_point);
    }
#endif
    for (; index < n; ++index) {
        const int64_t centered = (int32_t) input[index] - input_zero_point;
        output[index] = ggml_u16_to_u8_requant_q31(
            centered, input_to_output_q31, output_zero_point);
    }
}

void ggml_vec_matmul_u16_u8_qnn_fixed(
        int input_dimension,
        int output_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        input_dimension, output_dimension, (size_t) output_dimension,
        output, input, weights, input_zero_point, weight_zero_point,
        product_to_output_q20, output_zero_point);
}

#if defined(__aarch64__) && defined(__clang__) && defined(__ARM_NEON)
static inline int32_t ggml_qnn_balanced_radix64_digit(int32_t value) {
    int32_t digit = value % 64;
    if (digit > 31) {
        digit -= 64;
    } else if (digit < -32) {
        digit += 64;
    }
    return digit;
}

__attribute__((target("dotprod")))
static int ggml_vec_matmul_u16_u8_qnn_dotprod_strided(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    // Four strided rows are transposed entirely in registers into four
    // 4-byte dot-product groups. Centered U16 activations are represented
    // exactly as three signed radix-64 digits, so SDOT never needs a U16/U8
    // materialization buffer and the final S64 accumulator is bit-identical.
    static const uint8_t transpose_indices_data[16] = {
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15,
    };
    const uint8x16_t transpose_indices =
        vld1q_u8(transpose_indices_data);
    const uint8x16_t sign_flip = vdupq_n_u8(0x80);
    const int8x16_t ones = vdupq_n_s8(1);
    int8_t digit_storage[3 * input_dimension];
    int8_t * const digits0 = digit_storage;
    int8_t * const digits1 = digits0 + input_dimension;
    int8_t * const digits2 = digits1 + input_dimension;
    const int32x4_t activation_zero = vdupq_n_s32(input_zero_point);
    const int32x4_t radix_mask = vdupq_n_s32(63);
    const int32x4_t radix_half = vdupq_n_s32(32);
    int digit_row = 0;
    for (; digit_row + 4 <= input_dimension; digit_row += 4) {
        const int32x4_t centered = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + digit_row))),
            activation_zero);
        const int32x4_t d0 = vsubq_s32(
            vandq_s32(vaddq_s32(centered, radix_half), radix_mask),
            radix_half);
        const int32x4_t quotient1 = vshrq_n_s32(vsubq_s32(centered, d0), 6);
        const int32x4_t d1 = vsubq_s32(
            vandq_s32(vaddq_s32(quotient1, radix_half), radix_mask),
            radix_half);
        const int32x4_t d2 = vshrq_n_s32(vsubq_s32(quotient1, d1), 6);
#define GGML_QNN_STORE_FOUR_DIGITS(DST, VALUE) do { \
            const int16x4_t narrowed16 = vmovn_s32((VALUE)); \
            const int8x8_t narrowed8 = vmovn_s16( \
                vcombine_s16(narrowed16, vdup_n_s16(0))); \
            uint32_t packed_digits = vget_lane_u32( \
                vreinterpret_u32_s8(narrowed8), 0); \
            memcpy((DST) + digit_row, &packed_digits, 4); \
        } while (0)
        GGML_QNN_STORE_FOUR_DIGITS(digits0, d0);
        GGML_QNN_STORE_FOUR_DIGITS(digits1, d1);
        GGML_QNN_STORE_FOUR_DIGITS(digits2, d2);
#undef GGML_QNN_STORE_FOUR_DIGITS
    }
    for (; digit_row < input_dimension; ++digit_row) {
        const int32_t centered = (int32_t) input[digit_row] - input_zero_point;
        digits0[digit_row] = (int8_t) ggml_qnn_balanced_radix64_digit(centered);
        const int32_t quotient1 =
            (centered - digits0[digit_row]) / 64;
        digits1[digit_row] =
            (int8_t) ggml_qnn_balanced_radix64_digit(quotient1);
        digits2[digit_row] =
            (int8_t) ((quotient1 - digits1[digit_row]) / 64);
    }

    int column = 0;
    for (; column + 4 <= output_dimension; column += 4) {
        int32x4_t digit0_dot = vdupq_n_s32(0);
        int32x4_t digit1_dot = vdupq_n_s32(0);
        int32x4_t digit2_dot = vdupq_n_s32(0);
        int32x4_t weight_sum = vdupq_n_s32(0);
        int row = 0;
        for (; row + 4 <= input_dimension; row += 4) {
            uint32_t words[4];
            memcpy(&words[0], weights + (size_t) (row + 0) * weight_row_stride + column, 4);
            memcpy(&words[1], weights + (size_t) (row + 1) * weight_row_stride + column, 4);
            memcpy(&words[2], weights + (size_t) (row + 2) * weight_row_stride + column, 4);
            memcpy(&words[3], weights + (size_t) (row + 3) * weight_row_stride + column, 4);
            uint32x4_t rows = vdupq_n_u32(0);
            rows = vsetq_lane_u32(words[0], rows, 0);
            rows = vsetq_lane_u32(words[1], rows, 1);
            rows = vsetq_lane_u32(words[2], rows, 2);
            rows = vsetq_lane_u32(words[3], rows, 3);
            const int8x16_t packed_weights = vreinterpretq_s8_u8(
                veorq_u8(
                    vqtbl1q_u8(vreinterpretq_u8_u32(rows), transpose_indices),
                    sign_flip));

            uint32_t packed_d0;
            uint32_t packed_d1;
            uint32_t packed_d2;
            memcpy(&packed_d0, digits0 + row, 4);
            memcpy(&packed_d1, digits1 + row, 4);
            memcpy(&packed_d2, digits2 + row, 4);
            const int8x16_t digit0 =
                vreinterpretq_s8_u32(vdupq_n_u32(packed_d0));
            const int8x16_t digit1 =
                vreinterpretq_s8_u32(vdupq_n_u32(packed_d1));
            const int8x16_t digit2 =
                vreinterpretq_s8_u32(vdupq_n_u32(packed_d2));
            digit0_dot = vdotq_s32(digit0_dot, packed_weights, digit0);
            digit1_dot = vdotq_s32(digit1_dot, packed_weights, digit1);
            digit2_dot = vdotq_s32(digit2_dot, packed_weights, digit2);
            weight_sum = vdotq_s32(weight_sum, packed_weights, ones);
        }
        int32_t dot0_lanes[4];
        int32_t dot1_lanes[4];
        int32_t dot2_lanes[4];
        int32_t sum_lanes[4];
        vst1q_s32(dot0_lanes, digit0_dot);
        vst1q_s32(dot1_lanes, digit1_dot);
        vst1q_s32(dot2_lanes, digit2_dot);
        vst1q_s32(sum_lanes, weight_sum);
        for (int lane = 0; lane < 4; ++lane) {
            int64_t centered_dot =
                (int64_t) dot0_lanes[lane] +
                ((int64_t) dot1_lanes[lane] << 6) +
                ((int64_t) dot2_lanes[lane] << 12);
            int64_t expanded_weight_sum = sum_lanes[lane];
            for (int tail = row; tail < input_dimension; ++tail) {
                const int32_t weight =
                    (int32_t) weights[(size_t) tail * weight_row_stride +
                        column + lane] - 128;
                centered_dot +=
                    ((int32_t) input[tail] - input_zero_point) * weight;
                expanded_weight_sum += weight;
            }
            const int64_t reduced = ggml_qnn_a16s8_reduce_accumulator(
                centered_dot, expanded_weight_sum, input_zero_point);
            output[column + lane] = ggml_u16_matmul_requant_q31(
                reduced, product_to_output_q20, output_zero_point);
        }
    }
    return column;
}

__attribute__((target("dotprod")))
static void ggml_vec_matmul_u16_u8_qnn_dotprod_strided_pair(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output0,
        uint16_t * GGML_RESTRICT output1,
        const uint16_t * GGML_RESTRICT input0,
        const uint16_t * GGML_RESTRICT input1,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point0,
        int32_t input_zero_point1,
        int64_t product_to_output_q20_0,
        int64_t product_to_output_q20_1,
        int32_t output_zero_point0,
        int32_t output_zero_point1,
        bool token_major) {
    static const uint8_t transpose_indices_data[16] = {
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15,
    };
    const uint8x16_t transpose_indices =
        vld1q_u8(transpose_indices_data);
    const uint8x16_t sign_flip = vdupq_n_u8(0x80);
    const int8x16_t ones = vdupq_n_s8(1);

    int8_t digit_storage[6 * input_dimension];
    int8_t * digits[2][3] = {
        {
            digit_storage,
            digit_storage + input_dimension,
            digit_storage + 2 * input_dimension,
        },
        {
            digit_storage + 3 * input_dimension,
            digit_storage + 4 * input_dimension,
            digit_storage + 5 * input_dimension,
        },
    };
    const uint16_t * inputs[2] = { input0, input1 };
    const int32_t input_zero_points[2] = {
        input_zero_point0, input_zero_point1,
    };
    for (int pair = 0; pair < 2; ++pair) {
        for (int row = 0; row < input_dimension; ++row) {
            const int32_t centered =
                (int32_t) inputs[pair][row] - input_zero_points[pair];
            digits[pair][0][row] =
                (int8_t) ggml_qnn_balanced_radix64_digit(centered);
            const int32_t quotient1 =
                (centered - digits[pair][0][row]) / 64;
            digits[pair][1][row] =
                (int8_t) ggml_qnn_balanced_radix64_digit(quotient1);
            digits[pair][2][row] =
                (int8_t) ((quotient1 - digits[pair][1][row]) / 64);
        }
    }

    for (int column = 0; column < output_dimension; column += 4) {
        int32x4_t dot[2][3] = {
            { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) },
            { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) },
        };
        int32x4_t weight_sum = vdupq_n_s32(0);
        for (int row = 0; row < input_dimension; row += 4) {
            uint32_t words[4];
            if (token_major) {
                memcpy(&words[0], weights + (size_t) (column + 0) * weight_row_stride + row, 4);
                memcpy(&words[1], weights + (size_t) (column + 1) * weight_row_stride + row, 4);
                memcpy(&words[2], weights + (size_t) (column + 2) * weight_row_stride + row, 4);
                memcpy(&words[3], weights + (size_t) (column + 3) * weight_row_stride + row, 4);
            } else {
                memcpy(&words[0], weights + (size_t) (row + 0) * weight_row_stride + column, 4);
                memcpy(&words[1], weights + (size_t) (row + 1) * weight_row_stride + column, 4);
                memcpy(&words[2], weights + (size_t) (row + 2) * weight_row_stride + column, 4);
                memcpy(&words[3], weights + (size_t) (row + 3) * weight_row_stride + column, 4);
            }
            uint32x4_t rows = vdupq_n_u32(0);
            rows = vsetq_lane_u32(words[0], rows, 0);
            rows = vsetq_lane_u32(words[1], rows, 1);
            rows = vsetq_lane_u32(words[2], rows, 2);
            rows = vsetq_lane_u32(words[3], rows, 3);
            uint8x16_t packed_u8 = vreinterpretq_u8_u32(rows);
            if (!token_major) {
                packed_u8 = vqtbl1q_u8(packed_u8, transpose_indices);
            }
            const int8x16_t packed_weights =
                vreinterpretq_s8_u8(veorq_u8(packed_u8, sign_flip));

            for (int pair = 0; pair < 2; ++pair) {
                for (int digit = 0; digit < 3; ++digit) {
                    uint32_t packed_digit;
                    memcpy(&packed_digit, digits[pair][digit] + row, 4);
                    dot[pair][digit] = vdotq_s32(
                        dot[pair][digit],
                        packed_weights,
                        vreinterpretq_s8_u32(vdupq_n_u32(packed_digit)));
                }
            }
            weight_sum = vdotq_s32(weight_sum, packed_weights, ones);
        }

        int32_t sum_lanes[4];
        vst1q_s32(sum_lanes, weight_sum);
        for (int pair = 0; pair < 2; ++pair) {
            int32_t dot_lanes[3][4];
            for (int digit = 0; digit < 3; ++digit) {
                vst1q_s32(dot_lanes[digit], dot[pair][digit]);
            }
            uint16_t * output = pair == 0 ? output0 : output1;
            const int64_t multiplier = pair == 0
                ? product_to_output_q20_0
                : product_to_output_q20_1;
            const int32_t output_zero_point = pair == 0
                ? output_zero_point0
                : output_zero_point1;
            for (int lane = 0; lane < 4; ++lane) {
                const int64_t centered_dot =
                    (int64_t) dot_lanes[0][lane] +
                    ((int64_t) dot_lanes[1][lane] << 6) +
                    ((int64_t) dot_lanes[2][lane] << 12);
                const int64_t reduced = ggml_qnn_a16s8_reduce_accumulator(
                    centered_dot, sum_lanes[lane],
                    input_zero_points[pair]);
                output[column + lane] = ggml_u16_matmul_requant_q31(
                    reduced, multiplier, output_zero_point);
            }
        }
    }
}
#endif

void ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    GGML_ASSERT(input_dimension > 0 && output_dimension > 0);
    GGML_ASSERT(output != NULL && input != NULL && weights != NULL);
    GGML_ASSERT(weight_row_stride >= (size_t) output_dimension);
    GGML_ASSERT(product_to_output_q20 >= 0);

    int column = 0;
#if defined(__aarch64__) && defined(__clang__) && defined(__ARM_NEON)
    if (weight_zero_point == 128 && input_dimension <= 16384 &&
            ggml_qnn_u16_dotprod_enabled()) {
        column = ggml_vec_matmul_u16_u8_qnn_dotprod_strided(
            input_dimension, output_dimension, weight_row_stride,
            output, input, weights, input_zero_point,
            product_to_output_q20, output_zero_point);
    }
#endif
#if defined(__ARM_NEON)
    const uint64_t max_abs_input = (uint64_t) MAX(
        input_zero_point, (int32_t) UINT16_MAX - input_zero_point);
    const uint64_t max_abs_weight = (uint64_t) MAX(
        weight_zero_point, (int32_t) UINT8_MAX - weight_zero_point);
    const bool int32_accumulator_safe =
        max_abs_input * max_abs_weight * (uint64_t) input_dimension <= INT32_MAX &&
        product_to_output_q20 <= INT32_MAX;
    const bool skip_zero_rows = input_zero_point == 0;
    if (int32_accumulator_safe) {
        const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
        for (; column + 16 <= output_dimension; column += 16) {
            int32x4_t accumulators[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            int32x4_t weight_sums[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            for (int row = 0; row < input_dimension; ++row) {
                if (skip_zero_rows && input[row] == 0) {
                    continue;
                }
                const int32_t input_value =
                    (int32_t) input[row] - input_zero_point;
                const uint8x16_t packed = vld1q_u8(
                    weights + (size_t) row * weight_row_stride + column);
                const uint16x8_t low_u16 = vmovl_u8(vget_low_u8(packed));
                const uint16x8_t high_u16 = vmovl_u8(vget_high_u8(packed));
                const int32x4_t weight_values[4] = {
                    vsubq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(low_u16))),
                        weight_zero),
                    vsubq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(low_u16))),
                        weight_zero),
                    vsubq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(high_u16))),
                        weight_zero),
                    vsubq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(high_u16))),
                        weight_zero),
                };
                for (int block = 0; block < 4; ++block) {
                    accumulators[block] = vmlaq_n_s32(
                        accumulators[block], weight_values[block], input_value);
                    weight_sums[block] = vaddq_s32(
                        weight_sums[block], weight_values[block]);
                }
            }
            for (int block = 0; block < 4; ++block) {
                int32_t accumulator_lanes[4];
                int32_t weight_sum_lanes[4];
                vst1q_s32(accumulator_lanes, accumulators[block]);
                vst1q_s32(weight_sum_lanes, weight_sums[block]);
                for (int lane = 0; lane < 4; ++lane) {
                    const int64_t reduced =
                        ggml_qnn_a16s8_reduce_accumulator(
                            accumulator_lanes[lane],
                            weight_sum_lanes[lane],
                            input_zero_point);
                    output[column + block * 4 + lane] =
                        ggml_u16_matmul_requant_q31(
                            reduced,
                            product_to_output_q20,
                            output_zero_point);
                }
            }
        }
    }
    const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
    for (; column + 8 <= output_dimension; column += 8) {
        int64x2_t accumulators[4] = {
            vdupq_n_s64(0), vdupq_n_s64(0),
            vdupq_n_s64(0), vdupq_n_s64(0),
        };
        int32x4_t weight_sums[2] = {
            vdupq_n_s32(0), vdupq_n_s32(0),
        };
        for (int row = 0; row < input_dimension; ++row) {
            if (skip_zero_rows && input[row] == 0) {
                continue;
            }
            const int32_t input_value = (int32_t) input[row] - input_zero_point;
            const int32x2_t input_pair = vdup_n_s32(input_value);
            const uint16x8_t packed = vmovl_u8(vld1_u8(
                weights + (size_t) row * weight_row_stride + column));
            const int32x4_t low = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(packed))),
                weight_zero);
            const int32x4_t high = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(packed))),
                weight_zero);
            weight_sums[0] = vaddq_s32(weight_sums[0], low);
            weight_sums[1] = vaddq_s32(weight_sums[1], high);
            accumulators[0] = vmlal_s32(
                accumulators[0], vget_low_s32(low), input_pair);
            accumulators[1] = vmlal_s32(
                accumulators[1], vget_high_s32(low), input_pair);
            accumulators[2] = vmlal_s32(
                accumulators[2], vget_low_s32(high), input_pair);
            accumulators[3] = vmlal_s32(
                accumulators[3], vget_high_s32(high), input_pair);
        }
        int32_t weight_sum_lanes[8];
        int64_t accumulator_lanes[8];
        vst1q_s32(weight_sum_lanes, weight_sums[0]);
        vst1q_s32(weight_sum_lanes + 4, weight_sums[1]);
        for (int block = 0; block < 4; ++block) {
            vst1q_s64(accumulator_lanes + block * 2, accumulators[block]);
        }
        for (int lane = 0; lane < 8; ++lane) {
            const int64_t accumulator = ggml_qnn_a16s8_reduce_accumulator(
                accumulator_lanes[lane],
                weight_sum_lanes[lane],
                input_zero_point);
            output[column + lane] = ggml_u16_matmul_requant_q31(
                accumulator, product_to_output_q20, output_zero_point);
        }
    }
#endif
    for (; column < output_dimension; ++column) {
        int64_t accumulator = 0;
        int64_t weight_sum = 0;
        for (int row = 0; row < input_dimension; ++row) {
            if (input_zero_point == 0 && input[row] == 0) {
                continue;
            }
            const int32_t weight =
                (int32_t) weights[(size_t) row * weight_row_stride + column] -
                weight_zero_point;
            accumulator +=
                ((int32_t) input[row] - input_zero_point) * weight;
            weight_sum += weight;
        }
        accumulator = ggml_qnn_a16s8_reduce_accumulator(
            accumulator, weight_sum, input_zero_point);
        output[column] = ggml_u16_matmul_requant_q31(
            accumulator, product_to_output_q20, output_zero_point);
    }
}

void ggml_vec_matmul_u16_u8_qnn_fixed_strided_pair(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output0,
        uint16_t * GGML_RESTRICT output1,
        const uint16_t * GGML_RESTRICT input0,
        const uint16_t * GGML_RESTRICT input1,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point0,
        int32_t input_zero_point1,
        int32_t weight_zero_point,
        int64_t product_to_output_q20_0,
        int64_t product_to_output_q20_1,
        int32_t output_zero_point0,
        int32_t output_zero_point1) {
#if defined(__aarch64__) && defined(__clang__) && defined(__ARM_NEON)
    if (weight_zero_point == 128 &&
            input_dimension % 4 == 0 &&
            output_dimension % 4 == 0 &&
            ggml_qnn_u16_dotprod_enabled()) {
        ggml_vec_matmul_u16_u8_qnn_dotprod_strided_pair(
            input_dimension, output_dimension, weight_row_stride,
            output0, output1, input0, input1, weights,
            input_zero_point0, input_zero_point1,
            product_to_output_q20_0, product_to_output_q20_1,
            output_zero_point0, output_zero_point1, false);
        return;
    }
#endif
    ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        input_dimension, output_dimension, weight_row_stride,
        output0, input0, weights, input_zero_point0, weight_zero_point,
        product_to_output_q20_0, output_zero_point0);
    ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        input_dimension, output_dimension, weight_row_stride,
        output1, input1, weights, input_zero_point1, weight_zero_point,
        product_to_output_q20_1, output_zero_point1);
}

void ggml_vec_matmul_u16_u8_qnn_fixed_token_major(
        int input_dimension,
        int output_dimension,
        size_t token_stride,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point) {
    for (int token = 0; token < output_dimension; ++token) {
        const uint8_t * token_weights =
            weights + (size_t) token * token_stride;
        int64_t centered_dot = 0;
        int64_t expanded_weight_sum = 0;
        for (int dim = 0; dim < input_dimension; ++dim) {
            const int32_t weight =
                (int32_t) token_weights[dim] - weight_zero_point;
            centered_dot +=
                ((int32_t) input[dim] - input_zero_point) * weight;
            expanded_weight_sum += weight;
        }
        const int64_t reduced = ggml_qnn_a16s8_reduce_accumulator(
            centered_dot, expanded_weight_sum, input_zero_point);
        output[token] = ggml_u16_matmul_requant_q31(
            reduced, product_to_output_q20, output_zero_point);
    }
}

void ggml_vec_matmul_u16_u8_qnn_fixed_token_major_pair(
        int input_dimension,
        int output_dimension,
        size_t token_stride,
        uint16_t * GGML_RESTRICT output0,
        uint16_t * GGML_RESTRICT output1,
        const uint16_t * GGML_RESTRICT input0,
        const uint16_t * GGML_RESTRICT input1,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point0,
        int32_t input_zero_point1,
        int32_t weight_zero_point,
        int64_t product_to_output_q20_0,
        int64_t product_to_output_q20_1,
        int32_t output_zero_point0,
        int32_t output_zero_point1) {
#if defined(__aarch64__) && defined(__clang__) && defined(__ARM_NEON)
    if (weight_zero_point == 128 &&
            input_dimension % 4 == 0 &&
            output_dimension % 4 == 0 &&
            ggml_qnn_u16_dotprod_enabled()) {
        ggml_vec_matmul_u16_u8_qnn_dotprod_strided_pair(
            input_dimension, output_dimension, token_stride,
            output0, output1, input0, input1, weights,
            input_zero_point0, input_zero_point1,
            product_to_output_q20_0, product_to_output_q20_1,
            output_zero_point0, output_zero_point1, true);
        return;
    }
#endif
    ggml_vec_matmul_u16_u8_qnn_fixed_token_major(
        input_dimension, output_dimension, token_stride,
        output0, input0, weights, input_zero_point0, weight_zero_point,
        product_to_output_q20_0, output_zero_point0);
    ggml_vec_matmul_u16_u8_qnn_fixed_token_major(
        input_dimension, output_dimension, token_stride,
        output1, input1, weights, input_zero_point1, weight_zero_point,
        product_to_output_q20_1, output_zero_point1);
}

static inline uint16_t ggml_u16_requant_q20(
        int64_t centered,
        int64_t multiplier,
        int32_t output_zero_point) {
    const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
        centered * multiplier, 20) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_requant_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint16_t * input,
        int32_t input_zero_point,
        int64_t input_to_output_q20,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0 && output != NULL && input != NULL);
    GGML_ASSERT(input_to_output_q20 >= 0 && input_to_output_q20 <= INT32_MAX);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
    const int32x2_t multiplier = vdup_n_s32((int32_t) input_to_output_q20);
    for (; index + 4 <= n; index += 4) {
        const int32x4_t centered = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + index))),
            input_zero);
        const int64x2_t product_lo = vmull_s32(vget_low_s32(centered), multiplier);
        const int64x2_t product_hi = vmull_s32(vget_high_s32(centered), multiplier);
        output[index + 0] = ggml_u16_requant_q20(
            vgetq_lane_s64(product_lo, 0), 1, output_zero_point);
        output[index + 1] = ggml_u16_requant_q20(
            vgetq_lane_s64(product_lo, 1), 1, output_zero_point);
        output[index + 2] = ggml_u16_requant_q20(
            vgetq_lane_s64(product_hi, 0), 1, output_zero_point);
        output[index + 3] = ggml_u16_requant_q20(
            vgetq_lane_s64(product_hi, 1), 1, output_zero_point);
    }
#endif
    for (; index < n; ++index) {
        output[index] = ggml_u16_requant_q20(
            (int32_t) input[index] - input_zero_point,
            input_to_output_q20, output_zero_point);
    }
}

uint16_t ggml_vec_min_u16_qnn(int n, const uint16_t * input) {
    GGML_ASSERT(n > 0 && input != NULL);
    int index = 0;
    uint16_t result = UINT16_MAX;
#if defined(__ARM_NEON)
    uint16x8_t minimum = vdupq_n_u16(UINT16_MAX);
    for (; index + 8 <= n; index += 8) {
        minimum = vminq_u16(minimum, vld1q_u16(input + index));
    }
#if defined(__aarch64__)
    result = vminvq_u16(minimum);
#else
    const uint16x4_t pair = vmin_u16(vget_low_u16(minimum), vget_high_u16(minimum));
    const uint16x4_t pair2 = vpmin_u16(pair, pair);
    const uint16x4_t pair3 = vpmin_u16(pair2, pair2);
    result = vget_lane_u16(pair3, 0);
#endif
#endif
    for (; index < n; ++index) {
        result = MIN(result, input[index]);
    }
    return result;
}

void ggml_vec_select_affine_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint8_t * GGML_RESTRICT condition,
        const uint16_t * when_true,
        int32_t true_zero_point,
        int32_t true_multiplier_q15,
        int32_t true_right_shift,
        const uint16_t * GGML_RESTRICT when_false,
        int false_stride,
        int32_t false_zero_point,
        int32_t false_multiplier_q15,
        int32_t false_right_shift,
        int32_t output_zero_point) {
    GGML_ASSERT(n >= 0 && output != NULL && condition != NULL);
    GGML_ASSERT(when_true != NULL && when_false != NULL);
    GGML_ASSERT(false_stride == 0 || false_stride == 1);
    GGML_ASSERT(true_multiplier_q15 > 0 && true_multiplier_q15 <= INT16_MAX);
    GGML_ASSERT(false_multiplier_q15 > 0 && false_multiplier_q15 <= INT16_MAX);
    GGML_ASSERT(true_right_shift > 0 && true_right_shift < 63);
    GGML_ASSERT(false_right_shift > 0 && false_right_shift < 63);
    int index = 0;
#if defined(__ARM_NEON)
    const int32x4_t true_zero = vdupq_n_s32(true_zero_point);
    const int32x4_t false_zero = vdupq_n_s32(false_zero_point);
    const int32x4_t true_multiplier = vdupq_n_s32(true_multiplier_q15);
    const int32x4_t false_multiplier = vdupq_n_s32(false_multiplier_q15);
    const int32x4_t true_shift = vdupq_n_s32(-true_right_shift);
    const int32x4_t false_shift = vdupq_n_s32(-false_right_shift);
    const int32x4_t output_zero = vdupq_n_s32(output_zero_point);
    const uint32x4_t mask_zero = vdupq_n_u32(0);
    for (; index + 8 <= n; index += 8) {
        const uint8x8_t condition_values = vld1_u8(condition + index);
        const uint16x8_t condition_wide = vmovl_u8(condition_values);
        const uint32x4_t condition_mask_lo = vcgtq_u32(
            vmovl_u16(vget_low_u16(condition_wide)), mask_zero);
        const uint32x4_t condition_mask_hi = vcgtq_u32(
            vmovl_u16(vget_high_u16(condition_wide)), mask_zero);

        const uint16x8_t true_values = vld1q_u16(when_true + index);
        const uint16x8_t false_values = false_stride == 0
            ? vdupq_n_u16(when_false[0])
            : vld1q_u16(when_false + index);
        const int32x4_t true_centered_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(true_values))),
            true_zero);
        const int32x4_t true_centered_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(true_values))),
            true_zero);
        const int32x4_t false_centered_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(false_values))),
            false_zero);
        const int32x4_t false_centered_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(false_values))),
            false_zero);
        const int32x4_t centered_lo = vbslq_s32(
            condition_mask_lo, true_centered_lo, false_centered_lo);
        const int32x4_t centered_hi = vbslq_s32(
            condition_mask_hi, true_centered_hi, false_centered_hi);
        const int32x4_t multiplier_lo = vbslq_s32(
            condition_mask_lo, true_multiplier, false_multiplier);
        const int32x4_t multiplier_hi = vbslq_s32(
            condition_mask_hi, true_multiplier, false_multiplier);
        const int32x4_t shift_lo = vbslq_s32(
            condition_mask_lo, true_shift, false_shift);
        const int32x4_t shift_hi = vbslq_s32(
            condition_mask_hi, true_shift, false_shift);

        const int64x2_t product_0 = vmull_s32(
            vget_low_s32(centered_lo), vget_low_s32(multiplier_lo));
        const int64x2_t product_1 = vmull_s32(
            vget_high_s32(centered_lo), vget_high_s32(multiplier_lo));
        const int64x2_t product_2 = vmull_s32(
            vget_low_s32(centered_hi), vget_low_s32(multiplier_hi));
        const int64x2_t product_3 = vmull_s32(
            vget_high_s32(centered_hi), vget_high_s32(multiplier_hi));

        const int32x4_t codes_lo = vaddq_s32(
            vcombine_s32(
                vmovn_s64(vrshlq_s64(product_0, vmovl_s32(vget_low_s32(shift_lo)))),
                vmovn_s64(vrshlq_s64(product_1, vmovl_s32(vget_high_s32(shift_lo))))),
            output_zero);
        const int32x4_t codes_hi = vaddq_s32(
            vcombine_s32(
                vmovn_s64(vrshlq_s64(product_2, vmovl_s32(vget_low_s32(shift_hi)))),
                vmovn_s64(vrshlq_s64(product_3, vmovl_s32(vget_high_s32(shift_hi))))),
            output_zero);
        vst1q_u16(
            output + index,
            vcombine_u16(vqmovun_s32(codes_lo), vqmovun_s32(codes_hi)));
    }
#endif
    for (; index < n; ++index) {
        const int64_t product = condition[index]
            ? ((int32_t) when_true[index] - true_zero_point) *
                true_multiplier_q15
            : ((int32_t) when_false[index * false_stride] - false_zero_point) *
                false_multiplier_q15;
        const int32_t right_shift = condition[index]
            ? true_right_shift : false_right_shift;
        const int64_t centered = ggml_u16_htp_round_shift(product, right_shift);
        const int64_t quantized = centered + output_zero_point;
        output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
    }
}

static inline uint32_t ggml_u16_softmax_exp_q31(
        uint32_t code_delta,
        int64_t scale_over_ln2_q24,
        const uint32_t * GGML_RESTRICT exp2_lut_q31) {
    const uint64_t exponent_q24 = (uint64_t) code_delta *
        (uint64_t) scale_over_ln2_q24;
    const uint32_t integer_part = (uint32_t) (exponent_q24 >> 24);
    if (integer_part >= 31) {
        return 0;
    }
    const uint32_t fraction = (uint32_t) exponent_q24 & 0x00ffffffU;
    const uint32_t table_index = fraction >> 16;
    const uint32_t remainder = fraction & 0xffffU;
    const uint32_t high = exp2_lut_q31[table_index];
    const uint32_t low = exp2_lut_q31[table_index + 1];
    const uint32_t interpolated = high - (uint32_t) (
        ((uint64_t) (high - low) * remainder + UINT64_C(32768)) >> 16);
    return (interpolated + (integer_part == 0 ? 0U : 1U << (integer_part - 1))) >>
        integer_part;
}

static inline int32_t ggml_u16_softmax_sat_i16(int64_t value) {
    return (int32_t) MAX(INT16_MIN, MIN(INT16_MAX, value));
}

static inline uint16_t ggml_u16_softmax_exp_htp(
        uint32_t code_delta,
        uint16_t exponent_multiplier_q22) {
    static const uint16_t base[4] = {
        UINT16_C(0x10cc), UINT16_C(0x13ff),
        UINT16_C(0x17c7), UINT16_C(0x1c3e),
    };
    static const uint16_t linear[4] = {
        UINT16_C(0x587c), UINT16_C(0x5538),
        UINT16_C(0x4d91), UINT16_C(0x4014),
    };
    static const uint16_t quadratic[4] = {
        UINT16_C(0x8001), UINT16_C(0x806c),
        UINT16_C(0x825b), UINT16_C(0x8773),
    };

    const uint32_t exponent_q16 =
        (uint32_t) (((uint64_t) code_delta * exponent_multiplier_q22) >> 6);
    const uint16_t fraction = (uint16_t) ~(uint16_t) exponent_q16;
    const uint32_t segment = fraction >> 14;
    int32_t polynomial = ggml_u16_softmax_sat_i16(
        ((int64_t) base[segment] * fraction +
            ((int64_t) linear[segment] << 15)) >> 16);
    polynomial = ggml_u16_softmax_sat_i16(
        ((int64_t) polynomial * fraction +
            ((int64_t) quadratic[segment] << 15)) >> 16);
    const uint32_t integer_part = MIN(UINT32_C(15), exponent_q16 >> 16);
    return (uint16_t) ((uint16_t) polynomial >> integer_part);
}

static inline int32_t ggml_u16_softmax_vmpye(
        int32_t value,
        uint16_t multiplier) {
    return (int32_t) (((int64_t) value * multiplier) >> 16);
}

static inline int32_t ggml_u16_softmax_shift_left_variable(
        int32_t value,
        int32_t shift) {
    return shift >= 0 ? (int32_t) ((uint32_t) value << shift) :
        value >> -shift;
}

static inline int32_t ggml_u16_softmax_shift_right_variable(
        int32_t value,
        int32_t shift) {
    return shift >= 0 ? value >> shift :
        (int32_t) ((uint32_t) value << -shift);
}

static inline uint16_t ggml_u16_softmax_reciprocal_htp(
        uint32_t exponential_sum,
        int32_t * final_right_shift) {
    const int32_t scaled_sum = ggml_u16_softmax_vmpye(
        (int32_t) exponential_sum, UINT16_C(0x8000));
    GGML_ASSERT(scaled_sum > 0);
    const int32_t normalization_shift =
        15 - (int32_t) __builtin_clz((uint32_t) scaled_sum);
    const int32_t target = ggml_u16_softmax_shift_left_variable(
        INT32_C(0x10000), normalization_shift);
    int32_t reciprocal = INT32_C(0x8000);
    for (int iteration = 0; iteration < 5; ++iteration) {
        const int32_t error = target -
            ggml_u16_softmax_vmpye(scaled_sum, (uint16_t) reciprocal);
        const int32_t correction = ggml_u16_softmax_shift_right_variable(
            ggml_u16_softmax_vmpye(error, (uint16_t) reciprocal),
            normalization_shift);
        reciprocal += correction;
    }
    *final_right_shift = normalization_shift + 17;
    return (uint16_t) reciprocal;
}

static inline uint64_t ggml_u64_divide_reciprocal_exact(
        uint64_t numerator,
        uint64_t denominator,
        uint64_t reciprocal_q64) {
    uint64_t quotient = (uint64_t) (
        (ggml_uint128_t) numerator * reciprocal_q64 >> 64);
    const uint64_t remainder = numerator - quotient * denominator;
    quotient += remainder >= denominator;
    return quotient;
}

void ggml_vec_softmax_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint16_t * input,
        int64_t scale_over_ln2_q24,
        int64_t output_unit_code,
        int32_t output_zero_point,
        const uint32_t * GGML_RESTRICT exp2_lut_q31) {
    GGML_ASSERT(n > 0 && output != NULL && input != NULL && exp2_lut_q31 != NULL);
    GGML_ASSERT(scale_over_ln2_q24 > 0 && output_unit_code > 0);
    uint16_t maximum = input[0];
    for (int index = 1; index < n; ++index) {
        maximum = MAX(maximum, input[index]);
    }
    if ((output_unit_code == UINT16_MAX ||
         output_unit_code == (int64_t) UINT16_MAX + 1) &&
        output_zero_point == 0) {
        const uint16_t exponent_multiplier_q22 =
            (uint16_t) (scale_over_ln2_q24 >> 2);
        GGML_ASSERT(exponent_multiplier_q22 > 0);
        uint32_t exponential_sum = 0;
        for (int index = 0; index < n; ++index) {
            output[index] = ggml_u16_softmax_exp_htp(
                (uint32_t) maximum - input[index],
                exponent_multiplier_q22);
            exponential_sum += output[index];
        }
        int32_t final_right_shift = 0;
        const uint16_t reciprocal = ggml_u16_softmax_reciprocal_htp(
            exponential_sum, &final_right_shift);
        GGML_ASSERT(final_right_shift >= 0 && final_right_shift < 32);
        for (int index = 0; index < n; ++index) {
            output[index] = (uint16_t) MIN(UINT16_MAX,
                ((uint32_t) output[index] * reciprocal) >> final_right_shift);
        }
        return;
    }
    uint64_t sum = 0;
    if (n <= 4096) {
        uint32_t exponentials[n];
        for (int index = 0; index < n; ++index) {
            exponentials[index] = ggml_u16_softmax_exp_q31(
                (uint32_t) maximum - input[index],
                scale_over_ln2_q24, exp2_lut_q31);
            sum += exponentials[index];
        }
        GGML_ASSERT(sum > 0);
        const uint64_t reciprocal_q64 = (uint64_t) (
            ((ggml_uint128_t) 1 << 64) / sum);
        for (int index = 0; index < n; ++index) {
            const uint64_t numerator = (uint64_t) exponentials[index] *
                (uint64_t) output_unit_code;
            const int64_t centered = (int64_t)
                ggml_u64_divide_reciprocal_exact(
                    numerator, sum, reciprocal_q64);
            output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                centered + output_zero_point));
        }
        return;
    }
    for (int index = 0; index < n; ++index) {
        sum += ggml_u16_softmax_exp_q31(
            (uint32_t) maximum - input[index],
            scale_over_ln2_q24, exp2_lut_q31);
    }
    GGML_ASSERT(sum > 0);
    const uint64_t reciprocal_q64 = (uint64_t) (
        ((ggml_uint128_t) 1 << 64) / sum);
    for (int index = 0; index < n; ++index) {
        const uint64_t exponential = ggml_u16_softmax_exp_q31(
            (uint32_t) maximum - input[index],
            scale_over_ln2_q24, exp2_lut_q31);
        const uint64_t numerator = exponential * (uint64_t) output_unit_code;
        const int64_t centered = (int64_t)
            ggml_u64_divide_reciprocal_exact(
                numerator, sum, reciprocal_q64);
        output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX,
            centered + output_zero_point));
    }
}

static inline int64_t ggml_u16_square_sum_scalar(
        const uint16_t * GGML_RESTRICT input,
        int n,
        int32_t input_zero_point) {
    int64_t square_sum = 0;
    for (int index = 0; index < n; ++index) {
        const int64_t centered = (int32_t) input[index] - input_zero_point;
        square_sum += centered * centered;
    }
    return square_sum;
}

#if defined(__ARM_NEON)
static inline int64_t ggml_u16_square_sum_neon(
        const uint16_t * GGML_RESTRICT input,
        int n,
        int32_t input_zero_point) {
    const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
    int64_t square_sum = 0;
    int index = 0;

    for (; index + 4 <= n; index += 4) {
        const int32x4_t centered = vsubq_s32(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + index))), input_zero);
        const int64x2_t square_lo = vmull_s32(
            vget_low_s32(centered), vget_low_s32(centered));
        const int64x2_t square_hi = vmull_s32(
            vget_high_s32(centered), vget_high_s32(centered));
        square_sum += vgetq_lane_s64(square_lo, 0) + vgetq_lane_s64(square_lo, 1);
        square_sum += vgetq_lane_s64(square_hi, 0) + vgetq_lane_s64(square_hi, 1);
    }

    return square_sum + ggml_u16_square_sum_scalar(
        input + index, n - index, input_zero_point);
}
#endif

void ggml_vec_rms_norm_affine_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        float input_scale,
        int32_t input_zero_point,
        const uint16_t * GGML_RESTRICT weight,
        float weight_scale,
        int32_t weight_zero_point,
        float epsilon,
        float output_scale,
        int32_t output_zero_point) {
    GGML_ASSERT(n > 0);
    GGML_ASSERT(input_scale > 0.0f);
    GGML_ASSERT(weight_scale > 0.0f);
    GGML_ASSERT(epsilon >= 0.0f);
    GGML_ASSERT(output_scale > 0.0f);

#if defined(__ARM_NEON)
    const int64_t square_sum = ggml_u16_square_sum_neon(
        input, n, input_zero_point);
#else
    const int64_t square_sum = ggml_u16_square_sum_scalar(
        input, n, input_zero_point);
#endif
    const double variance_in_codes = (double) square_sum / (double) n;
    const double epsilon_in_codes = (double) epsilon /
        ((double) input_scale * (double) input_scale);
    const double inverse_rms = 1.0 / sqrt(variance_in_codes + epsilon_in_codes);
    const int requant_shift = 20;
    const int64_t multiplier = (int64_t) llround(ldexp(
        ((double) weight_scale * inverse_rms) /
            (double) output_scale, requant_shift));
    GGML_ASSERT(multiplier >= INT32_MIN && multiplier <= INT32_MAX);
    int index = 0;

#if defined(__ARM_NEON)
    {
        const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
        const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
        for (; index + 4 <= n; index += 4) {
            const int32x4_t input_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + index))), input_zero);
            const int32x4_t weight_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(weight + index))), weight_zero);
            const int64x2_t product_lo = vmull_s32(
                vget_low_s32(input_values), vget_low_s32(weight_values));
            const int64x2_t product_hi = vmull_s32(
                vget_high_s32(input_values), vget_high_s32(weight_values));
            const int64_t product0 = vgetq_lane_s64(product_lo, 0);
            const int64_t product1 = vgetq_lane_s64(product_lo, 1);
            const int64_t product2 = vgetq_lane_s64(product_hi, 0);
            const int64_t product3 = vgetq_lane_s64(product_hi, 1);
            output[index + 0] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(product0 * multiplier, requant_shift) + output_zero_point));
            output[index + 1] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(product1 * multiplier, requant_shift) + output_zero_point));
            output[index + 2] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(product2 * multiplier, requant_shift) + output_zero_point));
            output[index + 3] = (uint16_t) MAX(0, MIN(UINT16_MAX,
                ggml_gptq2_32_round_shift_away_from_zero(product3 * multiplier, requant_shift) + output_zero_point));
        }
    }
#endif

    for (; index < n; ++index) {
        const int64_t input_value = (int32_t) input[index] - input_zero_point;
        const int64_t weight_value = (int32_t) weight[index] - weight_zero_point;
        const int64_t quantized = ggml_gptq2_32_round_shift_away_from_zero(
            input_value * weight_value * multiplier, requant_shift) + output_zero_point;
        output[index] = (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
    }
}

static uint64_t ggml_u128_isqrt(__uint128_t value) {
    __uint128_t bit = ((__uint128_t) 1) << 126;
    __uint128_t root = 0;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    GGML_ASSERT(root <= UINT64_MAX);
    return (uint64_t) root;
}

static int64_t ggml_i128_floor_shift(ggml_int128_t value, int shift) {
    GGML_ASSERT(shift > 0 && shift < 127);
    const bool negative = value < 0;
    const __uint128_t magnitude = negative
        ? (__uint128_t) (-(value + 1)) + 1 : (__uint128_t) value;
    const __uint128_t shifted = negative
        ? (magnitude + (((__uint128_t) 1) << shift) - 1) >> shift
        : magnitude >> shift;
    GGML_ASSERT(shifted <= (negative
        ? (__uint128_t) INT64_MAX + 1 : (__uint128_t) INT64_MAX));
    if (negative && shifted == (__uint128_t) INT64_MAX + 1) {
        return INT64_MIN;
    }
    return negative ? -(int64_t) shifted : (int64_t) shifted;
}

static inline uint16_t ggml_u16_rms_requant_fixed(
        int64_t product,
        uint64_t inverse_rms_q47,
        int64_t weight_to_output_q31,
        int32_t output_zero_point) {
    // QNN HTP floors the fused RMS normalization and affine requantization.
    // Retain the guarded Q47 inverse RMS through the fused product. Reducing it
    // to Q31 first loses visible precision for low-amplitude Q/K vectors.
    const int64_t quantized = ggml_i128_floor_shift(
        (ggml_int128_t) product * inverse_rms_q47 * weight_to_output_q31,
        78) + output_zero_point;
    return (uint16_t) MAX(0, MIN(UINT16_MAX, quantized));
}

void ggml_vec_rms_norm_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        const uint16_t * GGML_RESTRICT weight,
        int32_t weight_zero_point,
        uint64_t epsilon_in_codes_q16,
        int64_t weight_to_output_q31,
        int32_t output_zero_point) {
    GGML_ASSERT(n > 0);
    GGML_ASSERT(weight_to_output_q31 >= 0);
#if defined(__ARM_NEON)
    const int64_t square_sum_signed = ggml_u16_square_sum_neon(input, n, input_zero_point);
#else
    const int64_t square_sum_signed = ggml_u16_square_sum_scalar(input, n, input_zero_point);
#endif
    const uint64_t square_sum = square_sum_signed > 0 ? (uint64_t) square_sum_signed : 0;
    const __uint128_t denominator_q16 =
        ((__uint128_t) square_sum << 16) +
        (__uint128_t) epsilon_in_codes_q16 * (uint64_t) n;
    GGML_ASSERT(denominator_q16 > 0);
    // sqrt(n / denominator) in Q31, retaining 16 guard bits through isqrt.
    const __uint128_t inverse_square_q94 =
        (((__uint128_t) (uint64_t) n) << 110) / denominator_q16;
    const uint64_t inverse_rms_q47 = ggml_u128_isqrt(inverse_square_q94);
    int index = 0;
#if defined(__ARM_NEON)
    {
        const int32x4_t input_zero = vdupq_n_s32(input_zero_point);
        const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
        for (; index + 4 <= n; index += 4) {
            const int32x4_t input_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(input + index))), input_zero);
            const int32x4_t weight_values = vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vld1_u16(weight + index))), weight_zero);
            const int64x2_t product_lo = vmull_s32(vget_low_s32(input_values), vget_low_s32(weight_values));
            const int64x2_t product_hi = vmull_s32(vget_high_s32(input_values), vget_high_s32(weight_values));
            output[index + 0] = ggml_u16_rms_requant_fixed(vgetq_lane_s64(product_lo, 0), inverse_rms_q47, weight_to_output_q31, output_zero_point);
            output[index + 1] = ggml_u16_rms_requant_fixed(vgetq_lane_s64(product_lo, 1), inverse_rms_q47, weight_to_output_q31, output_zero_point);
            output[index + 2] = ggml_u16_rms_requant_fixed(vgetq_lane_s64(product_hi, 0), inverse_rms_q47, weight_to_output_q31, output_zero_point);
            output[index + 3] = ggml_u16_rms_requant_fixed(vgetq_lane_s64(product_hi, 1), inverse_rms_q47, weight_to_output_q31, output_zero_point);
        }
    }
#endif
    for (; index < n; ++index) {
        const int64_t input_value =
            (int32_t) input[index] - input_zero_point;
        const int64_t weight_value =
            (int32_t) weight[index] - weight_zero_point;
        const int64_t product = input_value * weight_value;
        output[index] = ggml_u16_rms_requant_fixed(product, inverse_rms_q47, weight_to_output_q31, output_zero_point);
    }
}

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}
