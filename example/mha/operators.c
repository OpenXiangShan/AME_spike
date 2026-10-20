/*
 * Copyright (c) 2026 BOSC & ICT, CAS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>

#include "ztt_llm_case.h"

static float ztt_llm_output[ZTT_LLM_MHA_ELEMENTS] ZTT_LLM_ALIGNED;

#if !defined(__riscv)
static float
ztt_llm_pow2_integer(int exponent)
{
    uint32_t bits;
    if (exponent < -126)
        bits = 0;
    else if (exponent > 127)
        bits = UINT32_C(0x7f800000);
    else
        bits = (uint32_t)(exponent + 127) << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
ztt_llm_host_matmul_tile(const float *a, const float *b, float *output,
                         int transpose_b)
{
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            float value = 0.0f;
            for (unsigned depth = 0; depth < ZTT_LLM_TILE; ++depth) {
                const float b_value = transpose_b ?
                    b[col * ZTT_LLM_TILE + depth] :
                    b[depth * ZTT_LLM_TILE + col];
                value += a[row * ZTT_LLM_TILE + depth] * b_value;
            }
            output[row * ZTT_LLM_TILE + col] = value;
        }
}

static void
ztt_llm_host_softmax_tile(const float *input, float *output)
{
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
        float maximum = input[row * ZTT_LLM_TILE];
        for (unsigned col = 1; col < ZTT_LLM_TILE; ++col) {
            const float value = input[row * ZTT_LLM_TILE + col];
            if (value > maximum)
                maximum = value;
        }
        float exponentials[ZTT_LLM_TILE];
        float sum = 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            const int shifted = (int)(
                input[row * ZTT_LLM_TILE + col] - maximum);
            exponentials[col] = ztt_llm_pow2_integer(shifted);
            sum += exponentials[col];
        }
        const float reciprocal = 1.0f / sum;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            output[row * ZTT_LLM_TILE + col] =
                exponentials[col] * reciprocal;
    }
}
#endif

#if defined(__riscv)

static inline void
ztt_llm_matmul_tile(const float *a, const float *b, float *output,
                    int transpose_b)
{
    float zero[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i)
        zero[i] = 0.0f;
    ztt_llm_init_fp32_acc(zero);
    if (transpose_b)
        ztt_llm_accumulate_fp32_bt(a, b);
    else
        ztt_llm_accumulate_fp32(a, b);
    ztt_llm_store_fp32_acc(output, 0);
}

static inline void
ztt_llm_softmax_tile(const float *input, float *output)
{
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m1, %[fp32]\n\t"
        "msettyp m2, %[fp32]\n\t"
        "msettyp m3, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mreducemax.row m1, m0\n\t"
        "mmax.ew m1, m1, m0\n\t"
        "msub.ew m2, m1, m0\n\t"
        "mexp2.ew m2, m2\n\t"
        "mreduceadd.row m3, m2\n\t"
        "mrec.ew m3, m3\n\t"
        "mmul.ew m4, m2, m3\n\t"
        "mss.rm m4, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input),
           [output] "r"(output)
        : "memory");
}

#endif /* __riscv */

uint64_t
ztt_llm_run_case(struct ztt_llm_case_result *result)
{
    enum { heads = 2 };
    float input[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float wq[heads][ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float wk[heads][ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float wv[heads][ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float q[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float k[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float v[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float scores[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float probabilities[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;

    const float score_scale = 8.0f;
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            const unsigned index = row * ZTT_LLM_TILE + col;
            input[index] = (float)((int)(row * 3 + col * 2) % 5 - 2);
            for (unsigned head = 0; head < heads; ++head) {
                wq[head][index] = score_scale *
                    (float)((int)(row + col + head * 2) % 3 - 1);
                wk[head][index] = score_scale * (float)((int)(
                    row * (head + 2) + col * 2 + head) % 5 - 2);
                wv[head][index] =
                    (float)((int)(row * 2 + col * (head + 1) + 1) % 3 - 1);
            }
        }

#if defined(__riscv)
    ztt_llm_acquire();
    ztt_llm_configure_fp32_gemm();
#endif
    for (unsigned head = 0; head < heads; ++head) {
        float *context = ztt_llm_output + head * ZTT_LLM_TILE_ELEMENTS;
#if defined(__riscv)
        ztt_llm_matmul_tile(input, wq[head], q, 0);
        ztt_llm_matmul_tile(input, wk[head], k, 0);
        ztt_llm_matmul_tile(input, wv[head], v, 0);
        ztt_llm_matmul_tile(q, k, scores, 1);
        ztt_llm_softmax_tile(scores, probabilities);
        ztt_llm_matmul_tile(probabilities, v, context, 0);
#else
        ztt_llm_host_matmul_tile(input, wq[head], q, 0);
        ztt_llm_host_matmul_tile(input, wk[head], k, 0);
        ztt_llm_host_matmul_tile(input, wv[head], v, 0);
        ztt_llm_host_matmul_tile(q, k, scores, 1);
        ztt_llm_host_softmax_tile(scores, probabilities);
        ztt_llm_host_matmul_tile(probabilities, v, context, 0);
#endif
    }
#if defined(__riscv)
    result->status = ztt_llm_release();
#else
    result->status = 0;
#endif

    result->output = ztt_llm_output;
    result->element_count = ZTT_LLM_MHA_ELEMENTS;
    result->element_bytes = sizeof(ztt_llm_output[0]);
    return result->status;
}
