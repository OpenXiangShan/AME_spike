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

static float ztt_llm_output[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;

#if !defined(__riscv)
static float
ztt_llm_host_sqrtf(float value)
{
    if (value <= 0.0f)
        return 0.0f;
    float estimate = value > 1.0f ? value : 1.0f;
    for (unsigned i = 0; i < 6; ++i)
        estimate = 0.5f * (estimate + value / estimate);
    return estimate;
}
#endif

uint64_t
ztt_llm_run_case(struct ztt_llm_case_result *result)
{
    float input[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float quarter[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            const float value = row == 2 ? -3.0f : (float)(row + 1);
            input[row * ZTT_LLM_TILE + col] = value;
            quarter[row * ZTT_LLM_TILE + col] = 0.25f;
        }
#if !defined(__riscv)
    (void)quarter;
#endif

#if defined(__riscv)
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    ztt_llm_acquire();
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m1, %[fp32]\n\t"
        "msettyp m2, %[fp32]\n\t"
        "msettyp m3, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mls.rm m3, %[quarter]\n\t"
        "mmul.ew m1, m0, m0\n\t"
        "mreduceadd.row m2, m1\n\t"
        "mmul.ew m2, m2, m3\n\t"
        "msqrt.ew m2, m2\n\t"
        "mrec.ew m2, m2\n\t"
        "mmul.ew m4, m0, m2\n\t"
        "mss.rm m4, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input),
           [quarter] "r"(quarter), [output] "r"(ztt_llm_output)
        : "memory");
    result->status = ztt_llm_release();
#else
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
        float sum = 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            const float value = input[row * ZTT_LLM_TILE + col];
            sum += value * value;
        }
        const float mean = sum * 0.25f;
        const float root = ztt_llm_host_sqrtf(mean);
        const float reciprocal = root > 0.0f ? 1.0f / root : 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            ztt_llm_output[row * ZTT_LLM_TILE + col] =
                input[row * ZTT_LLM_TILE + col] * reciprocal;
    }
    result->status = 0;
#endif

    result->output = ztt_llm_output;
    result->element_count = ZTT_LLM_TILE_ELEMENTS;
    result->element_bytes = sizeof(ztt_llm_output[0]);
    return result->status;
}
