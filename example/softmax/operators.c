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
#endif

uint64_t
ztt_llm_run_case(struct ztt_llm_case_result *result)
{
    float input[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            input[row * ZTT_LLM_TILE + col] =
                (float)((int)col - (int)row);

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
        "mreducemax.row m1, m0\n\t"
        "mmax.ew m1, m1, m0\n\t"
        "msub.ew m2, m1, m0\n\t"
        "mexp2.ew m2, m2\n\t"
        "mreduceadd.row m3, m2\n\t"
        "mrec.ew m3, m3\n\t"
        "mmul.ew m4, m2, m3\n\t"
        "mss.rm m4, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input),
           [output] "r"(ztt_llm_output)
        : "memory");
    result->status = ztt_llm_release();
#else
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
            ztt_llm_output[row * ZTT_LLM_TILE + col] =
                exponentials[col] * reciprocal;
    }
    result->status = 0;
#endif

    result->output = ztt_llm_output;
    result->element_count = ZTT_LLM_TILE_ELEMENTS;
    result->element_bytes = sizeof(ztt_llm_output[0]);
    return result->status;
}
