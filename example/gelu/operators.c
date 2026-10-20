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
    float four[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float zero[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float one[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i) {
        input[i] = (float)((int)(i % 5) - 2);
        four[i] = 4.0f;
        zero[i] = 0.0f;
        one[i] = 1.0f;
    }
#if !defined(__riscv)
    (void)four;
    (void)zero;
    (void)one;
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
        "msettyp m5, %[fp32]\n\t"
        "msettyp m6, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mls.rm m1, %[four]\n\t"
        "mls.rm m4, %[zero]\n\t"
        "mls.rm m5, %[one]\n\t"
        "msqrt.ew m2, m1\n\t"
        "mmul.ew m3, m0, m2\n\t"
        "msub.ew m3, m3, m4\n\t"
        "mexp2.ew m3, m3\n\t"
        "madd.ew m3, m3, m5\n\t"
        "mrec.ew m3, m3\n\t"
        "mmul.ew m6, m0, m3\n\t"
        "mss.rm m6, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input), [four] "r"(four),
           [zero] "r"(zero), [one] "r"(one),
           [output] "r"(ztt_llm_output)
        : "memory");
    result->status = ztt_llm_release();
#else
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i) {
        const float exponent = 0.0f - input[i] * 2.0f;
        const float denominator = 1.0f +
            ztt_llm_pow2_integer((int)exponent);
        ztt_llm_output[i] = input[i] * (1.0f / denominator);
    }
    result->status = 0;
#endif

    result->output = ztt_llm_output;
    result->element_count = ZTT_LLM_TILE_ELEMENTS;
    result->element_bytes = sizeof(ztt_llm_output[0]);
    return result->status;
}
