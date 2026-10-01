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

#ifndef ZTT_LLM_CASE_H
#define ZTT_LLM_CASE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ztt_gemm_test.h"

#define ZTT_LLM_GEMM_FP32 1
#define ZTT_LLM_GEMM_INT8 2
#define ZTT_LLM_GEMM_MIXED 3
#define ZTT_LLM_QKV_PROJECTION 4
#define ZTT_LLM_ATTENTION_SCORES 5
#define ZTT_LLM_RMSNORM 6
#define ZTT_LLM_GELU 7
#define ZTT_LLM_SILU_SWISH 8
#define ZTT_LLM_SOFTMAX 9
#define ZTT_LLM_SOFTMAX_SCALE_MASK 10
#define ZTT_LLM_MHA 12
#define ZTT_LLM_FFN 13

/* Matrix dimensions may be supplied by a generated external-input header. */
#ifndef ZTT_LLM_M
#define ZTT_LLM_M 128
#endif
#ifndef ZTT_LLM_K
#define ZTT_LLM_K 128
#endif
#ifndef ZTT_LLM_N
#define ZTT_LLM_N 128
#endif
#ifndef ZTT_LLM_TILE
#define ZTT_LLM_TILE 4
#endif

enum
{
    ZTT_LLM_TILE_ELEMENTS = ZTT_LLM_TILE * ZTT_LLM_TILE,
    ZTT_LLM_MHA_ELEMENTS = 2 * ZTT_LLM_TILE_ELEMENTS,
    ZTT_LLM_A_ELEMENTS = ZTT_LLM_M * ZTT_LLM_K,
    ZTT_LLM_B_ELEMENTS = ZTT_LLM_K * ZTT_LLM_N,
    ZTT_LLM_MATRIX_ELEMENTS = ZTT_LLM_M * ZTT_LLM_N,
    ZTT_LLM_INT8_PACK = 4,
    ZTT_LLM_INT8_K_STEP = ZTT_LLM_TILE * ZTT_LLM_INT8_PACK,
};

#if defined(__GNUC__)
#define ZTT_LLM_ALIGNED __attribute__((aligned(64)))
#else
#define ZTT_LLM_ALIGNED
#endif

struct ztt_llm_case_result
{
    const void *output;
    unsigned element_count;
    unsigned element_bytes;
    uint64_t status;
};

/* Each case supplies this function from its local operators.c file. */
uint64_t ztt_llm_run_case(struct ztt_llm_case_result *result);

static uint64_t
ztt_llm_hash(const void *data, uint64_t size)
{
    const uint8_t *bytes = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static inline int
ztt_llm_write_report(const char *path, const void *output,
                     unsigned element_count, unsigned element_bytes,
                     uint64_t status, uint64_t suite,
                     const char *case_name)
{
    struct ztt_gemm_report report;
    ztt_gemm_report_init(&report, suite, 1);
    report.values[0][0] = status;
    report.values[0][1] = element_count;
    report.values[0][2] = ztt_llm_hash(
        output, (uint64_t)element_count * element_bytes);

    const unsigned samples = element_count < 32 ? element_count : 32;
    const uint8_t *bytes = output;
    for (unsigned i = 0; i < samples; ++i) {
        const unsigned index = samples == element_count ? i :
            (unsigned)(((uint64_t)i * element_count) / samples);
        uint64_t value = 0;
        memcpy(&value, bytes + (uint64_t)index * element_bytes,
               element_bytes);
        report.values[0][3 + i] = value;
    }
    report.case_values[0] = 3 + samples;

    const char *names[ZTT_GEMM_MAX_CASES] = {case_name};
    return ztt_gemm_report_write(path, &report, names);
}

#if defined(__riscv)

static inline void
ztt_llm_acquire(void)
{
    __asm__ volatile(
        "li t0, 0x3c\n\t"
        "ame.acquire t1, t0\n\t"
        "csrw amestatus, zero"
        ::: "t0", "t1", "memory");
}

static inline uint64_t
ztt_llm_release(void)
{
    uint64_t status;
    __asm__ volatile(
        "csrr %[status], amestatus\n\t"
        "ame.release"
        : [status] "=r"(status)
        :: "memory");
    return status;
}

static inline void
ztt_llm_configure_fp32_gemm(void)
{
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "msettyp m8, %[fp32]\n\t"
        "asettyp acc0, %[fp32]"
        :: [fp32] "r"(fp32)
        : "memory");
}

static inline void
ztt_llm_init_fp32_acc(const float *tile)
{
    __asm__ volatile(
        "mls.rm m8, %[tile]\n\t"
        "mmov.a.m acc0, m8"
        :: [tile] "r"(tile)
        : "memory");
}

static inline void
ztt_llm_accumulate_fp32(const float *a, const float *b)
{
    __asm__ volatile(
        "mls.rm m0, %[a]\n\t"
        "mls.rm m4, %[b]\n\t"
        "mmulacc.2d acc0, m0, m4"
        :: [a] "r"(a), [b] "r"(b)
        : "memory");
}

static inline void
ztt_llm_accumulate_fp32_bt(const float *a, const float *b)
{
    __asm__ volatile(
        "mls.rm m0, %[a]\n\t"
        "mls.rm m4, %[b]\n\t"
        "mmulbtacc.2d acc0, m0, m4"
        :: [a] "r"(a), [b] "r"(b)
        : "memory");
}

static inline void
ztt_llm_store_fp32_acc(float *tile, int copy_matrix)
{
    if (copy_matrix) {
        const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
        __asm__ volatile(
            "msettyp m12, %[fp32]\n\t"
            "mmov.m.a m8, acc0\n\t"
            "mmov.m.m m12, m8\n\t"
            "mss.rm m12, %[tile]"
            :: [tile] "r"(tile), [fp32] "r"(fp32)
            : "memory");
    } else {
        __asm__ volatile(
            "mmov.m.a m8, acc0\n\t"
            "mss.rm m8, %[tile]"
            :: [tile] "r"(tile)
            : "memory");
    }
}

static inline void
ztt_llm_configure_int8_gemm(int convert_result)
{
    const uint64_t int8 = ZTT_DTYPE_INT8;
    const uint64_t int32 = ZTT_DTYPE_INT32;
    const uint64_t int64 = ZTT_DTYPE_INT64;
    __asm__ volatile(
        "msettyp m0, %[int8]\n\t"
        "msettyp m4, %[int8]\n\t"
        "msettyp m8, %[int32]\n\t"
        "asettyp acc0, %[int32]"
        :: [int8] "r"(int8), [int32] "r"(int32)
        : "memory");
    if (convert_result) {
        __asm__ volatile(
            "msettyp m12, %[int64]\n\t"
            "msettyp m16, %[int32]"
            :: [int64] "r"(int64), [int32] "r"(int32)
            : "memory");
    }
}

static inline void
ztt_llm_init_int32_acc(const int32_t *tile)
{
    __asm__ volatile(
        "mls.rm m8, %[tile]\n\t"
        "mmov.a.m acc0, m8"
        :: [tile] "r"(tile)
        : "memory");
}

static inline void
ztt_llm_accumulate_int8(const int8_t *a, const int8_t *b)
{
    __asm__ volatile(
        "mls.rm m0, %[a]\n\t"
        "mls.rm m4, %[b]\n\t"
        "mmulacc.2d acc0, m0, m4"
        :: [a] "r"(a), [b] "r"(b)
        : "memory");
}

static inline void
ztt_llm_store_int32_acc(int32_t *tile, int convert_result)
{
    if (convert_result) {
        __asm__ volatile(
            "mmov.m.a m8, acc0\n\t"
            "mconv.ew m12, m8\n\t"
            "mconv.ew m16, m12\n\t"
            "mss.rm m16, %[tile]"
            :: [tile] "r"(tile)
            : "memory");
    } else {
        __asm__ volatile(
            "mmov.m.a m8, acc0\n\t"
            "mss.rm m8, %[tile]"
            :: [tile] "r"(tile)
            : "memory");
    }
}

#endif /* __riscv */

#endif /* ZTT_LLM_CASE_H */
