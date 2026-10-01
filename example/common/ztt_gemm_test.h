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

#ifndef ZTT_GEMM_TEST_H
#define ZTT_GEMM_TEST_H

#include <stdint.h>

#include "ztt_report.h"

/* ABI consumed by LLVM's canonical RISC-V Ztt GEMM recognizer. */
#define ZTT_DECLARE_GEMM_PARAMS(NAME, ELEM) \
    struct NAME { \
        ELEM *c; const ELEM *a; const ELEM *b; \
        uint64_t m, k, p; uint64_t ldc, lda, ldb; uint64_t datatype; \
    }

ZTT_DECLARE_GEMM_PARAMS(ztt_matrix_f16_params, _Float16);
ZTT_DECLARE_GEMM_PARAMS(ztt_matrix_f32_params, float);

#undef ZTT_DECLARE_GEMM_PARAMS

#define ZTT_DTYPE_FP16_RNE UINT64_C(0x14300110)
#define ZTT_DTYPE_BF16_RNE UINT64_C(0x20300110)
#define ZTT_DTYPE_FP32_RNE UINT64_C(0x20300120)
#define ZTT_DTYPE_FP64_RNE UINT64_C(0x2c300140)
#define ZTT_DTYPE_INT8 UINT64_C(0x40000008)
#define ZTT_DTYPE_INT4 UINT64_C(0x40000004)
#define ZTT_DTYPE_INT32 UINT64_C(0x40000020)
#define ZTT_DTYPE_INT64 UINT64_C(0x40000040)
#define ZTT_DTYPE_INT128 UINT64_C(0x40000080)
#define ZTT_DTYPE_UINT4 UINT64_C(0x00000004)
#define ZTT_DTYPE_UINT8 UINT64_C(0x00000008)
#define ZTT_DTYPE_UINT32 UINT64_C(0x00000020)
#define ZTT_DTYPE_UINT64 UINT64_C(0x00000040)
#define ZTT_DTYPE_UINT128 UINT64_C(0x00000080)

enum {
    ZTT_GEMM_DIM = 16,
    ZTT_GEMM_ELEMENTS = ZTT_GEMM_DIM * ZTT_GEMM_DIM,
};

#endif
