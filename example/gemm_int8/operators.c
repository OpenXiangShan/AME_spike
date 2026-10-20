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

static int8_t ztt_llm_a_i8[ZTT_LLM_MATRIX_ELEMENTS] ZTT_LLM_ALIGNED;
static int8_t ztt_llm_b_i8[ZTT_LLM_MATRIX_ELEMENTS] ZTT_LLM_ALIGNED;
static int32_t ztt_llm_c_i32[ZTT_LLM_MATRIX_ELEMENTS] ZTT_LLM_ALIGNED;

uint64_t
ztt_llm_run_case(struct ztt_llm_case_result *result)
{
    for (unsigned i = 0; i < ZTT_LLM_N; ++i) {
        for (unsigned j = 0; j < ZTT_LLM_N; ++j) {
            ztt_llm_a_i8[i * ZTT_LLM_N + j] =
                (int8_t)((int)((i * 3 + j * 5 + 1) % 5) - 2);
            ztt_llm_b_i8[i * ZTT_LLM_N + j] =
                (int8_t)((int)((i * 7 + j * 2 + 3) % 7) - 3);
            ztt_llm_c_i32[i * ZTT_LLM_N + j] = (int32_t)((i + j) % 3) - 1;
        }
    }

#if defined(__riscv)
    int8_t a_tiles[ZTT_LLM_INT8_PACK * ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    int8_t b_tiles[ZTT_LLM_INT8_PACK * ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    int32_t c_tile[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    ztt_llm_acquire();
    ztt_llm_configure_int8_gemm(1);
    for (unsigned i = 0; i < ZTT_LLM_N; i += ZTT_LLM_TILE) {
        for (unsigned j = 0; j < ZTT_LLM_N; j += ZTT_LLM_TILE) {
            for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
                for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
                    c_tile[row * ZTT_LLM_TILE + col] =
                        ztt_llm_c_i32[(i + row) * ZTT_LLM_N + j + col];
                }
            }
            ztt_llm_init_int32_acc(c_tile);
            for (unsigned k = 0; k < ZTT_LLM_N; k += ZTT_LLM_INT8_K_STEP) {
                for (unsigned square = 0; square < ZTT_LLM_INT8_PACK;
                     ++square) {
                    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
                        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
                            const unsigned depth = k + square * ZTT_LLM_TILE;
                            const unsigned element =
                                square * ZTT_LLM_TILE_ELEMENTS +
                                row * ZTT_LLM_TILE + col;
                            a_tiles[element] =
                                ztt_llm_a_i8[(i + row) * ZTT_LLM_N + depth +
                                             col];
                            b_tiles[element] =
                                ztt_llm_b_i8[(depth + row) * ZTT_LLM_N + j +
                                             col];
                        }
                    }
                }
                ztt_llm_accumulate_int8(a_tiles, b_tiles);
            }
            ztt_llm_store_int32_acc(c_tile, 1);
            for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
                for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
                    ztt_llm_c_i32[(i + row) * ZTT_LLM_N + j + col] =
                        c_tile[row * ZTT_LLM_TILE + col];
                }
            }
        }
    }
    result->status = ztt_llm_release();
#else
    for (unsigned i = 0; i < ZTT_LLM_N; ++i) {
        for (unsigned j = 0; j < ZTT_LLM_N; ++j) {
            int32_t value = ztt_llm_c_i32[i * ZTT_LLM_N + j];
            for (unsigned k = 0; k < ZTT_LLM_N; ++k) {
                value += (int32_t)ztt_llm_a_i8[i * ZTT_LLM_N + k] *
                         ztt_llm_b_i8[k * ZTT_LLM_N + j];
            }
            ztt_llm_c_i32[i * ZTT_LLM_N + j] = value;
        }
    }
    result->status = 0;
#endif

    result->output = ztt_llm_c_i32;
    result->element_count = ZTT_LLM_MATRIX_ELEMENTS;
    result->element_bytes = sizeof(ztt_llm_c_i32[0]);
    return result->status;
}
