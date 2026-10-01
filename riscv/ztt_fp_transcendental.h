// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_FP_TRANSCENDENTAL_H
#define _RISCV_ZTT_FP_TRANSCENDENTAL_H

#include "ztt_fp.h"

enum class ztt_opcode_t : unsigned;

enum class ztt_fp_trans_op {
  sin,
  cos,
  tanh,
  log2,
  exp2,
};

bool ztt_fp_transcendental_operation(ztt_opcode_t opcode, ztt_fp_trans_op& op);

// Returns whether this backend currently implements the complete tuple.
// Unsupported tuples must be reported as ZTT status.UN by the caller.
bool ztt_fp_transcendental_supported(ztt_fp_trans_op op,
                                     uint32_t src_dtype,
                                     uint32_t dst_dtype);

ztt_fp_result_t ztt_fp_transcendental(ztt_fp_trans_op op,
                                      uint32_t src_dtype,
                                      uint32_t dst_dtype,
                                      uint64_t src_bits);

#endif
