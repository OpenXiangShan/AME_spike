// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_FP_H
#define _RISCV_ZTT_FP_H

#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>

struct ztt_fp_result_t {
  uint64_t bits = 0;
  uint8_t flags = 0;
};

// Exact finite values use (-1)^sign * significand * 2^exponent.
struct ztt_exact_fp_t {
  bool sign = false;
  bool nan = false;
  bool signaling_nan = false;
  bool invalid = false;
  bool infinity = false;
  boost::multiprecision::cpp_int significand = 0;
  int64_t exponent = 0;
};

// RNO deliberately has no SoftFloat mapping until its architectural meaning
// is clarified. Callers must reject RNO tuples before using this function.
uint_fast8_t ztt_to_softfloat_round(uint32_t dtype);

uint64_t ztt_fp_canonical_nan(uint32_t dtype);

ztt_exact_fp_t ztt_decode_exact_fp(uint64_t bits, uint32_t dtype);

ztt_fp_result_t ztt_fp_finalize(uint64_t bits, uint32_t dst_dtype,
                                uint8_t flags = 0, bool inexact = false);

ztt_fp_result_t ztt_round_pack(const ztt_exact_fp_t& exact,
                               uint32_t dst_dtype);

ztt_fp_result_t ztt_fp_scale_exact(uint64_t bits, uint32_t dtype,
                                   int64_t amount);

ztt_fp_result_t ztt_fp_scale_acc_exact(uint64_t accumulator_bits,
                                       uint64_t value_bits,
                                       uint32_t dtype, int64_t amount);

// Convert an arbitrary-width integer value directly to the destination
// floating-point format.  The integer is never rounded through an
// intermediate host or binary64 representation.
ztt_fp_result_t ztt_fp_from_integer(
    const boost::multiprecision::cpp_int& value, uint32_t dst_dtype);

// Exact binary expressions used by the compound elementwise operations.  The
// returned value is rounded only once to dst_dtype.
ztt_fp_result_t ztt_fp_addsub_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                    uint32_t dst_dtype, bool subtract_rhs,
                                    int64_t scale_amount = 0,
                                    bool absolute = false);

// Add/subtract operands decoded in their own floating-point formats and round
// the complete expression once to dst_dtype.
ztt_fp_result_t ztt_fp_addsub_mixed_exact(
    uint64_t lhs_bits, uint32_t lhs_dtype,
    uint64_t rhs_bits, uint32_t rhs_dtype,
    uint32_t dst_dtype, bool subtract_rhs = false);

// Compute fp-integer or integer-fp without first rounding the integer to the
// destination floating-point format.
ztt_fp_result_t ztt_fp_sub_integer_exact(
    uint64_t fp_bits, const boost::multiprecision::cpp_int& integer,
    uint32_t dst_dtype, bool integer_minus_fp = false);
ztt_fp_result_t ztt_fp_mul_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint32_t dst_dtype, bool negate = false);

// Exact same-format operations used for FP8, where evaluating through FP32
// would otherwise make the architectural result depend on an intermediate
// rounding.  The quotient/root helpers retain guard and sticky information
// until the single destination-format rounding.
ztt_fp_result_t ztt_fp_fma_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint64_t addend_bits, uint32_t dst_dtype);
ztt_fp_result_t ztt_fp_div_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint32_t dst_dtype);
ztt_fp_result_t ztt_fp_sqrt_exact(uint64_t bits, uint32_t dst_dtype);

#endif
