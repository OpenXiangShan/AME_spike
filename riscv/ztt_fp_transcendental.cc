// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_fp_transcendental.h"
#include "ztt_execute.h"
#include "softfloat.h"
#include "ztt_state.h"
#include <cfenv>
#include <cstdint>

// CORE-MATH commit 3ae3930a594176981f8e215cb00e7401c0a3aa50.
// The source is kept at the fixed checkout supplied by the project setup.
extern "C" {
uint16_t ztt_core_math_cos_f16(uint16_t bits);
uint16_t ztt_core_math_cos_bf16(uint16_t bits);
uint32_t ztt_core_math_cos_f32(uint32_t bits);
uint64_t ztt_core_math_cos_f64(uint64_t bits);
uint16_t ztt_core_math_sin_f16(uint16_t bits);
uint16_t ztt_core_math_sin_bf16(uint16_t bits);
uint32_t ztt_core_math_sin_f32(uint32_t bits);
uint64_t ztt_core_math_sin_f64(uint64_t bits);
uint16_t ztt_core_math_tanh_f16(uint16_t bits);
uint16_t ztt_core_math_tanh_bf16(uint16_t bits);
uint32_t ztt_core_math_tanh_f32(uint32_t bits);
uint64_t ztt_core_math_tanh_f64(uint64_t bits);
uint16_t ztt_core_math_log2_f16(uint16_t bits);
uint16_t ztt_core_math_log2_bf16(uint16_t bits);
uint32_t ztt_core_math_log2_f32(uint32_t bits);
uint64_t ztt_core_math_log2_f64(uint64_t bits);
uint16_t ztt_core_math_exp2_f16(uint16_t bits);
uint16_t ztt_core_math_exp2_bf16(uint16_t bits);
uint32_t ztt_core_math_exp2_f32(uint32_t bits);
uint64_t ztt_core_math_exp2_f64(uint64_t bits);
}

namespace {

struct fp_format_t {
  uint64_t value_mask;
  uint64_t magnitude_mask;
  uint64_t exponent_mask;
  uint64_t fraction_mask;
  uint64_t quiet_bit;
  uint64_t canonical_nan;
  uint64_t one;
  unsigned fraction_bits;
  int exponent_bias;
  uint64_t exp2_overflow_input;
  uint64_t exp2_min_exact_magnitude;
};

class host_round_guard final {
 public:
  explicit host_round_guard(unsigned ztt_mode)
  {
    saved_ = std::feholdexcept(&old_env_) == 0;
    int host_mode = FE_TONEAREST;
    switch (ztt_mode) {
      case 0: host_mode = FE_TONEAREST; break;  // RNE
      case 1: host_mode = FE_TOWARDZERO; break; // RTZ
      case 2: host_mode = FE_DOWNWARD; break;   // RDN
      case 3: host_mode = FE_UPWARD; break;     // RUP
      default: break;
    }
    std::fesetround(host_mode);
  }

  ~host_round_guard()
  {
    if (saved_)
      std::fesetenv(&old_env_);
  }

 private:
  std::fenv_t old_env_;
  bool saved_ = false;
};

fp_format_t fp_format(ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::f16:
      return {0xffff, 0x7fff, 0x7c00, 0x03ff, 0x0200, 0x7e00, 0x3c00,
              10, 15, 0x4c00, 0x4e00};
    case ztt_float_kind_t::bf16:
      return {0xffff, 0x7fff, 0x7f80, 0x007f, 0x0040, 0x7fc0, 0x3f80,
              7, 127, 0x4300, 0x4305};
    case ztt_float_kind_t::f32:
      return {0xffffffff, 0x7fffffff, 0x7f800000, 0x007fffff,
              0x00400000, 0x7fc00000, 0x3f800000, 23, 127,
              0x43000000, 0x43150000};
    case ztt_float_kind_t::f64:
      return {UINT64_MAX, UINT64_C(0x7fffffffffffffff),
              UINT64_C(0x7ff0000000000000), UINT64_C(0x000fffffffffffff),
              UINT64_C(0x0008000000000000), UINT64_C(0x7ff8000000000000),
              UINT64_C(0x3ff0000000000000), 52, 1023,
              UINT64_C(0x4090000000000000),
              UINT64_C(0x4090c80000000000)};
    default:
      return {};
  }
}

bool is_core_math_format(uint32_t dtype)
{
  switch (ztt_unit_t::datatype_float_kind(dtype)) {
    case ztt_float_kind_t::f16:
    case ztt_float_kind_t::bf16:
    case ztt_float_kind_t::f32:
    case ztt_float_kind_t::f64:
      return true;
    default:
      return false;
  }
}

unsigned rounding_mode(uint32_t dtype)
{
  return ztt_unit_t::datatype_rounding_mode(dtype);
}

uint64_t core_math_unary(ztt_fp_trans_op op, uint64_t bits,
                         ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::f16:
      switch (op) {
        case ztt_fp_trans_op::sin: return ztt_core_math_sin_f16(uint16_t(bits));
        case ztt_fp_trans_op::cos: return ztt_core_math_cos_f16(uint16_t(bits));
        case ztt_fp_trans_op::tanh: return ztt_core_math_tanh_f16(uint16_t(bits));
        case ztt_fp_trans_op::log2: return ztt_core_math_log2_f16(uint16_t(bits));
        case ztt_fp_trans_op::exp2: return ztt_core_math_exp2_f16(uint16_t(bits));
      }
      break;
    case ztt_float_kind_t::bf16:
      switch (op) {
        case ztt_fp_trans_op::sin: return ztt_core_math_sin_bf16(uint16_t(bits));
        case ztt_fp_trans_op::cos: return ztt_core_math_cos_bf16(uint16_t(bits));
        case ztt_fp_trans_op::tanh: return ztt_core_math_tanh_bf16(uint16_t(bits));
        case ztt_fp_trans_op::log2: return ztt_core_math_log2_bf16(uint16_t(bits));
        case ztt_fp_trans_op::exp2: return ztt_core_math_exp2_bf16(uint16_t(bits));
      }
      break;
    case ztt_float_kind_t::f32:
      switch (op) {
        case ztt_fp_trans_op::sin: return ztt_core_math_sin_f32(uint32_t(bits));
        case ztt_fp_trans_op::cos: return ztt_core_math_cos_f32(uint32_t(bits));
        case ztt_fp_trans_op::tanh: return ztt_core_math_tanh_f32(uint32_t(bits));
        case ztt_fp_trans_op::log2: return ztt_core_math_log2_f32(uint32_t(bits));
        case ztt_fp_trans_op::exp2: return ztt_core_math_exp2_f32(uint32_t(bits));
      }
      break;
    case ztt_float_kind_t::f64:
      switch (op) {
        case ztt_fp_trans_op::sin: return ztt_core_math_sin_f64(bits);
        case ztt_fp_trans_op::cos: return ztt_core_math_cos_f64(bits);
        case ztt_fp_trans_op::tanh: return ztt_core_math_tanh_f64(bits);
        case ztt_fp_trans_op::log2: return ztt_core_math_log2_f64(bits);
        case ztt_fp_trans_op::exp2: return ztt_core_math_exp2_f64(bits);
      }
      break;
    default:
      break;
  }
  return 0;
}

bool is_integral(uint64_t bits, const fp_format_t& format)
{
  const uint64_t exponent =
    (bits & format.exponent_mask) >> format.fraction_bits;
  if (exponent == 0)
    return false;
  const int unbiased = int(exponent) - format.exponent_bias;
  if (unbiased < 0)
    return false;
  if (unsigned(unbiased) >= format.fraction_bits)
    return true;
  const unsigned fractional_bits = format.fraction_bits - unsigned(unbiased);
  const uint64_t mask = (UINT64_C(1) << fractional_bits) - 1;
  return (bits & format.fraction_mask & mask) == 0;
}

bool is_power_of_two(uint64_t bits, const fp_format_t& format)
{
  const uint64_t fraction = bits & format.fraction_mask;
  if ((bits & format.exponent_mask) != 0)
    return fraction == 0;
  return fraction != 0 && (fraction & (fraction - 1)) == 0;
}

} // namespace

bool ztt_fp_transcendental_operation(ztt_opcode_t opcode, ztt_fp_trans_op& op)
{
  switch (opcode) {
    case ztt_opcode_t::msin_ew: op = ztt_fp_trans_op::sin; return true;
    case ztt_opcode_t::mcos_ew: op = ztt_fp_trans_op::cos; return true;
    case ztt_opcode_t::mtanh_ew: op = ztt_fp_trans_op::tanh; return true;
    case ztt_opcode_t::mexp2_ew: op = ztt_fp_trans_op::exp2; return true;
    case ztt_opcode_t::mlog2_ew:
    case ztt_opcode_t::mlog2sub_ew:
    case ztt_opcode_t::mlog2sub_ew_x:
    case ztt_opcode_t::msublog2_ew:
    case ztt_opcode_t::msublog2_ew_x:
      op = ztt_fp_trans_op::log2;
      return true;
    default:
      return false;
  }
}

bool ztt_fp_transcendental_supported(ztt_fp_trans_op op,
                                     uint32_t src_dtype,
                                     uint32_t dst_dtype)
{
  switch (op) {
    case ztt_fp_trans_op::sin:
    case ztt_fp_trans_op::cos:
    case ztt_fp_trans_op::tanh:
    case ztt_fp_trans_op::log2:
    case ztt_fp_trans_op::exp2:
      break;
    default:
      return false;
  }
  return src_dtype == dst_dtype && is_core_math_format(src_dtype) &&
         rounding_mode(dst_dtype) <= 3;
}

ztt_fp_result_t ztt_fp_transcendental(ztt_fp_trans_op op,
                                      uint32_t src_dtype,
                                      uint32_t dst_dtype,
                                      uint64_t src_bits)
{
  ztt_fp_result_t result;
  if (!ztt_fp_transcendental_supported(op, src_dtype, dst_dtype))
    return result;

  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(src_dtype);
  const fp_format_t format = fp_format(kind);
  const uint64_t input_bits = src_bits & format.value_mask;
  const uint64_t sign_bit = format.value_mask ^ format.magnitude_mask;
  const bool negative = (input_bits & sign_bit) != 0;
  const uint64_t magnitude = input_bits & format.magnitude_mask;
  const bool nan = (input_bits & format.exponent_mask) == format.exponent_mask &&
                   (input_bits & format.fraction_mask) != 0;
  if (nan) {
    if ((input_bits & format.quiet_bit) == 0)
      result.flags |= softfloat_flag_invalid;
    result.bits = format.canonical_nan;
    return ztt_fp_finalize(result.bits, dst_dtype, result.flags);
  }
  const bool infinity = magnitude == format.exponent_mask;
  if (infinity) {
    switch (op) {
      case ztt_fp_trans_op::sin:
      case ztt_fp_trans_op::cos:
        result.bits = format.canonical_nan;
        result.flags |= softfloat_flag_invalid;
        break;
      case ztt_fp_trans_op::tanh:
        result.bits = (negative ? sign_bit : 0) | format.one;
        break;
      case ztt_fp_trans_op::log2:
        if (negative) {
          result.bits = format.canonical_nan;
          result.flags |= softfloat_flag_invalid;
        } else {
          result.bits = format.exponent_mask;
        }
        break;
      case ztt_fp_trans_op::exp2:
        result.bits = negative ? 0 : format.exponent_mask;
        break;
    }
    return ztt_fp_finalize(result.bits, dst_dtype, result.flags);
  }
  if (magnitude == 0) {
    switch (op) {
      case ztt_fp_trans_op::sin:
      case ztt_fp_trans_op::tanh:
        result.bits = input_bits;
        break;
      case ztt_fp_trans_op::cos:
      case ztt_fp_trans_op::exp2:
        result.bits = format.one;
        break;
      case ztt_fp_trans_op::log2:
        result.bits = sign_bit | format.exponent_mask;
        result.flags |= softfloat_flag_infinite;
        break;
    }
    return ztt_fp_finalize(result.bits, dst_dtype, result.flags);
  }
  if (op == ztt_fp_trans_op::log2 && negative) {
    result.bits = format.canonical_nan;
    result.flags |= softfloat_flag_invalid;
    return ztt_fp_finalize(result.bits, dst_dtype, result.flags);
  }

  host_round_guard guard(rounding_mode(dst_dtype));
  result.bits = core_math_unary(op, input_bits, kind);

  bool inexact = false;
  switch (op) {
    case ztt_fp_trans_op::sin:
    case ztt_fp_trans_op::cos:
    case ztt_fp_trans_op::tanh:
      inexact = true;
      break;
    case ztt_fp_trans_op::log2:
      inexact = !is_power_of_two(input_bits, format);
      break;
    case ztt_fp_trans_op::exp2: {
      const bool overflow = !negative &&
                            input_bits >= format.exp2_overflow_input;
      if (overflow) {
        return ztt_fp_finalize(result.bits, dst_dtype,
                               softfloat_flag_overflow);
      }
      const bool exact_range = negative
        ? magnitude <= format.exp2_min_exact_magnitude
        : input_bits < format.exp2_overflow_input;
      inexact = !(exact_range && is_integral(input_bits, format));
      break;
    }
  }
  return ztt_fp_finalize(result.bits, dst_dtype, 0, inexact);
}
