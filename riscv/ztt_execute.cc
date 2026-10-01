// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_execute.h"
#include "mmu.h"
#include "processor.h"
#include "softfloat.h"
#include "trap.h"
#include "ztt_csr.h"
#include "ztt_decode.h"
#include "ztt_fp.h"
#include "ztt_fp_transcendental.h"
#include "ztt_state.h"
#include "ztt_validation.h"
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;
using big_int = boost::multiprecision::cpp_int;

thread_local uint8_t* active_fp_flags = nullptr;
thread_local bool* active_sat_flag = nullptr;

void accrue_fp_flags(uint8_t flags)
{
  if (active_fp_flags)
    *active_fp_flags |= flags & ztt::kAmeFlagMask;
}

[[noreturn]] void illegal(insn_t insn)
{
  throw trap_illegal_instruction(insn.bits());
}

class dirty_on_success_t {
 public:
  dirty_on_success_t(processor_t& proc, bool enabled)
    : proc(proc), enabled(enabled), exceptions(std::uncaught_exceptions()) {}
  ~dirty_on_success_t()
  {
    if (enabled && exceptions == std::uncaught_exceptions())
      ztt_mark_state_dirty(proc);
  }

 private:
  processor_t& proc;
  bool enabled;
  int exceptions;
};

class ame_flags_on_success_t {
 public:
  explicit ame_flags_on_success_t(processor_t& proc)
    : proc(proc), exceptions(std::uncaught_exceptions())
  {
    active_fp_flags = &fp_flags;
    active_sat_flag = &sat_flag;
  }

  ~ame_flags_on_success_t()
  {
    active_fp_flags = nullptr;
    active_sat_flag = nullptr;
    if (exceptions != std::uncaught_exceptions())
      return;
    proc.ZTU.set_amefflags(fp_flags);
    if (sat_flag)
      proc.ZTU.set_amexsat();
  }

 private:
  processor_t& proc;
  int exceptions;
  uint8_t fp_flags = 0;
  bool sat_flag = false;
};

bool valid_acquire_descriptor(reg_t descriptor)
{
  if (descriptor >> 6)
    return false;
  const unsigned wait_mode = descriptor & 0x3;
  const unsigned timeout_class = (descriptor >> 2) & 0xf;
  return wait_mode == 0 ||
         (wait_mode == 1 && (timeout_class <= 3 || timeout_class == 0xf));
}

bool snapshot_m(processor_t& proc, const ztt_m_meta_t& meta,
                std::vector<elem_t>& values)
{
  return proc.ZTU.read_m(meta.reg, values);
}

bool snapshot_m(processor_t& proc, const ztt_m_operand_t& operand,
                std::vector<elem_t>& values)
{
  values.clear();
  values.reserve(operand.squares * ztt::kNumElements);
  for (std::size_t group = 0; group < operand.groups; ++group) {
    std::vector<elem_t> square_group;
    const unsigned reg = operand.meta.reg + group * operand.group_registers;
    if (!proc.ZTU.read_m_as(reg, operand.meta.dtype, square_group))
      return false;
    values.insert(values.end(), square_group.begin(), square_group.end());
  }
  return true;
}

bool commit_m(processor_t& proc, const ztt_m_operand_t& operand,
              const std::vector<elem_t>& values)
{
  if (values.size() != operand.squares * ztt::kNumElements)
    return false;
  const std::size_t elements_per_group =
    ztt_unit_t::element_count(operand.meta.dtype);
  std::vector<ztt_unit_t::m_write_t> writes;
  writes.reserve(operand.groups);
  for (std::size_t group = 0; group < operand.groups; ++group) {
    const auto begin = values.begin() + group * elements_per_group;
    const std::vector<elem_t> square_group(begin, begin + elements_per_group);
    const unsigned reg = operand.meta.reg + group * operand.group_registers;
    writes.push_back({reg, operand.meta.dtype, square_group});
  }
  return proc.ZTU.write_m_atomic(writes);
}

bool snapshot_acc(processor_t& proc, const ztt_acc_meta_t& meta,
                  std::vector<elem_t>& values)
{
  return proc.ZTU.read_acc(meta.reg, values);
}

ztt_dtype_tuple_t one_tuple(uint32_t dtype)
{
  ztt_dtype_tuple_t tuple;
  tuple.has_dest = true;
  tuple.dest = dtype;
  return tuple;
}

ztt_dtype_tuple_t unary_tuple(uint32_t dest, uint32_t source)
{
  ztt_dtype_tuple_t tuple = one_tuple(dest);
  tuple.has_source1 = true;
  tuple.source1 = source;
  return tuple;
}

ztt_dtype_tuple_t binary_tuple(uint32_t dest, uint32_t source1,
                               uint32_t source2)
{
  ztt_dtype_tuple_t tuple = unary_tuple(dest, source1);
  tuple.has_source2 = true;
  tuple.source2 = source2;
  return tuple;
}

ztt_dtype_tuple_t scalar_tuple(uint32_t dest, uint32_t source,
                               uint32_t scalar)
{
  ztt_dtype_tuple_t tuple = one_tuple(dest);
  tuple.has_source2 = true;
  tuple.source2 = source;
  tuple.has_scalar = true;
  tuple.scalar = scalar;
  return tuple;
}

std::vector<ztt_m_operand_t> form_m_operands(
  const processor_t& proc, insn_t insn,
  std::initializer_list<ztt_m_meta_t> operands)
{
  const std::size_t squares = ztt_instruction_square_count(operands);
  std::vector<ztt_m_operand_t> formed;
  formed.reserve(operands.size());
  for (const auto& operand : operands)
    formed.push_back(ztt_form_m_operand(proc, insn, operand, squares));
  return formed;
}

bool negative(elem_t value, uint32_t dtype)
{
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  return (value >> (width - 1)) & 1;
}

unsigned ilog2(elem_t value)
{
  unsigned result = 0;
  while (value >>= 1)
    ++result;
  return result;
}

enum class ew_op {
  abs, absdiff, add, bit_and, bit_andnot, cmpge, cmplt, exp2, hdiff,
  hdiff_scalar, ldexp, ldexp_scalar, ldexpacc, ldexpacc_scalar, log2,
  log2sub, log2sub_scalar, max, mean, min, mul, mulacc,
  mulaccneg, muladd, mulneg, mulsub, bit_or, bit_ornot, rdexp,
  rdexpacc, sub, sub_scalar, sublog2, sublog2_scalar, bit_xor,
};

enum class fp_unary_op {
  cos, frintm, frintn, frintp, frintz, rec, rsqrt, sin, sqrt, tanh,
};

big_int unsigned_value(elem_t value)
{
  big_int result = uint64_t(value >> 64);
  result <<= 64;
  result += uint64_t(value);
  return result;
}

big_int integer_value(elem_t value, uint32_t dtype)
{
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  big_int result = unsigned_value(value & ztt_unit_t::element_mask(width));
  if (ztt_unit_t::datatype_signed(dtype) && negative(value, dtype))
    result -= big_int(1) << width;
  return result;
}

big_int integer_min(uint32_t dtype)
{
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  return ztt_unit_t::datatype_signed(dtype) ? -(big_int(1) << (width - 1)) : 0;
}

big_int integer_max(uint32_t dtype)
{
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  return ztt_unit_t::datatype_signed(dtype)
       ? (big_int(1) << (width - 1)) - 1 : (big_int(1) << width) - 1;
}

elem_t low_element(big_int value, unsigned width)
{
  const big_int modulus = big_int(1) << width;
  value %= modulus;
  if (value < 0)
    value += modulus;
  const big_int word_mask = (big_int(1) << 64) - 1;
  const uint64_t low = (value & word_mask).convert_to<uint64_t>();
  const uint64_t high = width > 64
    ? ((value >> 64) & word_mask).convert_to<uint64_t>() : 0;
  return (elem_t(high) << 64) | low;
}

elem_t integer_result(big_int value, uint32_t dtype)
{
  if (ztt_unit_t::datatype_saturating(dtype)) {
    const big_int clamped = std::max(integer_min(dtype),
                                     std::min(integer_max(dtype), value));
    if (clamped != value && active_sat_flag)
      *active_sat_flag = true;
    value = clamped;
  }
  return low_element(value, ztt_unit_t::datatype_bits(dtype));
}

big_int floor_div_pow2(big_int value, unsigned shift)
{
  if (shift == 0)
    return value;
  if (value >= 0)
    return value >> shift;
  return -((-value + (big_int(1) << shift) - 1) >> shift);
}

big_int round_div_pow2(big_int value, unsigned shift, uint32_t dtype)
{
  if (shift == 0)
    return value;
  const big_int quotient = floor_div_pow2(value, shift);
  const big_int remainder = value - (quotient << shift);
  if (remainder == 0)
    return quotient;
  const big_int half = big_int(1) << (shift - 1);
  switch (ztt_unit_t::datatype_rounding_mode(dtype)) {
    case 0: return quotient + (remainder >= half ? 1 : 0); // RNU
    case 1: return quotient + (remainder > half ||
                   (remainder == half && bool(quotient & 1)) ? 1 : 0); // RNE
    case 2: return quotient; // RDN
    case 3: return bool(quotient & 1) ? quotient : quotient + 1; // ROD
  }
  return quotient;
}

int bounded_shift(const big_int& value, unsigned width)
{
  const int limit = int(2 * width + 2);
  if (value > limit)
    return limit;
  if (value < -limit)
    return -limit;
  return value.convert_to<int>();
}

big_int integer_scale(big_int value, const big_int& exponent, unsigned width,
                      uint32_t dtype)
{
  const int shift = bounded_shift(exponent, width);
  return shift >= 0 ? value << shift
                    : round_div_pow2(value, unsigned(-shift), dtype);
}

// Keep the accumulator and the scaled operand in a common binary exponent.
// Only the final conversion to the integer destination is rounded.  In
// particular, do not round A*2^B before adding the old accumulator.
big_int integer_scale_acc_exact(big_int accumulator, big_int value,
                                const big_int& exponent, unsigned width,
                                uint32_t dtype)
{
  const int shift = bounded_shift(exponent, width);
  if (shift >= 0)
    return accumulator + (value << unsigned(shift));

  const unsigned fractional_shift = unsigned(-shift);
  return round_div_pow2((accumulator << fractional_shift) + value,
                        fractional_shift, dtype);
}

class softfloat_scope_t {
 public:
  explicit softfloat_scope_t(uint32_t dtype)
    : old_round(softfloat_roundingMode), old_flags(softfloat_exceptionFlags)
  {
    softfloat_roundingMode = ztt_to_softfloat_round(dtype);
    softfloat_exceptionFlags = 0;
  }

  ~softfloat_scope_t()
  {
    accrue_fp_flags(softfloat_exceptionFlags);
    softfloat_roundingMode = old_round;
    softfloat_exceptionFlags = old_flags;
  }

 private:
  uint_fast8_t old_round;
  uint_fast8_t old_flags;
};

float32_t fp_to_f32(elem_t value, ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::e4m3:
      return bf16_to_f32(e4m3_to_bf16(e4m3_t{uint8_t(value)}));
    case ztt_float_kind_t::e5m2:
      return bf16_to_f32(e5m2_to_bf16(e5m2_t{uint8_t(value)}));
    case ztt_float_kind_t::f16:
      return f16_to_f32(float16_t{uint16_t(value)});
    case ztt_float_kind_t::bf16:
      return bf16_to_f32(bfloat16_t{uint16_t(value)});
    case ztt_float_kind_t::f32:
      return float32_t{uint32_t(value)};
    case ztt_float_kind_t::f64:
      return f64_to_f32(float64_t{uint64_t(value)});
    case ztt_float_kind_t::none:
      break;
  }
  return float32_t{0};
}

float64_t fp_to_f64(elem_t value, ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::e4m3:
      return bf16_to_f64(e4m3_to_bf16(e4m3_t{uint8_t(value)}));
    case ztt_float_kind_t::e5m2:
      return bf16_to_f64(e5m2_to_bf16(e5m2_t{uint8_t(value)}));
    case ztt_float_kind_t::f16:
      return f16_to_f64(float16_t{uint16_t(value)});
    case ztt_float_kind_t::bf16:
      return bf16_to_f64(bfloat16_t{uint16_t(value)});
    case ztt_float_kind_t::f32:
      return f32_to_f64(float32_t{uint32_t(value)});
    case ztt_float_kind_t::f64:
      return float64_t{uint64_t(value)};
    case ztt_float_kind_t::none:
      break;
  }
  return float64_t{0};
}

elem_t fp_from_f32(float32_t value, ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::e4m3: return f32_to_e4m3(value, true).v;
    case ztt_float_kind_t::e5m2: return f32_to_e5m2(value, false).v;
    case ztt_float_kind_t::f16: return f32_to_f16(value).v;
    case ztt_float_kind_t::bf16: return f32_to_bf16(value).v;
    case ztt_float_kind_t::f32: return value.v;
    case ztt_float_kind_t::f64: return f32_to_f64(value).v;
    case ztt_float_kind_t::none: break;
  }
  return 0;
}

elem_t fp_from_f64(float64_t value, ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::e4m3: return f32_to_e4m3(f64_to_f32(value), true).v;
    case ztt_float_kind_t::e5m2: return f32_to_e5m2(f64_to_f32(value), false).v;
    case ztt_float_kind_t::f16: return f64_to_f16(value).v;
    case ztt_float_kind_t::bf16: return f64_to_bf16(value).v;
    case ztt_float_kind_t::f32: return f64_to_f32(value).v;
    case ztt_float_kind_t::f64: return value.v;
    case ztt_float_kind_t::none: break;
  }
  return 0;
}

bool fp_is_nan(elem_t value, ztt_float_kind_t kind)
{
  const uint64_t bits = uint64_t(value);
  switch (kind) {
    case ztt_float_kind_t::e4m3: return (bits & 0x7f) == 0x7f;
    case ztt_float_kind_t::e5m2: return (bits & 0x7c) == 0x7c && (bits & 3);
    case ztt_float_kind_t::f16: return (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff);
    case ztt_float_kind_t::bf16: return (bits & 0x7f80) == 0x7f80 && (bits & 0x7f);
    case ztt_float_kind_t::f32:
      return (bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff);
    case ztt_float_kind_t::f64:
      return (bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
             (bits & UINT64_C(0xfffffffffffff));
    case ztt_float_kind_t::none: return false;
  }
  return false;
}

bool fp_is_signaling_nan(elem_t value, ztt_float_kind_t kind)
{
  if (!fp_is_nan(value, kind))
    return false;
  const uint64_t bits = uint64_t(value);
  switch (kind) {
    case ztt_float_kind_t::e4m3: return false;
    case ztt_float_kind_t::e5m2: return (bits & 0x2) == 0;
    case ztt_float_kind_t::f16: return (bits & 0x200) == 0;
    case ztt_float_kind_t::bf16: return (bits & 0x40) == 0;
    case ztt_float_kind_t::f32: return (bits & 0x400000) == 0;
    case ztt_float_kind_t::f64:
      return (bits & UINT64_C(0x8000000000000)) == 0;
    case ztt_float_kind_t::none: return false;
  }
  return false;
}

elem_t fp_negate(elem_t value, uint32_t dtype)
{
  return value ^ (elem_t(1) << (ztt_unit_t::datatype_bits(dtype) - 1));
}

elem_t fp_binary(elem_t a, elem_t b, uint32_t dtype, char op)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2) {
    const ztt_fp_result_t result = op == '*'
      ? ztt_fp_mul_exact(uint64_t(a), uint64_t(b), dtype)
      : ztt_fp_addsub_exact(uint64_t(a), uint64_t(b), dtype, op == '-');
    accrue_fp_flags(result.flags);
    return result.bits;
  }
  switch (kind) {
    case ztt_float_kind_t::f16: {
      const float16_t fa{uint16_t(a)}, fb{uint16_t(b)};
      return (op == '+' ? f16_add(fa, fb) : op == '-' ? f16_sub(fa, fb)
                                                     : f16_mul(fa, fb)).v;
    }
    case ztt_float_kind_t::bf16: {
      const bfloat16_t fa{uint16_t(a)}, fb{uint16_t(b)};
      return (op == '+' ? bf16_add(fa, fb) : op == '-' ? bf16_sub(fa, fb)
                                                      : bf16_mul(fa, fb)).v;
    }
    case ztt_float_kind_t::f32: {
      const float32_t fa{uint32_t(a)}, fb{uint32_t(b)};
      return (op == '+' ? f32_add(fa, fb) : op == '-' ? f32_sub(fa, fb)
                                                     : f32_mul(fa, fb)).v;
    }
    case ztt_float_kind_t::f64: {
      const float64_t fa{uint64_t(a)}, fb{uint64_t(b)};
      return (op == '+' ? f64_add(fa, fb) : op == '-' ? f64_sub(fa, fb)
                                                     : f64_mul(fa, fb)).v;
    }
    default: return 0;
  }
}

elem_t fp_fma(elem_t a, elem_t b, elem_t c, uint32_t dtype)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2) {
    const ztt_fp_result_t result = ztt_fp_fma_exact(
      uint64_t(a), uint64_t(b), uint64_t(c), dtype);
    accrue_fp_flags(result.flags);
    return result.bits;
  }
  switch (kind) {
    case ztt_float_kind_t::f16:
      return f16_mulAdd(float16_t{uint16_t(a)}, float16_t{uint16_t(b)},
                        float16_t{uint16_t(c)}).v;
    case ztt_float_kind_t::bf16:
      return bf16_mulAdd(bfloat16_t{uint16_t(a)}, bfloat16_t{uint16_t(b)},
                         bfloat16_t{uint16_t(c)}).v;
    case ztt_float_kind_t::f32:
      return f32_mulAdd(float32_t{uint32_t(a)}, float32_t{uint32_t(b)},
                        float32_t{uint32_t(c)}).v;
    case ztt_float_kind_t::f64:
      return f64_mulAdd(float64_t{uint64_t(a)}, float64_t{uint64_t(b)},
                        float64_t{uint64_t(c)}).v;
    default: return 0;
  }
}

elem_t fp_div(elem_t a, elem_t b, uint32_t dtype)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2) {
    const ztt_fp_result_t result = ztt_fp_div_exact(
      uint64_t(a), uint64_t(b), dtype);
    accrue_fp_flags(result.flags);
    return result.bits;
  }
  switch (kind) {
    case ztt_float_kind_t::f16:
      return f16_div(float16_t{uint16_t(a)}, float16_t{uint16_t(b)}).v;
    case ztt_float_kind_t::bf16:
      return bf16_div(bfloat16_t{uint16_t(a)}, bfloat16_t{uint16_t(b)}).v;
    case ztt_float_kind_t::f32:
      return f32_div(float32_t{uint32_t(a)}, float32_t{uint32_t(b)}).v;
    case ztt_float_kind_t::f64:
      return f64_div(float64_t{uint64_t(a)}, float64_t{uint64_t(b)}).v;
    default: return 0;
  }
}

elem_t fp_sqrt(elem_t value, uint32_t dtype)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2) {
    const ztt_fp_result_t result = ztt_fp_sqrt_exact(uint64_t(value), dtype);
    accrue_fp_flags(result.flags);
    return result.bits;
  }
  switch (kind) {
    case ztt_float_kind_t::f16: return f16_sqrt(float16_t{uint16_t(value)}).v;
    case ztt_float_kind_t::bf16: return bf16_sqrt(bfloat16_t{uint16_t(value)}).v;
    case ztt_float_kind_t::f32: return f32_sqrt(float32_t{uint32_t(value)}).v;
    case ztt_float_kind_t::f64: return f64_sqrt(float64_t{uint64_t(value)}).v;
    default: return 0;
  }
}

bool fp_less(elem_t a, elem_t b, uint32_t dtype)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2)
    return f32_lt_quiet(fp_to_f32(a, kind), fp_to_f32(b, kind));
  switch (kind) {
    case ztt_float_kind_t::f16:
      return f16_lt_quiet(float16_t{uint16_t(a)}, float16_t{uint16_t(b)});
    case ztt_float_kind_t::bf16:
      return f32_lt_quiet(float32_t{uint32_t(uint16_t(a)) << 16},
                          float32_t{uint32_t(uint16_t(b)) << 16});
    case ztt_float_kind_t::f32:
      return f32_lt_quiet(float32_t{uint32_t(a)}, float32_t{uint32_t(b)});
    case ztt_float_kind_t::f64:
      return f64_lt_quiet(float64_t{uint64_t(a)}, float64_t{uint64_t(b)});
    default: return false;
  }
}

bool fp_greater_equal(elem_t a, elem_t b, uint32_t dtype)
{
  softfloat_scope_t scope(dtype);
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2)
    return f32_le_quiet(fp_to_f32(b, kind), fp_to_f32(a, kind));
  switch (kind) {
    case ztt_float_kind_t::f16:
      return f16_le_quiet(float16_t{uint16_t(b)}, float16_t{uint16_t(a)});
    case ztt_float_kind_t::bf16:
      return f32_le_quiet(float32_t{uint32_t(uint16_t(b)) << 16},
                          float32_t{uint32_t(uint16_t(a)) << 16});
    case ztt_float_kind_t::f32:
      return f32_le_quiet(float32_t{uint32_t(b)}, float32_t{uint32_t(a)});
    case ztt_float_kind_t::f64:
      return f64_le_quiet(float64_t{uint64_t(b)}, float64_t{uint64_t(a)});
    default: return false;
  }
}

elem_t fp_minmax(elem_t a, elem_t b, uint32_t dtype, bool maximum)
{
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (fp_is_nan(a, kind) || fp_is_nan(b, kind)) {
    if (fp_is_signaling_nan(a, kind) || fp_is_signaling_nan(b, kind))
      accrue_fp_flags(softfloat_flag_invalid);
    return ztt_fp_canonical_nan(dtype);
  }
  softfloat_scope_t scope(dtype);
  if (kind == ztt_float_kind_t::e4m3 || kind == ztt_float_kind_t::e5m2) {
    const float32_t result = maximum
      ? f32_max(fp_to_f32(a, kind), fp_to_f32(b, kind))
      : f32_min(fp_to_f32(a, kind), fp_to_f32(b, kind));
    return fp_from_f32(result, kind);
  }
  switch (kind) {
    case ztt_float_kind_t::f16:
      return (maximum ? f16_max(float16_t{uint16_t(a)}, float16_t{uint16_t(b)})
                      : f16_min(float16_t{uint16_t(a)}, float16_t{uint16_t(b)})).v;
    case ztt_float_kind_t::bf16:
      return (maximum ? bf16_max(bfloat16_t{uint16_t(a)}, bfloat16_t{uint16_t(b)})
                      : bf16_min(bfloat16_t{uint16_t(a)}, bfloat16_t{uint16_t(b)})).v;
    case ztt_float_kind_t::f32:
      return (maximum ? f32_max(float32_t{uint32_t(a)}, float32_t{uint32_t(b)})
                      : f32_min(float32_t{uint32_t(a)}, float32_t{uint32_t(b)})).v;
    case ztt_float_kind_t::f64:
      return (maximum ? f64_max(float64_t{uint64_t(a)}, float64_t{uint64_t(b)})
                      : f64_min(float64_t{uint64_t(a)}, float64_t{uint64_t(b)})).v;
    default: return 0;
  }
}

double fp_to_host(elem_t value, uint32_t dtype)
{
  const float64_t converted = fp_to_f64(value, ztt_unit_t::datatype_float_kind(dtype));
  double result;
  const uint64_t bits = converted.v;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

elem_t host_to_fp(double value, uint32_t dtype)
{
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  softfloat_scope_t scope(dtype);
  return fp_from_f64(float64_t{bits}, ztt_unit_t::datatype_float_kind(dtype));
}

big_int rounded_float_integer(elem_t value, uint32_t source_dtype,
                              uint32_t dest_dtype)
{
  const ztt_exact_fp_t fp = ztt_decode_exact_fp(value, source_dtype);
  if (fp.nan) {
    accrue_fp_flags(softfloat_flag_invalid);
    return 0;
  }
  if (fp.infinity) {
    accrue_fp_flags(softfloat_flag_invalid);
    return fp.sign ? integer_min(dest_dtype) : integer_max(dest_dtype);
  }
  big_int value_int = fp.sign ? -fp.significand : fp.significand;
  if (fp.exponent >= 0)
    return value_int << fp.exponent;
  const unsigned shift = unsigned(-fp.exponent);
  if ((value_int - (floor_div_pow2(value_int, shift) << shift)) != 0)
    accrue_fp_flags(softfloat_flag_inexact);
  return round_div_pow2(value_int, shift, dest_dtype);
}

elem_t convert_value(elem_t value, uint32_t source_dtype, uint32_t dest_dtype)
{
  const bool source_float = ztt_unit_t::datatype_floating(source_dtype);
  const bool dest_float = ztt_unit_t::datatype_floating(dest_dtype);
  if (!source_float && !dest_float)
    return integer_result(integer_value(value, source_dtype), dest_dtype);
  if (source_float && !dest_float)
    return integer_result(rounded_float_integer(value, source_dtype, dest_dtype), dest_dtype);

  if (!source_float) {
    const ztt_fp_result_t converted = ztt_fp_from_integer(
      integer_value(value, source_dtype), dest_dtype);
    accrue_fp_flags(converted.flags);
    return elem_t(converted.bits);
  }

  // Every standard binary floating-point value is an exact integer
  // significand times a power of two.  Decode that value and round it once
  // directly to the destination format.  In particular, FP64 -> FP8 no
  // longer rounds through FP32.
  const ztt_fp_result_t converted = ztt_round_pack(
    ztt_decode_exact_fp(uint64_t(value), source_dtype), dest_dtype);
  accrue_fp_flags(converted.flags);
  return converted.bits;
}

bool less_than(elem_t a, elem_t b, uint32_t dtype)
{
  return ztt_unit_t::datatype_floating(dtype) ? fp_less(a, b, dtype)
                                               : integer_value(a, dtype) < integer_value(b, dtype);
}

bool greater_equal(elem_t a, elem_t b, uint32_t dtype)
{
  return ztt_unit_t::datatype_floating(dtype) ? fp_greater_equal(a, b, dtype)
                                               : integer_value(a, dtype) >= integer_value(b, dtype);
}

elem_t numeric_add(elem_t a, elem_t b, uint32_t dtype)
{
  return ztt_unit_t::datatype_floating(dtype) ? fp_binary(a, b, dtype, '+')
    : integer_result(integer_value(a, dtype) + integer_value(b, dtype), dtype);
}

elem_t numeric_fma(elem_t a, elem_t b, elem_t c, uint32_t dtype, bool negate)
{
  if (ztt_unit_t::datatype_floating(dtype))
    return fp_fma(negate ? fp_negate(a, dtype) : a, b, c, dtype);
  const big_int product = integer_value(a, dtype) * integer_value(b, dtype);
  return integer_result(integer_value(c, dtype) + (negate ? -product : product), dtype);
}

elem_t numeric_max(elem_t a, elem_t b, uint32_t dtype)
{
  return ztt_unit_t::datatype_floating(dtype) ? fp_minmax(a, b, dtype, true)
                                               : (less_than(a, b, dtype) ? b : a);
}

elem_t numeric_min(elem_t a, elem_t b, uint32_t dtype)
{
  return ztt_unit_t::datatype_floating(dtype) ? fp_minmax(a, b, dtype, false)
                                               : (less_than(a, b, dtype) ? a : b);
}

int compare_exact_fp(elem_t lhs_bits, uint32_t lhs_dtype,
                     elem_t rhs_bits, uint32_t rhs_dtype)
{
  const ztt_exact_fp_t lhs = ztt_decode_exact_fp(uint64_t(lhs_bits), lhs_dtype);
  const ztt_exact_fp_t rhs = ztt_decode_exact_fp(uint64_t(rhs_bits), rhs_dtype);
  if (lhs.significand == 0 && !lhs.infinity &&
      rhs.significand == 0 && !rhs.infinity)
    return 0;
  if (lhs.sign != rhs.sign)
    return lhs.sign ? -1 : 1;
  int magnitude = 0;
  if (lhs.infinity || rhs.infinity) {
    magnitude = lhs.infinity == rhs.infinity ? 0 : lhs.infinity ? 1 : -1;
  } else {
    const int64_t common_exponent = std::min(lhs.exponent, rhs.exponent);
    const big_int left = lhs.significand <<
      unsigned(lhs.exponent - common_exponent);
    const big_int right = rhs.significand <<
      unsigned(rhs.exponent - common_exponent);
    magnitude = left < right ? -1 : left > right ? 1 : 0;
  }
  return lhs.sign ? -magnitude : magnitude;
}

elem_t mixed_numeric_add(elem_t dest, uint32_t dest_dtype,
                         elem_t source, uint32_t source_dtype)
{
  if (!ztt_unit_t::datatype_floating(dest_dtype))
    return integer_result(integer_value(dest, dest_dtype) +
                          integer_value(source, source_dtype), dest_dtype);
  const ztt_fp_result_t result = ztt_fp_addsub_mixed_exact(
    uint64_t(dest), dest_dtype, uint64_t(source), source_dtype, dest_dtype);
  accrue_fp_flags(result.flags);
  return elem_t(result.bits);
}

elem_t mixed_numeric_max(elem_t dest, uint32_t dest_dtype,
                         elem_t source, uint32_t source_dtype)
{
  if (!ztt_unit_t::datatype_floating(dest_dtype)) {
    return integer_value(dest, dest_dtype) < integer_value(source, source_dtype)
      ? convert_value(source, source_dtype, dest_dtype) : dest;
  }

  const ztt_float_kind_t dest_kind =
    ztt_unit_t::datatype_float_kind(dest_dtype);
  const ztt_float_kind_t source_kind =
    ztt_unit_t::datatype_float_kind(source_dtype);
  if (fp_is_nan(dest, dest_kind) || fp_is_nan(source, source_kind)) {
    if (fp_is_signaling_nan(dest, dest_kind) ||
        fp_is_signaling_nan(source, source_kind))
      accrue_fp_flags(softfloat_flag_invalid);
    return ztt_fp_canonical_nan(dest_dtype);
  }

  const int order = compare_exact_fp(dest, dest_dtype, source, source_dtype);
  if (order < 0)
    return convert_value(source, source_dtype, dest_dtype);
  if (order == 0) {
    const ztt_exact_fp_t d = ztt_decode_exact_fp(uint64_t(dest), dest_dtype);
    const ztt_exact_fp_t s = ztt_decode_exact_fp(uint64_t(source), source_dtype);
    if (d.significand == 0 && s.significand == 0 && d.sign && !s.sign)
      return convert_value(source, source_dtype, dest_dtype);
  }
  return dest;
}

int scalar_exponent(elem_t value, unsigned width);

elem_t apply_integer_ew(ew_op op, elem_t a, elem_t b, elem_t old,
                        uint32_t dtype, unsigned scalar_width,
                        uint32_t exponent_dtype)
{
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  const elem_t mask = ztt_unit_t::element_mask(width);
  const big_int av = integer_value(a, dtype);
  const big_int bv = integer_value(b, dtype);
  const big_int ov = integer_value(old, dtype);
  const big_int exponent = exponent_dtype
                         ? integer_value(b, exponent_dtype) : bv;
  switch (op) {
    case ew_op::abs: return integer_result(av < 0 ? -av : av, dtype);
    case ew_op::absdiff: return integer_result(av < bv ? bv - av : av - bv, dtype);
    case ew_op::add: return integer_result(av + bv, dtype);
    case ew_op::bit_and: return a & b;
    case ew_op::bit_andnot: return a & ~b & mask;
    case ew_op::cmpge: return av >= bv ? mask : 0;
    case ew_op::cmplt: return av < bv ? mask : 0;
    case ew_op::exp2:
      if (av < 0)
        return 0;
      return integer_result(big_int(1) << bounded_shift(av, width), dtype);
    case ew_op::hdiff: return integer_result(round_div_pow2(bv - av, 1, dtype), dtype);
    case ew_op::hdiff_scalar: return integer_result(round_div_pow2(bv - av, 1, dtype), dtype);
    case ew_op::ldexp: return integer_result(integer_scale(av, exponent, width, dtype), dtype);
    case ew_op::ldexp_scalar:
      return integer_result(integer_scale(bv, scalar_exponent(a, scalar_width), width, dtype), dtype);
    case ew_op::ldexpacc:
      return integer_result(integer_scale_acc_exact(ov, av, exponent, width, dtype), dtype);
    case ew_op::ldexpacc_scalar:
      return integer_result(integer_scale_acc_exact(
        ov, bv, scalar_exponent(a, scalar_width), width, dtype), dtype);
    case ew_op::log2: {
      const elem_t magnitude = low_element(av < 0 ? -av : av, width);
      return magnitude ? ilog2(magnitude) : 0;
    }
    case ew_op::log2sub: {
      const elem_t magnitude = low_element(av < 0 ? -av : av, width);
      return integer_result(big_int(magnitude ? ilog2(magnitude) : 0) - bv, dtype);
    }
    case ew_op::log2sub_scalar: {
      const elem_t magnitude = low_element(bv < 0 ? -bv : bv, width);
      return integer_result(big_int(magnitude ? ilog2(magnitude) : 0) - av, dtype);
    }
    case ew_op::max: return av < bv ? b : a;
    case ew_op::mean: return integer_result(round_div_pow2(av + bv, 1, dtype), dtype);
    case ew_op::min: return av < bv ? a : b;
    case ew_op::mul: return integer_result(av * bv, dtype);
    case ew_op::mulacc: return integer_result(ov + av * bv, dtype);
    case ew_op::mulaccneg: return integer_result(ov - av * bv, dtype);
    case ew_op::muladd: return integer_result(av + bv * ov, dtype);
    case ew_op::mulneg: return integer_result(-av * bv, dtype);
    case ew_op::mulsub: return integer_result(av - bv * ov, dtype);
    case ew_op::bit_or: return a | b;
    case ew_op::bit_ornot: return a | (~b & mask);
    case ew_op::rdexp: return integer_result(integer_scale(av, -exponent, width, dtype), dtype);
    case ew_op::rdexpacc:
      return integer_result(integer_scale_acc_exact(ov, av, -exponent, width, dtype), dtype);
    case ew_op::sub: return integer_result(bv - av, dtype);
    case ew_op::sub_scalar: return integer_result(bv - av, dtype);
    case ew_op::sublog2: {
      const elem_t magnitude = low_element(av < 0 ? -av : av, width);
      return integer_result(bv - big_int(magnitude ? ilog2(magnitude) : 0), dtype);
    }
    case ew_op::sublog2_scalar: {
      const elem_t magnitude = low_element(bv < 0 ? -bv : bv, width);
      return integer_result(av - big_int(magnitude ? ilog2(magnitude) : 0), dtype);
    }
    case ew_op::bit_xor: return a ^ b;
  }
  return 0;
}

int fp_exponent(elem_t value, uint32_t dtype, bool& valid)
{
  const double exponent = fp_to_host(value, dtype);
  valid = std::isfinite(exponent);
  if (!valid)
    return 0;
  if (exponent > 4096)
    return 4096;
  if (exponent < -4096)
    return -4096;
  return int(exponent);
}

int scalar_exponent(elem_t value, unsigned width)
{
  big_int exponent = uint64_t(value);
  const unsigned dtype_width = ztt_unit_t::datatype_bits(ztt::kMaxIntDatatype);
  if (dtype_width < width)
    exponent = integer_value(value, ztt::kMaxIntDatatype);
  if (exponent > 4096)
    return 4096;
  if (exponent < -4096)
    return -4096;
  return int(exponent);
}

elem_t apply_float_ew(ztt_opcode_t opcode, ew_op op,
                      elem_t a, elem_t b, elem_t old,
                      uint32_t dtype, unsigned scalar_width,
                      uint32_t integer_operand_dtype)
{
  const elem_t mask = ztt_unit_t::element_mask(ztt_unit_t::datatype_bits(dtype));
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  const bool uses_old = op == ew_op::ldexpacc ||
    op == ew_op::ldexpacc_scalar || op == ew_op::mulacc ||
    op == ew_op::mulaccneg || op == ew_op::muladd ||
    op == ew_op::mulsub || op == ew_op::rdexpacc;
  const bool ordered_compare = op == ew_op::cmpge || op == ew_op::cmplt;
  ztt_fp_trans_op trans_op;
  const bool transcendental = ztt_fp_transcendental_operation(opcode, trans_op);
  const bool specialized_nan = ordered_compare || op == ew_op::max ||
                               op == ew_op::min || transcendental;
  const bool scalar_integer_bias = integer_operand_dtype != 0 &&
    (op == ew_op::log2sub_scalar || op == ew_op::sublog2_scalar);
  const bool a_is_float = scalar_width == 0 && !scalar_integer_bias;
  const bool b_is_float = integer_operand_dtype == 0 || scalar_integer_bias;
  if (!specialized_nan) {
    const bool a_nan = a_is_float && fp_is_nan(a, kind);
    const bool b_nan = b_is_float && fp_is_nan(b, kind);
    const bool old_nan = uses_old && fp_is_nan(old, kind);
    if ((a_nan && fp_is_signaling_nan(a, kind)) ||
        (b_nan && fp_is_signaling_nan(b, kind)) ||
        (old_nan && fp_is_signaling_nan(old, kind)))
      accrue_fp_flags(softfloat_flag_invalid);
    if (a_nan || b_nan || old_nan)
      return ztt_fp_canonical_nan(dtype);
  }
  bool valid_exponent = true;
  int exponent = 0;
  if (integer_operand_dtype && !scalar_integer_bias) {
    const big_int integral = integer_value(b, integer_operand_dtype);
    exponent = integral > 4096 ? 4096 : integral < -4096 ? -4096
                                                       : integral.convert_to<int>();
  } else {
    exponent = fp_exponent(b, dtype, valid_exponent);
  }
  auto scaled = [&](elem_t value, int amount) {
    const ztt_fp_result_t scaled =
      ztt_fp_scale_exact(uint64_t(value), dtype, amount);
    accrue_fp_flags(scaled.flags);
    return elem_t(scaled.bits);
  };
  auto scaled_acc = [&](elem_t accumulator, elem_t value, int amount) {
    const ztt_fp_result_t scaled = ztt_fp_scale_acc_exact(
      uint64_t(accumulator), uint64_t(value), dtype, amount);
    accrue_fp_flags(scaled.flags);
    return elem_t(scaled.bits);
  };
  auto logarithm = [&](elem_t value, bool absolute) {
    if (absolute)
      value &= ~(elem_t(1) << (ztt_unit_t::datatype_bits(dtype) - 1));
    const ztt_fp_result_t trans = ztt_fp_transcendental(
      ztt_fp_trans_op::log2, dtype, dtype, uint64_t(value));
    accrue_fp_flags(trans.flags);
    return elem_t(trans.bits);
  };
  auto exact_addsub = [&](elem_t lhs, elem_t rhs, bool subtract_rhs,
                          int64_t scale_amount, bool absolute) {
    const ztt_fp_result_t result = ztt_fp_addsub_exact(
      uint64_t(lhs), uint64_t(rhs), dtype, subtract_rhs, scale_amount,
      absolute);
    accrue_fp_flags(result.flags);
    return elem_t(result.bits);
  };
  auto exact_mul = [&](elem_t lhs, elem_t rhs, bool negate) {
    const ztt_fp_result_t result = ztt_fp_mul_exact(
      uint64_t(lhs), uint64_t(rhs), dtype, negate);
    accrue_fp_flags(result.flags);
    return elem_t(result.bits);
  };
  auto exact_sub_integer = [&](elem_t fp, elem_t integer_bits,
                               bool integer_minus_fp) {
    const ztt_fp_result_t result = ztt_fp_sub_integer_exact(
      uint64_t(fp), integer_value(integer_bits, integer_operand_dtype), dtype,
      integer_minus_fp);
    accrue_fp_flags(result.flags);
    return elem_t(result.bits);
  };
  switch (op) {
    case ew_op::abs: return a & ~(elem_t(1) << (ztt_unit_t::datatype_bits(dtype) - 1));
    case ew_op::absdiff: return exact_addsub(a, b, true, 0, true);
    case ew_op::add: return fp_binary(a, b, dtype, '+');
    case ew_op::cmpge: return fp_greater_equal(a, b, dtype) ? mask : 0;
    case ew_op::cmplt: return fp_less(a, b, dtype) ? mask : 0;
    case ew_op::exp2: {
      const ztt_fp_result_t trans = ztt_fp_transcendental(
        ztt_fp_trans_op::exp2, dtype, dtype, uint64_t(a));
      accrue_fp_flags(trans.flags);
      return elem_t(trans.bits);
    }
    case ew_op::hdiff: {
      return exact_addsub(b, a, true, -1, false);
    }
    case ew_op::hdiff_scalar: {
      return exact_addsub(b, a, true, -1, false);
    }
    case ew_op::ldexp:
      return valid_exponent ? scaled(a, exponent) : ztt_fp_canonical_nan(dtype);
    case ew_op::ldexp_scalar: return scaled(b, scalar_exponent(a, scalar_width));
    case ew_op::ldexpacc:
      return valid_exponent ? scaled_acc(old, a, exponent) : ztt_fp_canonical_nan(dtype);
    case ew_op::ldexpacc_scalar:
      return scaled_acc(old, b, scalar_exponent(a, scalar_width));
    case ew_op::log2: return logarithm(a, false);
    case ew_op::log2sub:
      return integer_operand_dtype
        ? exact_sub_integer(logarithm(a, true), b, false)
        : fp_binary(logarithm(a, true), b, dtype, '-');
    case ew_op::log2sub_scalar:
      return integer_operand_dtype
        ? exact_sub_integer(logarithm(b, true), a, false)
        : fp_binary(logarithm(b, true), a, dtype, '-');
    case ew_op::max: return fp_minmax(a, b, dtype, true);
    case ew_op::mean: {
      return exact_addsub(a, b, false, -1, false);
    }
    case ew_op::min: return fp_minmax(a, b, dtype, false);
    case ew_op::mul: return fp_binary(a, b, dtype, '*');
    case ew_op::mulacc: return fp_fma(a, b, old, dtype);
    case ew_op::mulaccneg: return fp_fma(fp_negate(a, dtype), b, old, dtype);
    case ew_op::muladd: return fp_fma(b, old, a, dtype);
    case ew_op::mulneg: {
      return exact_mul(a, b, true);
    }
    case ew_op::mulsub: return fp_fma(fp_negate(b, dtype), old, a, dtype);
    case ew_op::rdexp:
      return valid_exponent ? scaled(a, -exponent) : ztt_fp_canonical_nan(dtype);
    case ew_op::rdexpacc:
      return valid_exponent ? scaled_acc(old, a, -exponent) : ztt_fp_canonical_nan(dtype);
    case ew_op::sub: return fp_binary(b, a, dtype, '-');
    case ew_op::sub_scalar: return fp_binary(b, a, dtype, '-');
    case ew_op::sublog2:
      return integer_operand_dtype
        ? exact_sub_integer(logarithm(a, true), b, true)
        : fp_binary(b, logarithm(a, true), dtype, '-');
    case ew_op::sublog2_scalar:
      return integer_operand_dtype
        ? exact_sub_integer(logarithm(b, true), a, true)
        : fp_binary(a, logarithm(b, true), dtype, '-');
    case ew_op::bit_and: return a & b;
    case ew_op::bit_andnot: return a & ~b & mask;
    case ew_op::bit_or: return a | b;
    case ew_op::bit_ornot: return a | (~b & mask);
    case ew_op::bit_xor: return a ^ b;
  }
  return 0;
}

elem_t apply_ew(ztt_opcode_t opcode, ew_op op, elem_t a, elem_t b, elem_t old,
                uint32_t dtype, unsigned scalar_width,
                uint32_t integer_operand_dtype)
{
  return ztt_unit_t::datatype_floating(dtype)
       ? apply_float_ew(opcode, op, a, b, old, dtype, scalar_width,
                        integer_operand_dtype)
       : apply_integer_ew(op, a, b, old, dtype, scalar_width,
                          integer_operand_dtype);
}

void execute_elementwise(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                         ew_op op,
                         bool scalar, bool unary = false,
                         bool scalar_is_exponent = false)
{
  const unsigned dest = insn.rd();
  const unsigned source1 = insn.rs1();
  const unsigned source2 = insn.rs2();
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, dest);
  const ztt_m_meta_t source1_meta = !scalar
    ? ztt_read_m_meta(proc, source1) : ztt_m_meta_t{};
  const ztt_m_meta_t source2_meta = !unary
    ? ztt_read_m_meta(proc, source2) : ztt_m_meta_t{};
  const uint32_t scalar_dtype = scalar && !scalar_is_exponent
    ? proc.ZTU.amestype() : 0;

  ztt_dtype_tuple_t tuple;
  if (unary)
    tuple = unary_tuple(dest_meta.dtype, source1_meta.dtype);
  else if (scalar && !scalar_is_exponent)
    tuple = scalar_tuple(dest_meta.dtype, source2_meta.dtype, scalar_dtype);
  else if (scalar)
    tuple = unary_tuple(dest_meta.dtype, source2_meta.dtype);
  else
    tuple = binary_tuple(dest_meta.dtype, source1_meta.dtype,
                         source2_meta.dtype);
  if (!ztt_validate_tuple(proc, opcode, tuple))
    return;

  std::vector<ztt_m_meta_t> operand_meta{dest_meta};
  if (!scalar)
    operand_meta.push_back(source1_meta);
  if (!unary)
    operand_meta.push_back(source2_meta);
  std::size_t instruction_squares = 1;
  for (const auto& meta : operand_meta)
    instruction_squares = std::max(
      instruction_squares, ztt_unit_t::square_count(meta.dtype));
  std::size_t operand_index = 0;
  const ztt_m_operand_t dest_operand = ztt_form_m_operand(
    proc, insn, operand_meta[operand_index++], instruction_squares);
  const ztt_m_operand_t source1_operand = !scalar
    ? ztt_form_m_operand(proc, insn, operand_meta[operand_index++],
                         instruction_squares)
    : ztt_m_operand_t{};
  const ztt_m_operand_t source2_operand = !unary
    ? ztt_form_m_operand(proc, insn, operand_meta[operand_index++],
                         instruction_squares)
    : ztt_m_operand_t{};

  const uint32_t dd = dest_meta.dtype;
  const uint32_t d1 = source1_meta.dtype;
  const uint32_t d2 = source2_meta.dtype;
  std::vector<elem_t> old, a, b;
  if (!snapshot_m(proc, dest_operand, old) ||
      (!scalar && !snapshot_m(proc, source1_operand, a)) ||
      (!unary && !snapshot_m(proc, source2_operand, b)))
    return;
  const bool vector_exponent = !scalar &&
    (op == ew_op::ldexp || op == ew_op::ldexpacc ||
     op == ew_op::rdexp || op == ew_op::rdexpacc);
  const bool compound_log = op == ew_op::log2sub ||
    op == ew_op::log2sub_scalar || op == ew_op::sublog2 ||
    op == ew_op::sublog2_scalar;
  const bool integer_bias = compound_log &&
    ztt_unit_t::datatype_integer(scalar ? scalar_dtype : d2);

  if (scalar) {
    elem_t scalar_value = proc.get_state()->XPR[source1];
    if (!scalar_is_exponent) {
      const unsigned scalar_bits = ztt_unit_t::datatype_bits(scalar_dtype);
      if (scalar_bits < proc.get_xlen())
        scalar_value &= ztt_unit_t::element_mask(scalar_bits);
      if (!integer_bias)
        scalar_value = convert_value(scalar_value, scalar_dtype, d2);
    }
    a.assign(old.size(), scalar_value);
  }
  if (unary)
    b.assign(old.size(), 0);
  const bool comparison = op == ew_op::cmpge || op == ew_op::cmplt;
  const uint32_t operation_dtype = comparison ? (scalar ? d2 : d1) : dd;
  std::vector<elem_t> result(old.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = apply_ew(opcode, op, a[i], b[i], old[i], operation_dtype,
                         scalar_is_exponent ? proc.get_xlen() : 0,
                         vector_exponent ? d2 : integer_bias
                           ? (scalar ? scalar_dtype : d2) : 0);
  commit_m(proc, dest_operand, result);
}

double round_nearest_even(double value)
{
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5)
    return lower;
  if (fraction > 0.5)
    return lower + 1.0;
  return std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
}

elem_t apply_fp_unary(ztt_opcode_t opcode, fp_unary_op op, elem_t value,
                      uint32_t dtype)
{
  ztt_fp_trans_op trans_op;
  if (ztt_fp_transcendental_operation(opcode, trans_op)) {
    const ztt_fp_result_t trans = ztt_fp_transcendental(
      trans_op, dtype, dtype, uint64_t(value));
    accrue_fp_flags(trans.flags);
    return elem_t(trans.bits);
  }
  const ztt_float_kind_t kind = ztt_unit_t::datatype_float_kind(dtype);
  if (fp_is_nan(value, kind)) {
    if (fp_is_signaling_nan(value, kind))
      accrue_fp_flags(softfloat_flag_invalid);
    return ztt_fp_canonical_nan(dtype);
  }
  const double input = fp_to_host(value, dtype);
  double result = input;
  switch (op) {
    case fp_unary_op::cos:
    case fp_unary_op::sin:
    case fp_unary_op::tanh:
      break;
    case fp_unary_op::rec:
      return fp_div(host_to_fp(1.0, dtype), value, dtype);
    case fp_unary_op::rsqrt:
      return fp_div(host_to_fp(1.0, dtype), fp_sqrt(value, dtype), dtype);
    case fp_unary_op::sqrt:
      return fp_sqrt(value, dtype);
    case fp_unary_op::frintm: result = std::floor(input); break;
    case fp_unary_op::frintn: result = round_nearest_even(input); break;
    case fp_unary_op::frintp: result = std::ceil(input); break;
    case fp_unary_op::frintz: result = std::trunc(input); break;
  }
  if (std::isfinite(input) &&
      (op == fp_unary_op::frintm || op == fp_unary_op::frintn ||
       op == fp_unary_op::frintp || op == fp_unary_op::frintz) &&
      result != input)
    accrue_fp_flags(softfloat_flag_inexact);
  if (result == 0.0 && std::signbit(input))
    result = -0.0;
  return host_to_fp(result, dtype);
}

void execute_fp_unary(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                      fp_unary_op op)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, insn.rs1());
  if (!ztt_validate_tuple(proc, opcode,
                          unary_tuple(dest_meta.dtype, source_meta.dtype)))
    return;
  const auto operands = form_m_operands(proc, insn, {dest_meta, source_meta});
  std::vector<elem_t> dest, source;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], source))
    return;
  for (std::size_t i = 0; i < dest.size(); ++i)
    dest[i] = apply_fp_unary(opcode, op, source[i], dest_meta.dtype);
  commit_m(proc, operands[0], dest);
}

void execute_conditional(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                         bool on_negative, bool preserve)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t pred_meta = ztt_read_m_meta(proc, insn.rs1());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, insn.rs2());
  if (!ztt_validate_tuple(proc, opcode,
                          binary_tuple(dest_meta.dtype, pred_meta.dtype,
                                       source_meta.dtype)))
    return;
  const auto operands = form_m_operands(
    proc, insn, {dest_meta, pred_meta, source_meta});
  std::vector<elem_t> dest, pred, source;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], pred) ||
      !snapshot_m(proc, operands[2], source))
    return;
  for (std::size_t i = 0; i < dest.size(); ++i) {
    const bool selected = on_negative
      ? less_than(pred[i], 0, pred_meta.dtype)
      : greater_equal(pred[i], 0, pred_meta.dtype);
    if (selected)
      dest[i] = source[i];
    else if (!preserve)
      dest[i] = 0;
  }
  commit_m(proc, operands[0], dest);
}

void execute_prefix_reduce(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                           bool column, bool maximum, bool reduce,
                           bool minimum = false)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, insn.rs1());
  if (!ztt_validate_tuple(proc, opcode,
                          unary_tuple(dest_meta.dtype, source_meta.dtype)))
    return;
  const auto operands = form_m_operands(proc, insn, {dest_meta, source_meta});
  std::vector<elem_t> dest, source;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], source))
    return;
  const uint32_t dtype = dest_meta.dtype;
  for (auto& value : source)
    value = convert_value(value, source_meta.dtype, dtype);
  const std::size_t squares = operands[0].squares;
  for (std::size_t sq = 0; sq < squares; ++sq) {
    const std::size_t base = sq * ztt::kNumElements;
    for (std::size_t outer = 0; outer < ztt::kTileLength; ++outer) {
      elem_t running = source[base + (column ? outer : outer * ztt::kTileLength)];
      for (std::size_t inner = 0; inner < ztt::kTileLength; ++inner) {
        const std::size_t index = base + (column
          ? inner * ztt::kTileLength + outer
          : outer * ztt::kTileLength + inner);
        if (inner)
          running = maximum ? numeric_max(running, source[index], dtype)
                    : minimum ? numeric_min(running, source[index], dtype)
                              : numeric_add(running, source[index], dtype);
        if (!reduce)
          dest[index] = running;
      }
      if (reduce) {
        for (std::size_t inner = 0; inner < ztt::kTileLength; ++inner) {
          const std::size_t index = base + (column
            ? inner * ztt::kTileLength + outer
            : outer * ztt::kTileLength + inner);
          dest[index] = running;
        }
      }
    }
  }
  commit_m(proc, operands[0], dest);
}

void execute_gather_scatter(processor_t& proc, ztt_opcode_t opcode,
                            insn_t insn, bool column, bool maximum)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t values_meta = ztt_read_m_meta(proc, insn.rs1());
  const ztt_m_meta_t indices_meta = ztt_read_m_meta(proc, insn.rs2());
  if (!ztt_validate_tuple(proc, opcode,
                          binary_tuple(dest_meta.dtype, values_meta.dtype,
                                       indices_meta.dtype)))
    return;
  const auto operands = form_m_operands(
    proc, insn, {dest_meta, values_meta, indices_meta});
  std::vector<elem_t> dest, values, indices;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], values) ||
      !snapshot_m(proc, operands[2], indices))
    return;
  const uint32_t dest_dtype = dest_meta.dtype;
  const uint32_t source_dtype = values_meta.dtype;
  const std::size_t squares = operands[0].squares;
  for (std::size_t sq = 0; sq < squares; ++sq) {
    const std::size_t base = sq * ztt::kNumElements;
    for (std::size_t r = 0; r < ztt::kTileLength; ++r)
      for (std::size_t c = 0; c < ztt::kTileLength; ++c) {
        const std::size_t src = base + r * ztt::kTileLength + c;
        const std::size_t index = uint32_t(indices[src]) & (ztt::kTileLength - 1);
        const std::size_t dst = base + (column
          ? r * ztt::kTileLength + index
          : index * ztt::kTileLength + c);
        if (maximum)
          dest[dst] = mixed_numeric_max(dest[dst], dest_dtype,
                                        values[src], source_dtype);
        else
          dest[dst] = mixed_numeric_add(dest[dst], dest_dtype,
                                        values[src], source_dtype);
      }
  }
  commit_m(proc, operands[0], dest);
}

void execute_gather(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                    bool column)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, insn.rs1());
  const ztt_m_meta_t indices_meta = ztt_read_m_meta(proc, insn.rs2());
  if (!ztt_validate_tuple(proc, opcode,
                          binary_tuple(dest_meta.dtype, source_meta.dtype,
                                       indices_meta.dtype)))
    return;
  const auto operands = form_m_operands(
    proc, insn, {dest_meta, source_meta, indices_meta});
  std::vector<elem_t> dest, source, indices;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], source) ||
      !snapshot_m(proc, operands[2], indices))
    return;
  const std::size_t squares = ztt_unit_t::square_count(dest_meta.dtype);
  for (std::size_t sq = 0; sq < squares; ++sq) {
    const std::size_t base = sq * ztt::kNumElements;
    for (std::size_t row = 0; row < ztt::kTileLength; ++row)
      for (std::size_t col = 0; col < ztt::kTileLength; ++col) {
        const std::size_t index = base + row * ztt::kTileLength + col;
        const std::size_t selected = uint32_t(indices[index]) &
                                     (ztt::kTileLength - 1);
        dest[index] = column
          ? source[base + row * ztt::kTileLength + selected]
          : source[base + selected * ztt::kTileLength + col];
      }
  }
  commit_m(proc, operands[0], dest);
}

void execute_axis_control(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                          bool rows, bool broadcast)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, insn.rs2());
  if (!ztt_validate_tuple(proc, opcode,
                          unary_tuple(dest_meta.dtype, source_meta.dtype)))
    return;
  const auto operands = form_m_operands(proc, insn, {dest_meta, source_meta});
  std::vector<elem_t> dest, source;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], source))
    return;
  const uint32_t raw = uint32_t(proc.get_state()->XPR[insn.rs1()]);
  if (broadcast && raw >= ztt::kTileLength)
    illegal(insn);
  const int64_t offset = int64_t(int32_t(raw));
  for (std::size_t row = 0; row < ztt::kTileLength; ++row)
    for (std::size_t col = 0; col < ztt::kTileLength; ++col) {
      int64_t source_row = int64_t(row), source_col = int64_t(col);
      if (broadcast)
        (rows ? source_row : source_col) = int64_t(raw);
      else
        (rows ? source_row : source_col) += offset;
      const std::size_t out = row * ztt::kTileLength + col;
      dest[out] = source_row >= 0 && source_col >= 0 &&
                  source_row < int64_t(ztt::kTileLength) &&
                  source_col < int64_t(ztt::kTileLength)
        ? source[std::size_t(source_row) * ztt::kTileLength +
                 std::size_t(source_col)] : 0;
    }
  commit_m(proc, operands[0], dest);
}

void execute_axis_id(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                     bool rows)
{
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  if (!ztt_validate_tuple(proc, opcode, one_tuple(dest_meta.dtype)))
    return;
  const ztt_m_operand_t dest_operand =
    ztt_form_m_operand(proc, insn, dest_meta);
  std::vector<elem_t> dest;
  if (!snapshot_m(proc, dest_operand, dest))
    return;
  const uint32_t dtype = dest_meta.dtype;
  const std::size_t squares = ztt_unit_t::square_count(dtype);
  for (std::size_t square = 0; square < squares; ++square)
    for (std::size_t row = 0; row < ztt::kTileLength; ++row)
      for (std::size_t col = 0; col < ztt::kTileLength; ++col)
        dest[square * ztt::kNumElements + row * ztt::kTileLength + col] =
          rows ? row : col;
  commit_m(proc, dest_operand, dest);
}

enum class shift_op { left, arithmetic_right, logical_right };

void execute_shift(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                   shift_op op, bool scalar)
{
  const unsigned source_reg = scalar ? insn.rs2() : insn.rs1();
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, source_reg);
  const ztt_m_meta_t amounts_meta = !scalar
    ? ztt_read_m_meta(proc, insn.rs2()) : ztt_m_meta_t{};
  const ztt_dtype_tuple_t tuple = scalar
    ? unary_tuple(dest_meta.dtype, source_meta.dtype)
    : binary_tuple(dest_meta.dtype, source_meta.dtype, amounts_meta.dtype);
  if (!ztt_validate_tuple(proc, opcode, tuple))
    return;
  const auto operands = scalar
    ? form_m_operands(proc, insn, {dest_meta, source_meta})
    : form_m_operands(proc, insn,
                      {dest_meta, source_meta, amounts_meta});

  std::vector<elem_t> dest, source, amounts;
  if (!snapshot_m(proc, operands[0], dest) ||
      !snapshot_m(proc, operands[1], source) ||
      (!scalar && !snapshot_m(proc, operands[2], amounts)))
    return;
  const uint32_t source_dtype = source_meta.dtype;
  const uint32_t dest_dtype = dest_meta.dtype;
  const unsigned width = ztt_unit_t::datatype_bits(source_dtype);
  const elem_t mask = ztt_unit_t::element_mask(width);
  for (std::size_t i = 0; i < dest.size(); ++i) {
    const unsigned amount = unsigned(scalar
      ? proc.get_state()->XPR[insn.rs1()] : uint64_t(amounts[i])) % width;
    if (op == shift_op::left)
      dest[i] = integer_result(integer_value(source[i], source_dtype) << amount,
                               dest_dtype);
    else if (op == shift_op::logical_right)
      dest[i] = integer_result(unsigned_value(source[i] & mask) >> amount,
                               dest_dtype);
    else
      dest[i] = integer_result(integer_value(source[i], source_dtype) >> amount,
                               dest_dtype);
  }
  commit_m(proc, operands[0], dest);
}

void execute_zip_pair(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                      bool rows, bool zip)
{
  const unsigned r0 = insn.rs1(), r1 = insn.rs2();
  const ztt_m_meta_t meta0 = ztt_read_m_meta(proc, r0);
  const ztt_m_meta_t meta1 = ztt_read_m_meta(proc, r1);
  ztt_dtype_tuple_t tuple;
  tuple.has_source1 = true;
  tuple.source1 = meta0.dtype;
  tuple.has_source2 = true;
  tuple.source2 = meta1.dtype;
  if (!ztt_validate_tuple(proc, opcode, tuple))
    return;
  form_m_operands(proc, insn, {meta0, meta1});
  std::vector<elem_t> a, b;
  if (!snapshot_m(proc, meta0, a) || !snapshot_m(proc, meta1, b))
    return;
  std::vector<elem_t> out0(ztt::kNumElements), out1(ztt::kNumElements);
  if (rows) {
    for (std::size_t global = 0; global < 2 * ztt::kTileLength; ++global)
      for (std::size_t col = 0; col < ztt::kTileLength; ++col) {
        if (zip) {
          const std::vector<elem_t>& src = global & 1 ? b : a;
          std::vector<elem_t>& dst = global < ztt::kTileLength ? out0 : out1;
          dst[(global % ztt::kTileLength) * ztt::kTileLength + col] =
            src[(global / 2) * ztt::kTileLength + col];
        } else {
          const std::vector<elem_t>& src = global < ztt::kTileLength ? a : b;
          std::vector<elem_t>& dst = global & 1 ? out1 : out0;
          dst[(global / 2) * ztt::kTileLength + col] =
            src[(global % ztt::kTileLength) * ztt::kTileLength + col];
        }
      }
  } else {
    for (std::size_t row = 0; row < ztt::kTileLength; ++row)
      for (std::size_t global = 0; global < 2 * ztt::kTileLength; ++global) {
        if (zip) {
          const std::vector<elem_t>& src = global & 1 ? b : a;
          std::vector<elem_t>& dst = global < ztt::kTileLength ? out0 : out1;
          dst[row * ztt::kTileLength + global % ztt::kTileLength] =
            src[row * ztt::kTileLength + global / 2];
        } else {
          const std::vector<elem_t>& src = global < ztt::kTileLength ? a : b;
          std::vector<elem_t>& dst = global & 1 ? out1 : out0;
          dst[row * ztt::kTileLength + global / 2] =
            src[row * ztt::kTileLength + global % ztt::kTileLength];
        }
      }
  }
  proc.ZTU.write_m_atomic({
    {r0, meta0.dtype, out0},
    {r1, meta1.dtype, out1},
  });
}

void execute_acc_move(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                      bool to_acc)
{
  const unsigned mreg = to_acc ? insn.rs1() : insn.rd();
  const unsigned areg = to_acc ? insn.rd() : insn.rs1();
  const ztt_m_meta_t m_meta = ztt_read_m_meta(proc, mreg);
  const ztt_acc_meta_t acc_meta = ztt_read_acc_meta(proc, areg);
  const ztt_dtype_tuple_t tuple = unary_tuple(
    to_acc ? acc_meta.dtype : m_meta.dtype,
    to_acc ? m_meta.dtype : acc_meta.dtype);
  if (!ztt_validate_tuple(proc, opcode, tuple))
    return;
  ztt_form_m_operand(proc, insn, m_meta);
  const uint32_t dtype = m_meta.dtype;
  const std::size_t pack = ztt_unit_t::square_count(dtype);
  ztt_form_acc_span(insn, areg, pack);
  if (to_acc) {
    std::vector<elem_t> source;
    if (!snapshot_m(proc, m_meta, source))
      return;
    std::vector<ztt_unit_t::acc_write_t> writes;
    writes.reserve(pack);
    for (std::size_t square = 0; square < pack; ++square) {
      const auto begin = source.begin() + square * ztt::kNumElements;
      writes.push_back({areg + square, dtype,
        std::vector<elem_t>(begin, begin + ztt::kNumElements)});
    }
    proc.ZTU.write_acc_atomic(writes);
  } else {
    std::vector<elem_t> dest(pack * ztt::kNumElements);
    for (std::size_t square = 0; square < pack; ++square) {
      std::vector<elem_t> source;
      proc.ZTU.read_acc_as(areg + square, acc_meta.dtype, source);
      std::copy(source.begin(), source.end(),
                dest.begin() + square * ztt::kNumElements);
    }
    proc.ZTU.write_m(mreg, dest);
  }
}

void execute_debug_move(processor_t& proc, insn_t insn, unsigned width,
                        bool to_m)
{
  const unsigned mreg = to_m ? insn.rd() : insn.rs1();
  if (width > ztt::kTileLength * ztt::kUnitDatatypeBits)
    illegal(insn);
  const std::size_t count = ztt::kMRegisterBits / width;
  const std::size_t position = std::size_t(proc.get_state()->XPR[insn.rs2()]) &
                               (count - 1);
  const std::size_t bit_offset = position * width;
  const std::size_t word = bit_offset / 64;
  const unsigned shift = bit_offset % 64;
  const uint64_t mask = width == 64 ? ~UINT64_C(0)
                                    : (UINT64_C(1) << width) - 1;
  if (to_m) {
    ztt_unit_t::m_register_t raw = proc.ZTU.m_register(mreg);
    raw[word] = (raw[word] & ~(mask << shift)) |
                ((uint64_t(proc.get_state()->XPR[insn.rs1()]) & mask) << shift);
    proc.ZTU.write_m_register(mreg, raw);
  } else {
    const uint64_t value = (proc.ZTU.m_register(mreg)[word] >> shift) & mask;
    proc.get_state()->XPR.write(insn.rd(), value);
  }
}

void execute_matrix(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                    bool transpose_a, bool transpose_b, bool accumulate,
                    bool negate)
{
  const unsigned acc = ztt_acc_98(insn);
  const ztt_acc_meta_t acc_meta = ztt_read_acc_meta(proc, acc);
  const ztt_m_meta_t source1_meta = ztt_read_m_meta(proc, insn.rs1());
  const ztt_m_meta_t source2_meta = ztt_read_m_meta(proc, insn.rs2());
  if (!ztt_validate_tuple(proc, opcode,
                          binary_tuple(acc_meta.dtype, source1_meta.dtype,
                                       source2_meta.dtype)))
    return;
  const auto operands = form_m_operands(
    proc, insn, {source1_meta, source2_meta});
  std::vector<elem_t> dest, a, b;
  if (!snapshot_acc(proc, acc_meta, dest) ||
      !snapshot_m(proc, operands[0], a) ||
      !snapshot_m(proc, operands[1], b))
    return;
  const uint32_t da = acc_meta.dtype;
  const uint32_t d1 = source1_meta.dtype;
  const uint32_t d2 = source2_meta.dtype;
  const std::size_t squares = a.size() / ztt::kNumElements;
  const std::size_t inner_extent = squares * ztt::kTileLength;
  std::vector<elem_t> result(ztt::kNumElements, 0);
  for (std::size_t r = 0; r < ztt::kTileLength; ++r)
    for (std::size_t c = 0; c < ztt::kTileLength; ++c) {
      elem_t sum = accumulate ? dest[r * ztt::kTileLength + c] : 0;
      for (std::size_t k = 0; k < inner_extent; ++k) {
        const std::size_t square = k / ztt::kTileLength;
        const std::size_t inner = k % ztt::kTileLength;
        const std::size_t base = square * ztt::kNumElements;
        const std::size_t ia = base + (transpose_a
          ? inner * ztt::kTileLength + r : r * ztt::kTileLength + inner);
        const std::size_t ib = base + (transpose_b
          ? c * ztt::kTileLength + inner : inner * ztt::kTileLength + c);
        const elem_t av = convert_value(a[ia], d1, da);
        const elem_t bv = convert_value(b[ib], d2, da);
        sum = numeric_fma(av, bv, sum, da, negate);
      }
      result[r * ztt::kTileLength + c] = sum;
    }
  proc.ZTU.write_acc(acc, result);
}

void execute_conversion(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                        bool single, bool pack)
{
  const unsigned dest_reg = insn.rd();
  const unsigned source_reg = single ? insn.rs2() : insn.rs1();
  const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, dest_reg);
  const ztt_m_meta_t source_meta = ztt_read_m_meta(proc, source_reg);
  if (!ztt_validate_tuple(proc, opcode,
                          single ? ztt_dtype_tuple_t{
                            true, false, true, false, dest_meta.dtype, 0,
                            source_meta.dtype, 0}
                                 : unary_tuple(dest_meta.dtype,
                                               source_meta.dtype)))
    return;
  const std::size_t instruction_squares = single
    ? 0 : ztt_instruction_square_count({dest_meta, source_meta});
  const ztt_m_operand_t dest_operand = single
    ? ztt_form_m_operand(proc, insn, dest_meta)
    : ztt_form_m_operand(proc, insn, dest_meta, instruction_squares);
  const ztt_m_operand_t source_operand = single
    ? ztt_form_m_operand(proc, insn, source_meta)
    : ztt_form_m_operand(proc, insn, source_meta, instruction_squares);
  std::vector<elem_t> dest, source;
  if (!snapshot_m(proc, dest_operand, dest) ||
      !snapshot_m(proc, source_operand, source))
    return;
  const uint32_t dd = dest_meta.dtype;
  const uint32_t ds = source_meta.dtype;
  const std::size_t d_squares = ztt_unit_t::square_count(dd);
  const std::size_t s_squares = ztt_unit_t::square_count(ds);
  if (!single) {
    for (std::size_t e = 0; e < dest.size(); ++e)
      dest[e] = convert_value(source[e], ds, dd);
    commit_m(proc, dest_operand, dest);
    return;
  }

  const std::size_t slots = pack ? d_squares : s_squares;
  const std::size_t index =
    std::size_t(proc.get_state()->XPR[insn.rs1()]) & (slots - 1);
  for (std::size_t e = 0; e < ztt::kNumElements; ++e) {
    if (pack)
      dest[index * ztt::kNumElements + e] = convert_value(source[e], ds, dd);
    else
      dest[e] = convert_value(source[index * ztt::kNumElements + e], ds, dd);
  }
  commit_m(proc, dest_operand, dest);
}

void check_memory_alignment(processor_t& proc, insn_t insn, reg_t address,
                            std::size_t element_bytes, bool store)
{
  if (element_bytes != 0 && address % element_bytes != 0) {
    if (store)
      throw trap_store_address_misaligned(proc.get_state()->v, address, 0, 0);
    throw trap_load_address_misaligned(proc.get_state()->v, address, 0, 0);
  }
}

reg_t strided_address(const processor_t& proc, reg_t base, reg_t stride,
                      std::size_t segment)
{
  const reg_t address = base + reg_t(segment) * stride;
  return proc.get_xlen() == 32 ? reg_t(uint32_t(address)) : address;
}

void execute_opaque_memory(processor_t& proc, insn_t insn, bool store)
{
  const unsigned reg = insn.rd();
  if (reg >= ztt::kNumMRegisters)
    illegal(insn);
  const reg_t address = proc.get_state()->XPR[insn.rs1()];
  constexpr uint32_t unit_dtype = uint32_t(ztt::kUnitDatatypeBits);
  constexpr std::size_t bytes = ztt::kMRegisterBits / 8;
  constexpr std::size_t element_bytes = (ztt::kUnitDatatypeBits + 7) / 8;
  check_memory_alignment(proc, insn, address, element_bytes, store);

  // The 1r forms use the standardized row-major image of a unit datatype.
  // They deliberately bypass Md, but do not expose the implementation's
  // physical M-register layout.
  std::vector<elem_t> elements;
  if (!proc.ZTU.read_m_as(reg, unit_dtype, elements))
    illegal(insn);
  std::vector<uint8_t> memory(bytes, 0);
  if (store)
    for (std::size_t i = 0; i < elements.size(); ++i)
      for (unsigned bit = 0; bit < ztt::kUnitDatatypeBits; ++bit) {
        const std::size_t absolute = i * ztt::kUnitDatatypeBits + bit;
        memory[absolute / 8] |=
          uint8_t((elements[i] >> bit) & 1) << (absolute % 8);
      }
  if (store) {
    proc.get_mmu()->ztt_probe_store(address, bytes);
    proc.get_mmu()->ztt_store_bytes(address, memory.data(), bytes);
  } else {
    proc.get_mmu()->ztt_load_bytes(address, memory.data(), bytes);
    std::fill(elements.begin(), elements.end(), 0);
    for (std::size_t i = 0; i < elements.size(); ++i)
      for (unsigned bit = 0; bit < ztt::kUnitDatatypeBits; ++bit) {
        const std::size_t absolute = i * ztt::kUnitDatatypeBits + bit;
        elements[i] |= elem_t((memory[absolute / 8] >>
                               (absolute % 8)) & 1) << bit;
      }
    if (!proc.ZTU.write_m_as(reg, unit_dtype, elements))
      illegal(insn);
  }
}

void execute_memory(processor_t& proc, ztt_opcode_t opcode, insn_t insn,
                    bool store, bool column)
{
  const unsigned reg = insn.rd();
  const ztt_m_meta_t meta = ztt_read_m_meta(proc, reg);
  if (!ztt_validate_tuple(proc, opcode, one_tuple(meta.dtype)))
    return;
  const ztt_m_operand_t operand = ztt_form_m_operand(proc, insn, meta);
  const uint32_t dtype = meta.dtype;
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  const std::size_t element_count = operand.squares * ztt::kNumElements;
  const std::size_t bytes = (element_count * width + 7) / 8;
  const reg_t address = proc.get_state()->XPR[insn.rs1()];
  check_memory_alignment(proc, insn, address, (width + 7) / 8, store);
  std::vector<uint8_t> memory(bytes, 0);

  std::vector<elem_t> values;
  if (store && !snapshot_m(proc, operand, values))
    return;

  auto logical_index = [column](std::size_t index) {
    const std::size_t square = index / ztt::kNumElements;
    const std::size_t inner = index % ztt::kNumElements;
    const std::size_t row = column ? inner % ztt::kTileLength
                                   : inner / ztt::kTileLength;
    const std::size_t col = column ? inner / ztt::kTileLength
                                   : inner % ztt::kTileLength;
    return square * ztt::kNumElements + row * ztt::kTileLength + col;
  };

  if (store) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      const elem_t value = values[logical_index(i)];
      for (unsigned bit = 0; bit < width; ++bit) {
        const std::size_t absolute = i * width + bit;
        memory[absolute / 8] |= uint8_t((value >> bit) & 1) << (absolute % 8);
      }
    }
    proc.get_mmu()->ztt_probe_store(address, bytes);
    proc.get_mmu()->ztt_store_bytes(address, memory.data(), bytes);
  } else {
    proc.get_mmu()->ztt_load_bytes(address, memory.data(), bytes);
    std::vector<elem_t> result(element_count, 0);
    for (std::size_t i = 0; i < result.size(); ++i) {
      elem_t value = 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        const std::size_t absolute = i * width + bit;
        value |= elem_t((memory[absolute / 8] >> (absolute % 8)) & 1) << bit;
      }
      result[logical_index(i)] = value;
    }
    proc.ZTU.write_m(reg, result);
  }
}

void execute_strided_memory(processor_t& proc, ztt_opcode_t opcode,
                            insn_t insn, bool store, bool transpose)
{
  const unsigned reg = insn.rd();
  const ztt_m_meta_t meta = ztt_read_m_meta(proc, reg);
  if (!ztt_validate_tuple(proc, opcode, one_tuple(meta.dtype)))
    return;
  const ztt_m_operand_t operand = ztt_form_m_operand(proc, insn, meta);
  const uint32_t dtype = meta.dtype;
  const unsigned width = ztt_unit_t::datatype_bits(dtype);
  const std::size_t pack = ztt_unit_t::square_count(dtype);
  const reg_t base = proc.get_state()->XPR[insn.rs1()];
  const reg_t stride = proc.get_state()->XPR[insn.rs2()];
  check_memory_alignment(proc, insn, base, (width + 7) / 8, store);

  const std::size_t element_count = operand.squares * ztt::kNumElements;
  std::vector<elem_t> values;
  if (store) {
    if (!snapshot_m(proc, operand, values))
      return;
  } else {
    values.resize(element_count);
  }

  const std::size_t segments = transpose ? pack * ztt::kTileLength
                                         : ztt::kTileLength;
  const std::size_t elements_per_segment = transpose ? ztt::kTileLength
                                                      : pack * ztt::kTileLength;
  const std::size_t segment_bytes = (elements_per_segment * width + 7) / 8;
  std::vector<std::vector<uint8_t>> staged(
    segments, std::vector<uint8_t>(segment_bytes, 0));

  auto value_index = [transpose](std::size_t segment, std::size_t element) {
    if (transpose) {
      const std::size_t square = segment / ztt::kTileLength;
      const std::size_t col = segment % ztt::kTileLength;
      return square * ztt::kNumElements + element * ztt::kTileLength + col;
    }
    const std::size_t square = element / ztt::kTileLength;
    const std::size_t col = element % ztt::kTileLength;
    return square * ztt::kNumElements + segment * ztt::kTileLength + col;
  };

  if (!store) {
    for (std::size_t segment = 0; segment < segments; ++segment)
      proc.get_mmu()->ztt_load_bytes(strided_address(proc, base, stride, segment),
                                     staged[segment].data(), segment_bytes);
  }
  for (std::size_t segment = 0; segment < segments; ++segment)
    for (std::size_t element = 0; element < elements_per_segment; ++element) {
      elem_t value = store ? values[value_index(segment, element)] : 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        const std::size_t absolute = element * width + bit;
        if (store)
          staged[segment][absolute / 8] |=
            uint8_t((value >> bit) & 1) << (absolute % 8);
        else
          value |= elem_t((staged[segment][absolute / 8] >>
                           (absolute % 8)) & 1) << bit;
      }
      if (!store)
        values[value_index(segment, element)] = value;
    }
  if (store) {
    for (std::size_t segment = 0; segment < segments; ++segment)
      proc.get_mmu()->ztt_probe_store(
                                      strided_address(proc, base, stride, segment),
                                      segment_bytes);
    for (std::size_t segment = 0; segment < segments; ++segment)
      proc.get_mmu()->ztt_store_bytes(
                                      strided_address(proc, base, stride, segment),
                                      staged[segment].data(), segment_bytes);
  } else {
    proc.ZTU.write_m(reg, values);
  }
}

} // namespace

bool ztt_opcode_may_modify_state(ztt_opcode_t opcode)
{
  switch (opcode) {
    // These instructions cannot change any AME register, datatype, or
    // writable context CSR.  In particular, mss.1r does not inspect Md.
    case ztt_opcode_t::ame_acquire:
    case ztt_opcode_t::ame_release:
    case ztt_opcode_t::agettyp:
    case ztt_opcode_t::mgettyp:
    case ztt_opcode_t::mss_1r:
    case ztt_opcode_t::mmove8_x_m:
    case ztt_opcode_t::mmove16_x_m:
    case ztt_opcode_t::mmove32_x_m:
    case ztt_opcode_t::mmove64_x_m:
      return false;

    // The interoperability and strided stores are included here because an
    // unsupported datatype tuple sets amestatus.UN even though a supported
    // store only writes memory.  The specification permits conservative
    // Dirty updates for the remaining state-capable instructions.
    case ztt_opcode_t::asettyp:
    case ztt_opcode_t::mabs_ew:
    case ztt_opcode_t::mabsdiff_ew:
    case ztt_opcode_t::mabsdiff_ew_x:
    case ztt_opcode_t::madd_ew:
    case ztt_opcode_t::madd_ew_x:
    case ztt_opcode_t::mand_ew:
    case ztt_opcode_t::mand_ew_x:
    case ztt_opcode_t::mandnot_ew:
    case ztt_opcode_t::mandnot_ew_x:
    case ztt_opcode_t::mcmovge_ew:
    case ztt_opcode_t::mcmovlt_ew:
    case ztt_opcode_t::mcmpge_ew:
    case ztt_opcode_t::mcmpge_ew_x:
    case ztt_opcode_t::mcmplt_ew:
    case ztt_opcode_t::mcmplt_ew_x:
    case ztt_opcode_t::mcolbcast_ew_x:
    case ztt_opcode_t::mcolgather_ew:
    case ztt_opcode_t::mcolid_ew:
    case ztt_opcode_t::mcolshift_ew_x:
    case ztt_opcode_t::mcolunzip_ew:
    case ztt_opcode_t::mcolzip_ew:
    case ztt_opcode_t::mconv_ew:
    case ztt_opcode_t::mbcast_m_x:
    case ztt_opcode_t::mcos_ew:
    case ztt_opcode_t::mexp2_ew:
    case ztt_opcode_t::mfrintm_ew:
    case ztt_opcode_t::mfrintn_ew:
    case ztt_opcode_t::mfrintp_ew:
    case ztt_opcode_t::mfrintz_ew:
    case ztt_opcode_t::mhdiff_ew:
    case ztt_opcode_t::mhdiff_ew_x:
    case ztt_opcode_t::mldexp_ew:
    case ztt_opcode_t::mldexp_ew_x:
    case ztt_opcode_t::mldexpacc_ew:
    case ztt_opcode_t::mldexpacc_ew_x:
    case ztt_opcode_t::mlog2_ew:
    case ztt_opcode_t::mlog2sub_ew:
    case ztt_opcode_t::mlog2sub_ew_x:
    case ztt_opcode_t::mls_1r:
    case ztt_opcode_t::mls_cm:
    case ztt_opcode_t::mls_rm:
    case ztt_opcode_t::mls_st:
    case ztt_opcode_t::mls_tst:
    case ztt_opcode_t::mmax_ew:
    case ztt_opcode_t::mmax_ew_x:
    case ztt_opcode_t::mmean_ew:
    case ztt_opcode_t::mmean_ew_x:
    case ztt_opcode_t::mmin_ew:
    case ztt_opcode_t::mmin_ew_x:
    case ztt_opcode_t::mmov_m_a:
    case ztt_opcode_t::mmov_a_m:
    case ztt_opcode_t::mmov_m_m:
    case ztt_opcode_t::mmove8_m_x:
    case ztt_opcode_t::mmove16_m_x:
    case ztt_opcode_t::mmove32_m_x:
    case ztt_opcode_t::mmove64_m_x:
    case ztt_opcode_t::mmul_ew:
    case ztt_opcode_t::mmul_ew_x:
    case ztt_opcode_t::mmulacc_2d:
    case ztt_opcode_t::mmulacc_ew:
    case ztt_opcode_t::mmulacc_ew_x:
    case ztt_opcode_t::mmulaccneg_2d:
    case ztt_opcode_t::mmulaccneg_ew:
    case ztt_opcode_t::mmulaccneg_ew_x:
    case ztt_opcode_t::mmuladd_ew:
    case ztt_opcode_t::mmuladd_ew_x:
    case ztt_opcode_t::mmulatacc_2d:
    case ztt_opcode_t::mmulataccneg_2d:
    case ztt_opcode_t::mmulbtacc_2d:
    case ztt_opcode_t::mmulbtaccneg_2d:
    case ztt_opcode_t::mmulneg_ew:
    case ztt_opcode_t::mmulneg_ew_x:
    case ztt_opcode_t::mmulsub_ew:
    case ztt_opcode_t::mmulsub_ew_x:
    case ztt_opcode_t::mor_ew:
    case ztt_opcode_t::mor_ew_x:
    case ztt_opcode_t::mornot_ew:
    case ztt_opcode_t::mornot_ew_x:
    case ztt_opcode_t::mpack_ew_x:
    case ztt_opcode_t::mprefixadd_col:
    case ztt_opcode_t::mprefixadd_row:
    case ztt_opcode_t::mprefixmax_col:
    case ztt_opcode_t::mprefixmax_row:
    case ztt_opcode_t::mrdexp_ew:
    case ztt_opcode_t::mrdexpacc_ew:
    case ztt_opcode_t::mrec_ew:
    case ztt_opcode_t::mreduceadd_col:
    case ztt_opcode_t::mreduceadd_row:
    case ztt_opcode_t::mreducemax_col:
    case ztt_opcode_t::mreducemax_row:
    case ztt_opcode_t::mreducemin_col:
    case ztt_opcode_t::mreducemin_row:
    case ztt_opcode_t::mrowbcast_ew_x:
    case ztt_opcode_t::mrowgather_ew:
    case ztt_opcode_t::mrowid_ew:
    case ztt_opcode_t::mrowshift_ew_x:
    case ztt_opcode_t::mrowunzip_ew:
    case ztt_opcode_t::mrowzip_ew:
    case ztt_opcode_t::mrsqrt_ew:
    case ztt_opcode_t::mrowscatadd_ew:
    case ztt_opcode_t::mcolscatadd_ew:
    case ztt_opcode_t::mrowscatmax_ew:
    case ztt_opcode_t::mcolscatmax_ew:
    case ztt_opcode_t::mselge_ew:
    case ztt_opcode_t::msellt_ew:
    case ztt_opcode_t::msettyp:
    case ztt_opcode_t::msin_ew:
    case ztt_opcode_t::msll_ew:
    case ztt_opcode_t::msll_ew_x:
    case ztt_opcode_t::msqrt_ew:
    case ztt_opcode_t::msra_ew:
    case ztt_opcode_t::msra_ew_x:
    case ztt_opcode_t::msrl_ew:
    case ztt_opcode_t::msrl_ew_x:
    case ztt_opcode_t::mss_cm:
    case ztt_opcode_t::mss_rm:
    case ztt_opcode_t::mss_st:
    case ztt_opcode_t::mss_tst:
    case ztt_opcode_t::msub_ew:
    case ztt_opcode_t::msub_ew_x:
    case ztt_opcode_t::msublog2_ew:
    case ztt_opcode_t::msublog2_ew_x:
    case ztt_opcode_t::mtanh_ew:
    case ztt_opcode_t::munpack_ew_x:
    case ztt_opcode_t::mxor_ew:
    case ztt_opcode_t::mxor_ew_x:
    case ztt_opcode_t::mzero_2d_acc:
    case ztt_opcode_t::mzero_2d_m:
      return true;
  }
  return true;
}

void execute_ztt(processor_t& proc, ztt_opcode_t opcode, insn_t insn)
{
  if (!ztt_ame_state_enabled(proc))
    illegal(insn);

  if (opcode == ztt_opcode_t::ame_acquire) {
    ztt_validate_encoded_fields(proc, opcode, insn);
    const reg_t descriptor = proc.get_state()->XPR[insn.rs1()];
    if (!valid_acquire_descriptor(descriptor)) {
      proc.get_state()->XPR.write(insn.rd(), reg_t(0x04) << 1);
      return;
    }
    if (!proc.ZTU.owned()) {
      proc.ZTU.reset();
      proc.ZTU.acquire();
      ztt_set_state_initial(proc);
    }
    proc.get_state()->XPR.write(insn.rd(), 1);
    return;
  }

  if (opcode == ztt_opcode_t::ame_release) {
    ztt_validate_encoded_fields(proc, opcode, insn);
    if (proc.ZTU.owned()) {
      proc.ZTU.release();
      ztt_set_state_initial(proc);
    }
    return;
  }

  if (!proc.ZTU.owned())
    illegal(insn);

  ztt_validate_encoded_fields(proc, opcode, insn);

  dirty_on_success_t dirty(proc, ztt_opcode_may_modify_state(opcode));
  ame_flags_on_success_t flags(proc);

  switch (opcode) {
    case ztt_opcode_t::agettyp:
      if (ztt_acc_1516(insn) >= ztt::kNumAccRegisters)
        illegal(insn);
      proc.get_state()->XPR.write(insn.rd(), proc.ZTU.acc_datatype(ztt_acc_1516(insn)));
      return;
    case ztt_opcode_t::asettyp: {
      if (ztt_acc_98(insn) >= ztt::kNumAccRegisters)
        illegal(insn);
      const uint32_t dtype = uint32_t(proc.get_state()->XPR[insn.rs1()]);
      if (!ztt_validate_tuple(proc, opcode, one_tuple(dtype)))
        return;
      proc.ZTU.set_acc_datatype(ztt_acc_98(insn), dtype);
      return;
    }
    case ztt_opcode_t::mgettyp:
      proc.get_state()->XPR.write(insn.rd(), proc.ZTU.m_datatype(insn.rs1()));
      return;
    case ztt_opcode_t::msettyp: {
      const uint32_t dtype = uint32_t(proc.get_state()->XPR[insn.rs1()]);
      if (!ztt_validate_tuple(proc, opcode, one_tuple(dtype)))
        return;
      if (!proc.ZTU.valid_m_group_geometry(insn.rd(), dtype))
        illegal(insn);
      proc.ZTU.set_m_datatype(insn.rd(), dtype);
      return;
    }
    case ztt_opcode_t::mbcast_m_x:
    {
      const ztt_m_meta_t dest_meta = ztt_read_m_meta(proc, insn.rd());
      const uint32_t source_dtype = uint32_t(proc.get_state()->XPR[insn.rs2()]);
      ztt_dtype_tuple_t tuple = one_tuple(dest_meta.dtype);
      tuple.has_scalar = true;
      tuple.scalar = source_dtype;
      if (!ztt_validate_tuple(proc, opcode, tuple))
        return;
      ztt_form_m_operand(proc, insn, dest_meta);
      elem_t scalar = proc.get_state()->XPR[insn.rs1()];
      const unsigned source_bits = ztt_unit_t::datatype_bits(source_dtype);
      if (source_bits < proc.get_xlen())
        scalar &= ztt_unit_t::element_mask(source_bits);
      proc.ZTU.broadcast_x(insn.rd(),
                           convert_value(scalar, source_dtype, dest_meta.dtype));
      return;
    }
    case ztt_opcode_t::mmov_m_m: {
      const ztt_m_meta_t dest = ztt_read_m_meta(proc, insn.rd());
      const ztt_m_meta_t source = ztt_read_m_meta(proc, insn.rs1());
      if (!ztt_validate_tuple(proc, opcode,
                              unary_tuple(dest.dtype, source.dtype)))
        return;
      const auto operands = form_m_operands(proc, insn, {dest, source});
      std::vector<elem_t> values;
      if (snapshot_m(proc, operands[1], values))
        commit_m(proc, operands[0], values);
      return;
    }
    case ztt_opcode_t::mmov_m_a: execute_acc_move(proc, opcode, insn, false); return;
    case ztt_opcode_t::mmov_a_m: execute_acc_move(proc, opcode, insn, true); return;
    case ztt_opcode_t::mmove8_m_x: execute_debug_move(proc, insn, 8, true); return;
    case ztt_opcode_t::mmove16_m_x: execute_debug_move(proc, insn, 16, true); return;
    case ztt_opcode_t::mmove32_m_x: execute_debug_move(proc, insn, 32, true); return;
    case ztt_opcode_t::mmove64_m_x: execute_debug_move(proc, insn, 64, true); return;
    case ztt_opcode_t::mmove8_x_m: execute_debug_move(proc, insn, 8, false); return;
    case ztt_opcode_t::mmove16_x_m: execute_debug_move(proc, insn, 16, false); return;
    case ztt_opcode_t::mmove32_x_m: execute_debug_move(proc, insn, 32, false); return;
    case ztt_opcode_t::mmove64_x_m: execute_debug_move(proc, insn, 64, false); return;
    case ztt_opcode_t::mzero_2d_acc: {
      const ztt_acc_meta_t acc = ztt_read_acc_meta(proc, ztt_acc_98(insn));
      if (!ztt_validate_tuple(proc, opcode, one_tuple(acc.dtype)))
        return;
      proc.ZTU.zero_acc(ztt_acc_98(insn));
      return;
    }
    case ztt_opcode_t::mzero_2d_m: {
      const ztt_m_meta_t dest = ztt_read_m_meta(proc, insn.rd());
      if (!ztt_validate_tuple(proc, opcode, one_tuple(dest.dtype)))
        return;
      const ztt_m_operand_t operand = ztt_form_m_operand(proc, insn, dest);
      std::vector<elem_t> values;
      if (snapshot_m(proc, operand, values)) {
        std::fill(values.begin(), values.end(), 0);
        commit_m(proc, operand, values);
      }
      return;
    }

    case ztt_opcode_t::mabs_ew: execute_elementwise(proc, opcode, insn, ew_op::abs, false, true); return;
    case ztt_opcode_t::mcos_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::cos); return;
    case ztt_opcode_t::mfrintm_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::frintm); return;
    case ztt_opcode_t::mfrintn_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::frintn); return;
    case ztt_opcode_t::mfrintp_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::frintp); return;
    case ztt_opcode_t::mfrintz_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::frintz); return;
    case ztt_opcode_t::mrec_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::rec); return;
    case ztt_opcode_t::mrsqrt_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::rsqrt); return;
    case ztt_opcode_t::msin_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::sin); return;
    case ztt_opcode_t::msqrt_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::sqrt); return;
    case ztt_opcode_t::mtanh_ew: execute_fp_unary(proc, opcode, insn, fp_unary_op::tanh); return;
    case ztt_opcode_t::mabsdiff_ew: execute_elementwise(proc, opcode, insn, ew_op::absdiff, false); return;
    case ztt_opcode_t::mabsdiff_ew_x: execute_elementwise(proc, opcode, insn, ew_op::absdiff, true); return;
    case ztt_opcode_t::madd_ew: execute_elementwise(proc, opcode, insn, ew_op::add, false); return;
    case ztt_opcode_t::madd_ew_x: execute_elementwise(proc, opcode, insn, ew_op::add, true); return;
    case ztt_opcode_t::mand_ew: execute_elementwise(proc, opcode, insn, ew_op::bit_and, false); return;
    case ztt_opcode_t::mand_ew_x: execute_elementwise(proc, opcode, insn, ew_op::bit_and, true); return;
    case ztt_opcode_t::mandnot_ew: execute_elementwise(proc, opcode, insn, ew_op::bit_andnot, false); return;
    case ztt_opcode_t::mandnot_ew_x: execute_elementwise(proc, opcode, insn, ew_op::bit_andnot, true); return;
    case ztt_opcode_t::mcmpge_ew: execute_elementwise(proc, opcode, insn, ew_op::cmpge, false); return;
    case ztt_opcode_t::mcmpge_ew_x: execute_elementwise(proc, opcode, insn, ew_op::cmpge, true); return;
    case ztt_opcode_t::mcmplt_ew: execute_elementwise(proc, opcode, insn, ew_op::cmplt, false); return;
    case ztt_opcode_t::mcmplt_ew_x: execute_elementwise(proc, opcode, insn, ew_op::cmplt, true); return;
    case ztt_opcode_t::mexp2_ew: execute_elementwise(proc, opcode, insn, ew_op::exp2, false, true); return;
    case ztt_opcode_t::mhdiff_ew: execute_elementwise(proc, opcode, insn, ew_op::hdiff, false); return;
    case ztt_opcode_t::mhdiff_ew_x: execute_elementwise(proc, opcode, insn, ew_op::hdiff_scalar, true); return;
    case ztt_opcode_t::mldexp_ew: execute_elementwise(proc, opcode, insn, ew_op::ldexp, false); return;
    case ztt_opcode_t::mldexp_ew_x: execute_elementwise(proc, opcode, insn, ew_op::ldexp_scalar, true, false, true); return;
    case ztt_opcode_t::mldexpacc_ew: execute_elementwise(proc, opcode, insn, ew_op::ldexpacc, false); return;
    case ztt_opcode_t::mldexpacc_ew_x: execute_elementwise(proc, opcode, insn, ew_op::ldexpacc_scalar, true, false, true); return;
    case ztt_opcode_t::mlog2_ew: execute_elementwise(proc, opcode, insn, ew_op::log2, false, true); return;
    case ztt_opcode_t::mlog2sub_ew: execute_elementwise(proc, opcode, insn, ew_op::log2sub, false); return;
    case ztt_opcode_t::mlog2sub_ew_x: execute_elementwise(proc, opcode, insn, ew_op::log2sub_scalar, true); return;
    case ztt_opcode_t::mmax_ew: execute_elementwise(proc, opcode, insn, ew_op::max, false); return;
    case ztt_opcode_t::mmax_ew_x: execute_elementwise(proc, opcode, insn, ew_op::max, true); return;
    case ztt_opcode_t::mmean_ew: execute_elementwise(proc, opcode, insn, ew_op::mean, false); return;
    case ztt_opcode_t::mmean_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mean, true); return;
    case ztt_opcode_t::mmin_ew: execute_elementwise(proc, opcode, insn, ew_op::min, false); return;
    case ztt_opcode_t::mmin_ew_x: execute_elementwise(proc, opcode, insn, ew_op::min, true); return;
    case ztt_opcode_t::mmul_ew: execute_elementwise(proc, opcode, insn, ew_op::mul, false); return;
    case ztt_opcode_t::mmul_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mul, true); return;
    case ztt_opcode_t::mmulacc_ew: execute_elementwise(proc, opcode, insn, ew_op::mulacc, false); return;
    case ztt_opcode_t::mmulacc_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mulacc, true); return;
    case ztt_opcode_t::mmulaccneg_ew: execute_elementwise(proc, opcode, insn, ew_op::mulaccneg, false); return;
    case ztt_opcode_t::mmulaccneg_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mulaccneg, true); return;
    case ztt_opcode_t::mmuladd_ew: execute_elementwise(proc, opcode, insn, ew_op::muladd, false); return;
    case ztt_opcode_t::mmuladd_ew_x: execute_elementwise(proc, opcode, insn, ew_op::muladd, true); return;
    case ztt_opcode_t::mmulneg_ew: execute_elementwise(proc, opcode, insn, ew_op::mulneg, false); return;
    case ztt_opcode_t::mmulneg_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mulneg, true); return;
    case ztt_opcode_t::mmulsub_ew: execute_elementwise(proc, opcode, insn, ew_op::mulsub, false); return;
    case ztt_opcode_t::mmulsub_ew_x: execute_elementwise(proc, opcode, insn, ew_op::mulsub, true); return;
    case ztt_opcode_t::mor_ew: execute_elementwise(proc, opcode, insn, ew_op::bit_or, false); return;
    case ztt_opcode_t::mor_ew_x: execute_elementwise(proc, opcode, insn, ew_op::bit_or, true); return;
    case ztt_opcode_t::mornot_ew: execute_elementwise(proc, opcode, insn, ew_op::bit_ornot, false); return;
    case ztt_opcode_t::mornot_ew_x: execute_elementwise(proc, opcode, insn, ew_op::bit_ornot, true); return;
    case ztt_opcode_t::mrdexp_ew: execute_elementwise(proc, opcode, insn, ew_op::rdexp, false); return;
    case ztt_opcode_t::mrdexpacc_ew: execute_elementwise(proc, opcode, insn, ew_op::rdexpacc, false); return;
    case ztt_opcode_t::msub_ew: execute_elementwise(proc, opcode, insn, ew_op::sub, false); return;
    case ztt_opcode_t::msub_ew_x: execute_elementwise(proc, opcode, insn, ew_op::sub_scalar, true); return;
    case ztt_opcode_t::msublog2_ew: execute_elementwise(proc, opcode, insn, ew_op::sublog2, false); return;
    case ztt_opcode_t::msublog2_ew_x: execute_elementwise(proc, opcode, insn, ew_op::sublog2_scalar, true); return;
    case ztt_opcode_t::mxor_ew: execute_elementwise(proc, opcode, insn, ew_op::bit_xor, false); return;
    case ztt_opcode_t::mxor_ew_x: execute_elementwise(proc, opcode, insn, ew_op::bit_xor, true); return;

    case ztt_opcode_t::mcmovge_ew: execute_conditional(proc, opcode, insn, false, true); return;
    case ztt_opcode_t::mcmovlt_ew: execute_conditional(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::mselge_ew: execute_conditional(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::msellt_ew: execute_conditional(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mcolbcast_ew_x: execute_axis_control(proc, opcode, insn, false, true); return;
    case ztt_opcode_t::mrowbcast_ew_x: execute_axis_control(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::mcolshift_ew_x: execute_axis_control(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mrowshift_ew_x: execute_axis_control(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mcolid_ew: execute_axis_id(proc, opcode, insn, false); return;
    case ztt_opcode_t::mrowid_ew: execute_axis_id(proc, opcode, insn, true); return;
    case ztt_opcode_t::mcolunzip_ew: execute_zip_pair(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mcolzip_ew: execute_zip_pair(proc, opcode, insn, false, true); return;
    case ztt_opcode_t::mrowunzip_ew: execute_zip_pair(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mrowzip_ew: execute_zip_pair(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::mcolgather_ew: execute_gather(proc, opcode, insn, true); return;
    case ztt_opcode_t::mrowgather_ew: execute_gather(proc, opcode, insn, false); return;

    case ztt_opcode_t::mprefixadd_col: execute_prefix_reduce(proc, opcode, insn, true, false, false); return;
    case ztt_opcode_t::mprefixadd_row: execute_prefix_reduce(proc, opcode, insn, false, false, false); return;
    case ztt_opcode_t::mprefixmax_col: execute_prefix_reduce(proc, opcode, insn, true, true, false); return;
    case ztt_opcode_t::mprefixmax_row: execute_prefix_reduce(proc, opcode, insn, false, true, false); return;
    case ztt_opcode_t::mreduceadd_col: execute_prefix_reduce(proc, opcode, insn, true, false, true); return;
    case ztt_opcode_t::mreduceadd_row: execute_prefix_reduce(proc, opcode, insn, false, false, true); return;
    case ztt_opcode_t::mreducemax_col: execute_prefix_reduce(proc, opcode, insn, true, true, true); return;
    case ztt_opcode_t::mreducemax_row: execute_prefix_reduce(proc, opcode, insn, false, true, true); return;
    case ztt_opcode_t::mreducemin_col: execute_prefix_reduce(proc, opcode, insn, true, false, true, true); return;
    case ztt_opcode_t::mreducemin_row: execute_prefix_reduce(proc, opcode, insn, false, false, true, true); return;
    case ztt_opcode_t::mcolscatadd_ew: execute_gather_scatter(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mrowscatadd_ew: execute_gather_scatter(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mcolscatmax_ew: execute_gather_scatter(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::mrowscatmax_ew: execute_gather_scatter(proc, opcode, insn, false, true); return;

    case ztt_opcode_t::mmulacc_2d: execute_matrix(proc, opcode, insn, false, false, true, false); return;
    case ztt_opcode_t::mmulaccneg_2d: execute_matrix(proc, opcode, insn, false, false, true, true); return;
    case ztt_opcode_t::mmulatacc_2d: execute_matrix(proc, opcode, insn, true, false, true, false); return;
    case ztt_opcode_t::mmulataccneg_2d: execute_matrix(proc, opcode, insn, true, false, true, true); return;
    case ztt_opcode_t::mmulbtacc_2d: execute_matrix(proc, opcode, insn, false, true, true, false); return;
    case ztt_opcode_t::mmulbtaccneg_2d: execute_matrix(proc, opcode, insn, false, true, true, true); return;

    case ztt_opcode_t::mconv_ew: execute_conversion(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mpack_ew_x: execute_conversion(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::munpack_ew_x: execute_conversion(proc, opcode, insn, true, false); return;

    case ztt_opcode_t::msll_ew: execute_shift(proc, opcode, insn, shift_op::left, false); return;
    case ztt_opcode_t::msll_ew_x: execute_shift(proc, opcode, insn, shift_op::left, true); return;
    case ztt_opcode_t::msra_ew: execute_shift(proc, opcode, insn, shift_op::arithmetic_right, false); return;
    case ztt_opcode_t::msra_ew_x: execute_shift(proc, opcode, insn, shift_op::arithmetic_right, true); return;
    case ztt_opcode_t::msrl_ew: execute_shift(proc, opcode, insn, shift_op::logical_right, false); return;
    case ztt_opcode_t::msrl_ew_x: execute_shift(proc, opcode, insn, shift_op::logical_right, true); return;

    case ztt_opcode_t::mls_1r: execute_opaque_memory(proc, insn, false); return;
    case ztt_opcode_t::mls_rm: execute_memory(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mls_cm: execute_memory(proc, opcode, insn, false, true); return;
    case ztt_opcode_t::mls_st: execute_strided_memory(proc, opcode, insn, false, false); return;
    case ztt_opcode_t::mls_tst: execute_strided_memory(proc, opcode, insn, false, true); return;
    case ztt_opcode_t::mss_1r: execute_opaque_memory(proc, insn, true); return;
    case ztt_opcode_t::mss_rm: execute_memory(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mss_cm: execute_memory(proc, opcode, insn, true, true); return;
    case ztt_opcode_t::mss_st: execute_strided_memory(proc, opcode, insn, true, false); return;
    case ztt_opcode_t::mss_tst: execute_strided_memory(proc, opcode, insn, true, true); return;

    // Handled before ownership and flag setup above.
    case ztt_opcode_t::ame_acquire:
    case ztt_opcode_t::ame_release:
      break;
  }
  illegal(insn);
}
