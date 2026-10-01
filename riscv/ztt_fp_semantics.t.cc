// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "encoding.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "softfloat.h"
#include "ztt_execute.h"
#include "ztt_fp.h"
#include "ztt_fp_transcendental.h"
#include "ztt_state.h"
#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cstdint>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;

constexpr uint32_t fp32(unsigned rounding)
{
  return (8u << 26) | (rounding << 22) | (1u << 21) | (1u << 20) |
         (1u << 8) | 32u;
}

constexpr uint32_t standard_float(unsigned bits, unsigned exponent,
                                  unsigned rounding)
{
  return (exponent << 26) | (rounding << 22) | (1u << 21) |
         (1u << 20) | (1u << 8) | bits;
}

constexpr uint32_t fp16(unsigned rounding)
{
  return standard_float(16, 5, rounding);
}

constexpr uint32_t bf16(unsigned rounding)
{
  return standard_float(16, 8, rounding);
}

constexpr uint32_t fp64(unsigned rounding)
{
  return standard_float(64, 11, rounding);
}

constexpr uint32_t fp8e4m3(unsigned rounding)
{
  return (4u << 26) | (rounding << 22) | (1u << 20) | (1u << 8) | 8u;
}

constexpr uint32_t fp8e5m2(unsigned rounding)
{
  return standard_float(8, 5, rounding);
}

constexpr uint32_t kI32 = (1u << 30) | 32u;
constexpr uint32_t kI16 = (1u << 30) | 16u;
constexpr uint32_t kU64 = 64u;

class test_sim_t final : public simif_t {
 public:
  test_sim_t() { debug_mmu = nullptr; }
  char* addr_to_mem(reg_t) override { return nullptr; }
  bool mmio_load(reg_t, size_t, uint8_t*) override { return false; }
  bool mmio_store(reg_t, size_t, const uint8_t*) override { return false; }
  void proc_reset(unsigned) override {}
  const cfg_t& get_cfg() const override { return cfg; }
  const std::map<size_t, processor_t*>& get_harts() const override { return harts; }
  const char* get_symbol(uint64_t) override { return nullptr; }

  cfg_t cfg;
  std::map<size_t, processor_t*> harts;
};

insn_t operands(unsigned rd, unsigned rs1 = 0, unsigned rs2 = 0)
{
  return insn_t((insn_bits_t(rs2) << 20) |
                (insn_bits_t(rs1) << 15) |
                (insn_bits_t(rd) << 7));
}

class fixture_t {
 public:
  fixture_t() : proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output)
  {
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(reg_t(1) << 25);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1));
  }

  void set_m(unsigned reg, uint32_t dtype, elem_t value)
  {
    assert(proc.ZTU.set_m_datatype(reg, dtype));
    assert(proc.ZTU.write_m(reg,
      std::vector<elem_t>(ztt_unit_t::element_count(dtype), value)));
  }

  void expect_m(unsigned reg, elem_t expected)
  {
    std::vector<elem_t> values;
    assert(proc.ZTU.read_m(reg, values));
    for (elem_t value : values)
      assert(value == expected);
  }

  void set_m_span(unsigned reg, uint32_t dtype, elem_t value,
                  std::size_t instruction_squares)
  {
    const std::size_t native_squares = ztt_unit_t::square_count(dtype);
    const std::size_t groups = instruction_squares / native_squares;
    const std::size_t group_registers = ztt_unit_t::group_registers(dtype);
    assert(instruction_squares != 0 &&
           instruction_squares % native_squares == 0);
    for (std::size_t group = 0; group < groups; ++group)
      set_m(reg + group * group_registers, dtype, value);
  }

  void expect_m_span(unsigned reg, uint32_t dtype, elem_t expected,
                     std::size_t instruction_squares)
  {
    const std::size_t native_squares = ztt_unit_t::square_count(dtype);
    const std::size_t groups = instruction_squares / native_squares;
    const std::size_t group_registers = ztt_unit_t::group_registers(dtype);
    assert(instruction_squares != 0 &&
           instruction_squares % native_squares == 0);
    for (std::size_t group = 0; group < groups; ++group)
      expect_m(reg + group * group_registers, expected);
  }

  void set_acc(unsigned reg, uint32_t dtype, elem_t value)
  {
    assert(proc.ZTU.set_acc_datatype(reg, dtype));
    assert(proc.ZTU.write_acc(reg,
      std::vector<elem_t>(ztt::kNumElements, value)));
  }

  std::vector<elem_t> get_acc(unsigned reg)
  {
    std::vector<elem_t> values;
    assert(proc.ZTU.read_acc(reg, values));
    return values;
  }

  test_sim_t sim;
  std::ostringstream output;
  processor_t proc;
};

class softfloat_reference_scope_t {
 public:
  explicit softfloat_reference_scope_t(uint32_t dtype)
    : old_round(softfloat_roundingMode),
      old_flags(softfloat_exceptionFlags),
      old_tininess(softfloat_detectTininess)
  {
    softfloat_roundingMode = ztt_to_softfloat_round(dtype);
    softfloat_exceptionFlags = 0;
    softfloat_detectTininess = softfloat_tininess_afterRounding;
  }

  ~softfloat_reference_scope_t()
  {
    softfloat_roundingMode = old_round;
    softfloat_exceptionFlags = old_flags;
    softfloat_detectTininess = old_tininess;
  }

  uint8_t flags() const
  {
    return uint8_t(softfloat_exceptionFlags & ztt::kAmeFlagMask);
  }

 private:
  uint_fast8_t old_round;
  uint_fast8_t old_flags;
  uint_fast8_t old_tininess;
};

ztt_fp_result_t softfloat_binary_reference(uint32_t dtype, uint64_t lhs,
                                           uint64_t rhs, char op)
{
  softfloat_reference_scope_t scope(dtype);
  uint64_t bits = 0;
  switch (ztt_unit_t::datatype_float_kind(dtype)) {
    case ztt_float_kind_t::f16: {
      const float16_t a{uint16_t(lhs)}, b{uint16_t(rhs)};
      bits = (op == '+' ? f16_add(a, b) : op == '-' ? f16_sub(a, b)
                                                   : f16_mul(a, b)).v;
      break;
    }
    case ztt_float_kind_t::f32: {
      const float32_t a{uint32_t(lhs)}, b{uint32_t(rhs)};
      bits = (op == '+' ? f32_add(a, b) : op == '-' ? f32_sub(a, b)
                                                   : f32_mul(a, b)).v;
      break;
    }
    case ztt_float_kind_t::f64: {
      const float64_t a{lhs}, b{rhs};
      bits = (op == '+' ? f64_add(a, b) : op == '-' ? f64_sub(a, b)
                                                   : f64_mul(a, b)).v;
      break;
    }
    default: assert(false);
  }
  return {bits, scope.flags()};
}

ztt_fp_result_t softfloat_fma_reference(uint32_t dtype, uint64_t factor1,
                                        uint64_t factor2, uint64_t addend)
{
  softfloat_reference_scope_t scope(dtype);
  uint64_t bits = 0;
  switch (ztt_unit_t::datatype_float_kind(dtype)) {
    case ztt_float_kind_t::f16:
      bits = f16_mulAdd(float16_t{uint16_t(factor1)},
                        float16_t{uint16_t(factor2)},
                        float16_t{uint16_t(addend)}).v;
      break;
    case ztt_float_kind_t::f32:
      bits = f32_mulAdd(float32_t{uint32_t(factor1)},
                        float32_t{uint32_t(factor2)},
                        float32_t{uint32_t(addend)}).v;
      break;
    case ztt_float_kind_t::f64:
      bits = f64_mulAdd(float64_t{factor1}, float64_t{factor2},
                        float64_t{addend}).v;
      break;
    default: assert(false);
  }
  return {bits, scope.flags()};
}

ztt_fp_result_t softfloat_sqrt_reference(uint32_t dtype, uint64_t input)
{
  softfloat_reference_scope_t scope(dtype);
  uint64_t bits = 0;
  switch (ztt_unit_t::datatype_float_kind(dtype)) {
    case ztt_float_kind_t::f16:
      bits = f16_sqrt(float16_t{uint16_t(input)}).v;
      break;
    case ztt_float_kind_t::f32:
      bits = f32_sqrt(float32_t{uint32_t(input)}).v;
      break;
    case ztt_float_kind_t::f64:
      bits = f64_sqrt(float64_t{input}).v;
      break;
    default: assert(false);
  }
  return {bits, scope.flags()};
}

ztt_fp_result_t softfloat_convert_reference(uint32_t source_dtype,
                                            uint32_t dest_dtype,
                                            uint64_t input)
{
  softfloat_reference_scope_t scope(dest_dtype);
  const ztt_float_kind_t source =
    ztt_unit_t::datatype_float_kind(source_dtype);
  const ztt_float_kind_t dest = ztt_unit_t::datatype_float_kind(dest_dtype);
  uint64_t bits = 0;
  if (source == ztt_float_kind_t::f16 && dest == ztt_float_kind_t::f32)
    bits = f16_to_f32(float16_t{uint16_t(input)}).v;
  else if (source == ztt_float_kind_t::f16 && dest == ztt_float_kind_t::f64)
    bits = f16_to_f64(float16_t{uint16_t(input)}).v;
  else if (source == ztt_float_kind_t::f32 && dest == ztt_float_kind_t::f16)
    bits = f32_to_f16(float32_t{uint32_t(input)}).v;
  else if (source == ztt_float_kind_t::f32 && dest == ztt_float_kind_t::f64)
    bits = f32_to_f64(float32_t{uint32_t(input)}).v;
  else if (source == ztt_float_kind_t::f64 && dest == ztt_float_kind_t::f16)
    bits = f64_to_f16(float64_t{input}).v;
  else if (source == ztt_float_kind_t::f64 && dest == ztt_float_kind_t::f32)
    bits = f64_to_f32(float64_t{input}).v;
  else
    assert(false);
  return {bits, scope.flags()};
}

ztt_fp_result_t softfloat_integer_to_float_reference(uint32_t dest_dtype,
                                                     int64_t input)
{
  softfloat_reference_scope_t scope(dest_dtype);
  uint64_t bits = 0;
  switch (ztt_unit_t::datatype_float_kind(dest_dtype)) {
    case ztt_float_kind_t::f16: bits = i64_to_f16(input).v; break;
    case ztt_float_kind_t::f32: bits = i64_to_f32(input).v; break;
    case ztt_float_kind_t::f64: bits = i64_to_f64(input).v; break;
    default: assert(false);
  }
  return {bits, scope.flags()};
}

ztt_fp_result_t softfloat_round_to_int_reference(uint32_t dtype,
                                                 uint64_t input,
                                                 uint_fast8_t rounding)
{
  softfloat_reference_scope_t scope(dtype);
  uint64_t bits = 0;
  switch (ztt_unit_t::datatype_float_kind(dtype)) {
    case ztt_float_kind_t::f16:
      bits = f16_roundToInt(float16_t{uint16_t(input)}, rounding, true).v;
      break;
    case ztt_float_kind_t::f32:
      bits = f32_roundToInt(float32_t{uint32_t(input)}, rounding, true).v;
      break;
    case ztt_float_kind_t::f64:
      bits = f64_roundToInt(float64_t{input}, rounding, true).v;
      break;
    default: assert(false);
  }
  return {bits, scope.flags()};
}

void expect_fp_unary(ztt_opcode_t opcode, uint32_t dtype, elem_t input,
                     elem_t expected, uint8_t flags)
{
  fixture_t f;
  // Register 2 and 4 satisfy FP64's two-register alignment requirement and
  // are also valid for the narrower formats.
  f.set_m(2, dtype, UINT64_C(0xdeadbeefdeadbeef));
  f.set_m(4, dtype, input);
  execute_ztt(f.proc, opcode, operands(2, 4));
  f.expect_m(2, expected);
  assert(f.proc.ZTU.amefflags() == flags);
}

void expect_mcos(uint32_t dtype, elem_t input, elem_t expected, uint8_t flags)
{
  expect_fp_unary(ztt_opcode_t::mcos_ew, dtype, input, expected, flags);
}

void expect_transcendental(ztt_fp_trans_op op, uint32_t dtype,
                           uint64_t input, uint64_t expected, uint8_t flags)
{
  const ztt_fp_result_t result =
    ztt_fp_transcendental(op, dtype, dtype, input);
  assert(result.bits == expected);
  assert(result.flags == flags);
}

void test_softfloat_rounding_mapping()
{
  struct rounding_case_t {
    unsigned mode;
    uint32_t expected;
  };
  const rounding_case_t cases[] = {
    {0, 0x3f800000}, // RNE
    {1, 0x3f800000}, // RTZ
    {2, 0x3f800000}, // RDN
    {3, 0x3f800001}, // RUP
    {4, 0x3f800001}, // RMM
  };
  for (const auto& c : cases) {
    fixture_t f;
    const uint32_t dtype = fp32(c.mode);
    f.set_m(1, dtype, 0);
    f.set_m(2, dtype, 0x3f800000); // 1
    f.set_m(3, dtype, 0x33800000); // 2^-24, halfway at 1
    execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(1, 2, 3));
    f.expect_m(1, c.expected);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
  }
}

void test_integer_to_float_is_direct()
{
  using boost::multiprecision::cpp_int;
  const uint32_t dtype = fp32(0);
  const cpp_int value = (cpp_int(1) << 63) + (cpp_int(1) << 39) + 1;
  const ztt_fp_result_t result = ztt_fp_from_integer(value, dtype);
  // Direct RNE conversion.  Converting through binary64 first loses the low
  // bit and produces 0x5f000000 instead.
  assert(result.bits == UINT64_C(0x5f000001));
  assert(result.flags == softfloat_flag_inexact);

  fixture_t f;
  f.set_m(2, fp32(0), 0xdeadbeef);
  f.set_m(4, kU64, UINT64_C(0x8000008000000001));
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(2, 4));
  f.expect_m(2, 0x5f000001);
  assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
}

void test_fp64_to_fp8_rounds_directly()
{
  struct conversion_case_t {
    uint32_t dtype;
    uint64_t just_above_midpoint;
    uint8_t expected;
  };
  // Each input is one binary64 ulp above an exact FP8 midpoint.  A binary32
  // intermediate loses that low bit and would tie-to-even in the wrong
  // direction.  mconv must round the original value directly to FP8.
  const conversion_case_t cases[] = {
    {fp8e4m3(0), UINT64_C(0x3ff1000000000001), 0x39},
    {fp8e5m2(0), UINT64_C(0x3ff2000000000001), 0x3d},
  };
  for (const auto& c : cases) {
    fixture_t f;
    f.set_m(0, c.dtype, 0);
    for (unsigned reg = 8; reg < 16; reg += 2) {
      assert(f.proc.ZTU.set_m_datatype(reg, fp64(0)));
      assert(f.proc.ZTU.write_m_as(reg, fp64(0),
        std::vector<elem_t>(ztt::kNumElements, c.just_above_midpoint)));
    }
    execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(0, 8));
    f.expect_m(0, c.expected);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
  }

  struct directed_case_t {
    bool e4m3;
    uint64_t positive_midpoint;
    uint64_t negative_midpoint;
    uint64_t half_min_subnormal;
    uint8_t lower;
    uint8_t upper;
  };
  const directed_case_t directed[] = {
    {true, UINT64_C(0x3ff1000000000000),
     UINT64_C(0xbff1000000000000), UINT64_C(0x3f50000000000000),
     0x38, 0x39},
    {false, UINT64_C(0x3ff2000000000000),
     UINT64_C(0xbff2000000000000), UINT64_C(0x3ee0000000000000),
     0x3c, 0x3d},
  };
  const bool positive_up[] = {false, false, false, true, true};
  const bool negative_up_magnitude[] = {false, false, true, false, true};
  const bool half_min_up[] = {false, false, false, true, true};
  for (const auto& c : directed)
    for (unsigned mode = 0; mode <= 4; ++mode) {
      const uint32_t dst = c.e4m3 ? fp8e4m3(mode) : fp8e5m2(mode);
      ztt_fp_result_t result = ztt_round_pack(
        ztt_decode_exact_fp(c.positive_midpoint, fp64(0)), dst);
      assert(result.bits == (positive_up[mode] ? c.upper : c.lower));
      assert(result.flags == softfloat_flag_inexact);

      result = ztt_round_pack(
        ztt_decode_exact_fp(c.negative_midpoint, fp64(0)), dst);
      assert(result.bits == ((negative_up_magnitude[mode] ? c.upper
                                                          : c.lower) | 0x80));
      assert(result.flags == softfloat_flag_inexact);

      result = ztt_round_pack(
        ztt_decode_exact_fp(c.half_min_subnormal, fp64(0)), dst);
      assert(result.bits == (half_min_up[mode] ? 1 : 0));
      assert(result.flags ==
             (softfloat_flag_underflow | softfloat_flag_inexact));
    }
}

void test_fp8_exact_operation_boundaries()
{
  struct fp8_case_t {
    uint32_t dtype;
    uint8_t one;
    uint8_t two;
    uint8_t three;
    uint8_t one_third;
    uint8_t sqrt_two;
    uint8_t infinity;
    uint8_t canonical_nan;
  };
  const fp8_case_t cases[] = {
    {fp8e4m3(0), 0x38, 0x40, 0x44, 0x2b, 0x3b, 0x00, 0x7f},
    {fp8e5m2(0), 0x3c, 0x40, 0x42, 0x35, 0x3e, 0x7c, 0x7e},
  };
  for (const auto& c : cases) {
    ztt_fp_result_t result = ztt_fp_div_exact(c.one, c.three, c.dtype);
    assert(result.bits == c.one_third);
    assert(result.flags == softfloat_flag_inexact);

    result = ztt_fp_sqrt_exact(c.two, c.dtype);
    assert(result.bits == c.sqrt_two);
    assert(result.flags == softfloat_flag_inexact);

    // FMA must retain the exact product until the addend is included.
    result = ztt_fp_fma_exact(c.one, c.three, c.three ^ 0x80, c.dtype);
    assert(result.bits == 0);
    assert(result.flags == 0);

    result = ztt_fp_fma_exact(0, c.infinity, c.one, c.dtype);
    if (c.infinity != 0) {
      assert(result.bits == c.canonical_nan);
      assert(result.flags == softfloat_flag_invalid);
    }
  }

  // E4M3 has no infinity: overflow saturates to max finite.  E5M2 follows
  // IEEE directed-overflow selection.
  ztt_exact_fp_t huge;
  huge.significand = 1;
  huge.exponent = 20;
  ztt_fp_result_t result = ztt_round_pack(huge, fp8e4m3(0));
  assert(result.bits == 0x7e);
  assert(result.flags == (softfloat_flag_overflow | softfloat_flag_inexact));
  result = ztt_round_pack(huge, fp8e5m2(1));
  assert(result.bits == 0x7b);
  assert(result.flags == (softfloat_flag_overflow | softfloat_flag_inexact));
  result = ztt_round_pack(huge, fp8e5m2(3));
  assert(result.bits == 0x7c);
  assert(result.flags == (softfloat_flag_overflow | softfloat_flag_inexact));

  result = ztt_fp_div_exact(0x38, 0, fp8e4m3(0));
  assert(result.bits == 0x7e);
  assert(result.flags == softfloat_flag_infinite);
  result = ztt_fp_div_exact(0x3c, 0, fp8e5m2(0));
  assert(result.bits == 0x7c);
  assert(result.flags == softfloat_flag_infinite);
  result = ztt_fp_div_exact(0, 0, fp8e5m2(0));
  assert(result.bits == 0x7e);
  assert(result.flags == softfloat_flag_invalid);
  result = ztt_fp_addsub_exact(0x7c, 0xfc, fp8e5m2(0), false);
  assert(result.bits == 0x7e);
  assert(result.flags == softfloat_flag_invalid);
  result = ztt_fp_addsub_exact(0x7f, 0x3c, fp8e5m2(0), false);
  assert(result.bits == 0x7e);
  assert(result.flags == 0); // quiet NaN propagation
  result = ztt_fp_addsub_exact(0x7d, 0x3c, fp8e5m2(0), false);
  assert(result.bits == 0x7e);
  assert(result.flags == softfloat_flag_invalid); // signaling NaN
  result = ztt_fp_sqrt_exact(0xb8, fp8e4m3(0));
  assert(result.bits == 0x7f);
  assert(result.flags == softfloat_flag_invalid);
  result = ztt_fp_sqrt_exact(0xbc, fp8e5m2(0));
  assert(result.bits == 0x7e);
  assert(result.flags == softfloat_flag_invalid);
  result = ztt_fp_mul_exact(0x7e, 0x40, fp8e4m3(0));
  assert(result.bits == 0x7e);
  assert(result.flags == (softfloat_flag_overflow | softfloat_flag_inexact));
  result = ztt_fp_mul_exact(0x7b, 0x40, fp8e5m2(0));
  assert(result.bits == 0x7c);
  assert(result.flags == (softfloat_flag_overflow | softfloat_flag_inexact));
  result = ztt_fp_mul_exact(0x01, 0x30, fp8e4m3(0)); // minsub * 0.5
  assert(result.bits == 0);
  assert(result.flags == (softfloat_flag_underflow | softfloat_flag_inexact));
  result = ztt_fp_mul_exact(0x01, 0x38, fp8e5m2(0)); // minsub * 0.5
  assert(result.bits == 0);
  assert(result.flags == (softfloat_flag_underflow | softfloat_flag_inexact));

  // Exhaust every FP8 input for identity conversion and square root.  This
  // covers zero signs, all subnormals, normal exponent boundaries, maxima,
  // infinities (E5M2), and every NaN encoding.
  for (uint32_t dtype : {fp8e4m3(0), fp8e5m2(0)})
    for (unsigned raw = 0; raw < 256; ++raw) {
      const ztt_exact_fp_t decoded = ztt_decode_exact_fp(raw, dtype);
      result = ztt_round_pack(decoded, dtype);
      if (decoded.nan) {
        assert(result.bits == (dtype == fp8e5m2(0) ? 0x7e : 0x7f));
        assert(bool(result.flags & softfloat_flag_invalid) ==
               decoded.signaling_nan);
      } else {
        assert(result.bits == raw);
        assert(result.flags == 0);
      }
      if (!decoded.nan && !decoded.sign && !decoded.infinity) {
        result = ztt_fp_sqrt_exact(raw, dtype);
        assert((result.flags & softfloat_flag_invalid) == 0);
      }
    }
}

void test_fp8_instruction_paths()
{
  struct fp8_case_t {
    uint32_t dtype;
    uint8_t zero;
    uint8_t one;
    uint8_t two;
    uint8_t three;
    uint8_t one_third;
    uint8_t sqrt_two;
  };
  const fp8_case_t cases[] = {
    {fp8e4m3(0), 0, 0x38, 0x40, 0x44, 0x2b, 0x3b},
    {fp8e5m2(0), 0, 0x3c, 0x40, 0x42, 0x35, 0x3e},
  };
  for (const auto& c : cases) {
    {
      fixture_t f;
      f.set_m(0, c.dtype, c.zero);
      f.set_m(1, c.dtype, c.one);
      f.set_m(2, c.dtype, c.two);
      execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(0, 1, 2));
      f.expect_m(0, c.three);
      assert(f.proc.ZTU.amefflags() == 0);
    }
    {
      fixture_t f;
      f.set_m(0, c.dtype, c.zero);
      f.set_m(1, c.dtype, c.one);
      f.set_m(2, c.dtype, c.three);
      execute_ztt(f.proc, ztt_opcode_t::mrec_ew, operands(0, 2));
      f.expect_m(0, c.one_third);
      assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
    }
    {
      fixture_t f;
      f.set_m(0, c.dtype, c.zero);
      f.set_m(2, c.dtype, c.two);
      execute_ztt(f.proc, ztt_opcode_t::msqrt_ew, operands(0, 2));
      f.expect_m(0, c.sqrt_two);
      assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
    }
  }
}

void test_fp8_exhaustive_against_softfloat_reference()
{
  const uint_fast8_t saved_rounding = softfloat_roundingMode;
  const uint_fast8_t saved_flags = softfloat_exceptionFlags;
  const uint_fast8_t saved_tininess = softfloat_detectTininess;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  const auto to_f32 = [](uint8_t raw, bool e4m3) {
    return e4m3
      ? bf16_to_f32(e4m3_to_bf16(e4m3_t{raw}))
      : bf16_to_f32(e5m2_to_bf16(e5m2_t{raw}));
  };
  const auto from_f32 = [](float32_t value, bool e4m3) {
    return uint8_t(e4m3 ? f32_to_e4m3(value, true).v
                        : f32_to_e5m2(value, false).v);
  };
  const auto after_rounding_flags = [](uint8_t flags, uint8_t result,
                                        bool e4m3) {
    // The imported FP8 round-pack code can report tininess-before-rounding
    // when a tiny inexact value rounds up to the minimum normal.  Ztt 6.2.3
    // requires tininess after rounding, so normalize only UF while retaining
    // the independently generated NV/DZ/OF/NX classes.
    flags &= ~softfloat_flag_underflow;
    const uint8_t exponent_mask = e4m3 ? 0x78 : 0x7c;
    if ((flags & softfloat_flag_inexact) &&
        (result & exponent_mask) == 0)
      flags |= softfloat_flag_underflow;
    return flags;
  };
  for (bool e4m3 : {true, false})
    for (unsigned mode = 0; mode <= 4; ++mode) {
      const uint32_t dtype = e4m3 ? fp8e4m3(mode) : fp8e5m2(mode);
      softfloat_roundingMode = ztt_to_softfloat_round(dtype);
      for (unsigned lhs = 0; lhs < 256; ++lhs) {
        const ztt_exact_fp_t l = ztt_decode_exact_fp(lhs, dtype);
        if (l.nan || l.infinity)
          continue;
        const float32_t fl = to_f32(lhs, e4m3);
        softfloat_exceptionFlags = 0;
        const uint8_t reference_sqrt = from_f32(f32_sqrt(fl), e4m3);
        const uint8_t reference_sqrt_flags = after_rounding_flags(
          softfloat_exceptionFlags & ztt::kAmeFlagMask, reference_sqrt,
          e4m3);
        const ztt_fp_result_t exact_sqrt = ztt_fp_sqrt_exact(lhs, dtype);
        assert(exact_sqrt.bits == reference_sqrt);
        assert(exact_sqrt.flags == reference_sqrt_flags);

        for (unsigned rhs = 0; rhs < 256; ++rhs) {
          const ztt_exact_fp_t r = ztt_decode_exact_fp(rhs, dtype);
          if (r.nan || r.infinity)
            continue;
          const float32_t fr = to_f32(rhs, e4m3);

          softfloat_exceptionFlags = 0;
          const uint8_t reference_add = from_f32(f32_add(fl, fr), e4m3);
          const uint8_t reference_add_flags = after_rounding_flags(
            softfloat_exceptionFlags & ztt::kAmeFlagMask, reference_add,
            e4m3);
          ztt_fp_result_t exact = ztt_fp_addsub_exact(lhs, rhs, dtype, false);
          assert(exact.bits == reference_add);
          assert(exact.flags == reference_add_flags);

          softfloat_exceptionFlags = 0;
          const uint8_t reference_mul = from_f32(f32_mul(fl, fr), e4m3);
          const uint8_t reference_mul_flags = after_rounding_flags(
            softfloat_exceptionFlags & ztt::kAmeFlagMask, reference_mul,
            e4m3);
          exact = ztt_fp_mul_exact(lhs, rhs, dtype);
          assert(exact.bits == reference_mul);
          assert(exact.flags == reference_mul_flags);

          softfloat_exceptionFlags = 0;
          const uint8_t reference_div = from_f32(f32_div(fl, fr), e4m3);
          const uint8_t reference_div_flags = after_rounding_flags(
            softfloat_exceptionFlags & ztt::kAmeFlagMask, reference_div,
            e4m3);
          exact = ztt_fp_div_exact(lhs, rhs, dtype);
          assert(exact.bits == reference_div);
          assert(exact.flags == reference_div_flags);

        }
      }
    }

  // This exact E5M2 FMA is just above an FP8 midpoint.  An FP32 FMA loses
  // the 2^-16 addend at 288 and then rounds the FP8 tie to 256 (0x5c), while
  // the required single FP8 rounding produces 320 (0x5d).
  const ztt_fp_result_t fused = ztt_fp_fma_exact(
    0x1e, 0x7a, 0x01, fp8e5m2(0));
  assert(fused.bits == 0x5d);
  assert(fused.flags == softfloat_flag_inexact);
  softfloat_roundingMode = saved_rounding;
  softfloat_exceptionFlags = saved_flags;
  softfloat_detectTininess = saved_tininess;
}

void test_matrix_float_precision_contract()
{
  const uint32_t dtype = fp32(0);
  {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    // The specified q order gives (((2^24 + 1) - 2^24) + 1) = 1 in FP32.
    a[0] = 0x4b800000;
    a[1] = 0x3f800000;
    a[2] = 0xcb800000;
    a[3] = 0x3f800000;
    for (unsigned q = 0; q < ztt::kTileLength; ++q)
      b[q * ztt::kTileLength] = 0x3f800000;
    assert(f.proc.ZTU.set_m_datatype(2, dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, dtype, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == 0x3f800000);
  }

  // Every declared IEEE storage format uses its accumulator datatype as the
  // arithmetic and rounding domain.
  struct format_case_t {
    uint32_t dtype;
    elem_t one;
    elem_t one_point_five;
    elem_t two;
    elem_t negative_two;
    elem_t four;
  };
  const format_case_t formats[] = {
    {fp16(0), 0x3c00, 0x3e00, 0x4000, 0xc000, 0x4400},
    {bf16(0), 0x3f80, 0x3fc0, 0x4000, 0xc000, 0x4080},
    {fp32(0), 0x3f800000, 0x3fc00000, 0x40000000,
     0xc0000000, 0x40800000},
    {fp64(0), UINT64_C(0x3ff0000000000000),
     UINT64_C(0x3ff8000000000000), UINT64_C(0x4000000000000000),
     UINT64_C(0xc000000000000000),
     UINT64_C(0x4010000000000000)},
  };
  const ztt_opcode_t matrix_ops[] = {
    ztt_opcode_t::mmulacc_2d,
    ztt_opcode_t::mmulaccneg_2d,
    ztt_opcode_t::mmulatacc_2d,
    ztt_opcode_t::mmulataccneg_2d,
    ztt_opcode_t::mmulbtacc_2d,
    ztt_opcode_t::mmulbtaccneg_2d,
  };
  for (const auto& c : formats)
    for (unsigned op = 0; op < 6; ++op) {
      fixture_t f;
      std::vector<elem_t> a(ztt::kNumElements, 0);
      std::vector<elem_t> b(ztt::kNumElements, 0);
      a[0] = c.one_point_five;
      b[0] = c.two;
      assert(f.proc.ZTU.set_m_datatype(8, c.dtype));
      assert(f.proc.ZTU.write_m(8,
        [&] { std::vector<elem_t> v(ztt_unit_t::element_count(c.dtype));
              v[0] = c.one_point_five; return v; }()));
      assert(f.proc.ZTU.set_m_datatype(12, c.dtype));
      assert(f.proc.ZTU.write_m(12,
        [&] { std::vector<elem_t> v(ztt_unit_t::element_count(c.dtype));
              v[0] = c.two; return v; }()));
      f.set_acc(0, c.dtype, c.one);
      execute_ztt(f.proc, matrix_ops[op], operands(0, 8, 12));
      assert(f.get_acc(0)[0] == (op & 1 ? c.negative_two : c.four));
      assert(f.proc.ZTU.amefflags() == 0);
    }
  {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    a[0] = 0x3f800001; // 1 + 2^-23
    b[0] = 0x3f7fffff; // 1 - 2^-24
    assert(f.proc.ZTU.set_m_datatype(2, dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, dtype, 0xbf800000); // -1

    f.proc.ZTU.write_amefflags(0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    // Exact result is 2^-24 - 2^-47 = 0x337ffffe.  A non-fused FP32
    // multiply followed by add rounds the product to one and yields zero.
    assert(f.get_acc(0)[0] == 0x337ffffe);
    assert(f.proc.ZTU.amefflags() == 0);
  }

  // Packed FP8 sources contribute every one of their four logical squares
  // to the single accumulator result (N_sq * N terms).
  for (uint32_t fp8 : {fp8e4m3(0), fp8e5m2(0)})
    for (unsigned op = 0; op < 6; ++op) {
      fixture_t f;
      const elem_t one = fp8 == fp8e4m3(0) ? 0x38 : 0x3c;
      const elem_t four = fp8 == fp8e4m3(0) ? 0x48 : 0x44;
      std::vector<elem_t> a(ztt_unit_t::element_count(fp8), 0);
      std::vector<elem_t> b(ztt_unit_t::element_count(fp8), 0);
      for (unsigned square = 0; square < 4; ++square) {
        const unsigned base = square * ztt::kNumElements;
        a[base] = one;
        b[base] = one;
      }
      assert(f.proc.ZTU.set_m_datatype(2, fp8));
      assert(f.proc.ZTU.write_m(2, a));
      assert(f.proc.ZTU.set_m_datatype(3, fp8));
      assert(f.proc.ZTU.write_m(3, b));
      f.set_acc(0, fp8, 0);
      execute_ztt(f.proc, matrix_ops[op], operands(0, 2, 3));
      assert(f.get_acc(0)[0] == (op & 1 ? (four ^ 0x80) : four));
      assert(f.proc.ZTU.amefflags() == 0);
    }

  // FP8 matrix FMA is genuinely fused in the accumulator format.
  struct fp8_fma_case_t {
    uint32_t dtype;
    elem_t factor;
    elem_t addend;
    elem_t expected;
  };
  const fp8_fma_case_t fp8_fma_cases[] = {
    {fp8e4m3(0), 0x39, 0xba, 0x08}, // 1.125^2 - 1.25 = 2^-6
    {fp8e5m2(0), 0x3d, 0xbe, 0x2c}, // 1.25^2 - 1.5 = 2^-4
  };
  for (const auto& c : fp8_fma_cases) {
    fixture_t f;
    std::vector<elem_t> a(ztt_unit_t::element_count(c.dtype), 0);
    std::vector<elem_t> b(ztt_unit_t::element_count(c.dtype), 0);
    a[0] = c.factor;
    b[0] = c.factor;
    assert(f.proc.ZTU.set_m_datatype(2, c.dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, c.dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, c.dtype, c.addend);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == c.expected);
    assert(f.proc.ZTU.amefflags() == 0);
  }

  // Source conversion is direct to the accumulator format.  These FP64
  // values are one binary64 ulp above an FP8 midpoint and distinguish direct
  // rounding from an FP32 intermediate.
  struct mixed_case_t {
    uint32_t accumulator_dtype;
    uint64_t source;
    elem_t one64;
    elem_t expected;
  };
  const mixed_case_t mixed_cases[] = {
    {fp8e4m3(0), UINT64_C(0x3ff1000000000001),
     UINT64_C(0x3ff0000000000000), 0x39},
    {fp8e5m2(0), UINT64_C(0x3ff2000000000001),
     UINT64_C(0x3ff0000000000000), 0x3d},
  };
  for (const auto& c : mixed_cases) {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    a[0] = c.source;
    b[0] = c.one64;
    assert(f.proc.ZTU.set_m_datatype(8, fp64(0)));
    assert(f.proc.ZTU.write_m(8, a));
    assert(f.proc.ZTU.set_m_datatype(12, fp64(0)));
    assert(f.proc.ZTU.write_m(12, b));
    f.set_acc(0, c.accumulator_dtype, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 8, 12));
    assert(f.get_acc(0)[0] == c.expected);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
  }

  // Destination rounding is selected from Ad, independently of fcsr.frm.
  const uint32_t rounding_results[] = {
    0x3f800000, // RNE
    0x3f800000, // RTZ
    0x3f800000, // RDN
    0x3f800001, // RUP
    0x3f800001, // RMM
  };
  for (unsigned mode = 0; mode <= 4; ++mode) {
    fixture_t f;
    const uint32_t rounded_dtype = fp32(mode);
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    a[0] = 0x33800000; // 2^-24, half an ulp at 1
    b[0] = 0x3f800000;
    assert(f.proc.ZTU.set_m_datatype(2, rounded_dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, rounded_dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, rounded_dtype, 0x3f800000);
    softfloat_roundingMode = softfloat_round_max;
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == rounding_results[mode]);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
    assert(softfloat_roundingMode == softfloat_round_max);
  }

  // Invalid products are canonicalized and accrue NV rather than trapping.
  {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    b[0] = 0x7f800000;
    assert(f.proc.ZTU.set_m_datatype(2, dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, dtype, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == 0x7fc00000);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_invalid);
  }

  // Matrix flags are those of the documented destination-domain FMA fold.
  {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    a[0] = 0x7f7fffff;
    b[0] = 0x40000000;
    assert(f.proc.ZTU.set_m_datatype(2, dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, dtype, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == 0x7f800000);
    assert(f.proc.ZTU.amefflags() ==
           (softfloat_flag_overflow | softfloat_flag_inexact));
  }
  {
    fixture_t f;
    std::vector<elem_t> a(ztt::kNumElements, 0);
    std::vector<elem_t> b(ztt::kNumElements, 0);
    a[0] = 0x00000001;
    b[0] = 0x3f000000;
    assert(f.proc.ZTU.set_m_datatype(2, dtype));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, dtype));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, dtype, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == 0);
    assert(f.proc.ZTU.amefflags() ==
           (softfloat_flag_underflow | softfloat_flag_inexact));
  }


  // Integer matrix accumulation applies the documented conversion policy on
  // every fold step: saturation clamps and raises the sticky SAT flag.
  {
    fixture_t f;
    constexpr uint32_t sat_i8 = (1u << 30) | (1u << 29) | 8u;
    std::vector<elem_t> a(ztt_unit_t::element_count(sat_i8), 0);
    std::vector<elem_t> b(ztt_unit_t::element_count(sat_i8), 0);
    a[0] = 1;
    b[0] = 1;
    assert(f.proc.ZTU.set_m_datatype(2, sat_i8));
    assert(f.proc.ZTU.write_m(2, a));
    assert(f.proc.ZTU.set_m_datatype(3, sat_i8));
    assert(f.proc.ZTU.write_m(3, b));
    f.set_acc(0, sat_i8, 0x7f);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 3));
    assert(f.get_acc(0)[0] == 0x7f);
    assert(f.proc.ZTU.amexsat() == 1);
  }
}

void test_compound_float_expressions_round_once()
{
  {
    fixture_t f;
    const uint32_t dtype = fp32(0);
    f.set_m(2, dtype, 0xdeadbeef);
    f.set_m(4, dtype, 0x00000001); // minimum FP32 subnormal
    f.set_m(6, dtype, 0x00000001);
    execute_ztt(f.proc, ztt_opcode_t::mmean_ew, operands(2, 4, 6));
    f.expect_m(2, 0x00000001);
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    const uint32_t dtype = fp32(0);
    f.set_m(2, dtype, 0xdeadbeef);
    f.set_m(4, dtype, 0x3f800000); // 1
    f.set_m(6, dtype, 0x3f800001); // 1 + 1 ulp
    execute_ztt(f.proc, ztt_opcode_t::mabsdiff_ew, operands(2, 4, 6));
    f.expect_m(2, 0x34000000);
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    const uint32_t dtype = fp32(0);
    f.set_m(2, dtype, 0xdeadbeef);
    f.set_m(4, dtype, 0x3fc00000); // 1.5
    f.set_m(6, dtype, 0x40000000); // 2
    execute_ztt(f.proc, ztt_opcode_t::mmulneg_ew, operands(2, 4, 6));
    f.expect_m(2, 0xc0400000); // -3
    assert(f.proc.ZTU.amefflags() == 0);
  }
}

void test_integer_ldexpacc_rounds_complete_expression_once()
{
  fixture_t f;
  f.set_m(2, kI32, 1);          // old accumulator D
  f.set_m(4, kI32, 1);          // A
  f.set_m(6, kI32, UINT32_C(0xffffffff)); // B = -1
  execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew, operands(2, 4, 6));
  // D + A*2^-1 = 1.5, rounded once with integer RNU.
  f.expect_m(2, 2);
}

void test_bf16_quiet_nan_compare_does_not_set_invalid()
{
  fixture_t f;
  const uint32_t dtype = bf16(0);
  f.set_m(2, kI16, 0xffff);
  f.set_m(4, dtype, 0x7fc0); // quiet NaN
  f.set_m(6, dtype, 0x3f80); // 1
  execute_ztt(f.proc, ztt_opcode_t::mcmplt_ew, operands(2, 4, 6));
  f.expect_m(2, 0);
  assert(f.proc.ZTU.amefflags() == 0);
}

void test_log2_compound_integer_bias()
{
  const uint32_t dtype = fp32(0);
  const struct case_t {
    ztt_opcode_t opcode;
    bool scalar;
    uint32_t bias_dtype;
    uint64_t bias;
    elem_t logarithm_input;
    elem_t expected;
  } cases[] = {
    // Signed integer bias: log2(8) - (-2) = 5 and its reverse is -5.
    {ztt_opcode_t::mlog2sub_ew, false, kI32, UINT32_C(0xfffffffe),
     0x41000000, 0x40a00000},
    {ztt_opcode_t::msublog2_ew, false, kI32, UINT32_C(0xfffffffe),
     0x41000000, 0xc0a00000},
    {ztt_opcode_t::mlog2sub_ew_x, true, kI32, UINT32_C(0xfffffffe),
     0x41000000, 0x40a00000},
    {ztt_opcode_t::msublog2_ew_x, true, kI32, UINT32_C(0xfffffffe),
     0x41000000, 0xc0a00000},

    // 2^24+1 cannot be represented in FP32.  Keeping the unsigned bias exact
    // until subtraction distinguishes these results from pre-conversion.
    {ztt_opcode_t::mlog2sub_ew, false, kU64, UINT64_C(16777217),
     0x40000000, 0xcb800000},
    {ztt_opcode_t::msublog2_ew, false, kU64, UINT64_C(16777217),
     0x40000000, 0x4b800000},
    {ztt_opcode_t::mlog2sub_ew_x, true, kU64, UINT64_C(16777217),
     0x40000000, 0xcb800000},
    {ztt_opcode_t::msublog2_ew_x, true, kU64, UINT64_C(16777217),
     0x40000000, 0x4b800000},
  };

  for (const auto& item : cases) {
    fixture_t f;
    f.set_m(2, dtype, 0xdeadbeef);
    f.set_m(4, dtype, item.logarithm_input);
    if (item.scalar) {
      f.proc.ZTU.write_amestype(item.bias_dtype);
      f.proc.get_state()->XPR.write(7, item.bias);
      execute_ztt(f.proc, item.opcode, operands(2, 7, 4));
    } else {
      f.set_m(6, item.bias_dtype, item.bias);
      execute_ztt(f.proc, item.opcode, operands(2, 4, 6));
    }
    assert(f.proc.ZTU.amestatus() == 0);
    f.expect_m(2, item.expected);
    assert(f.proc.ZTU.amefflags() == 0);
  }
}

void test_rno_is_allowed_for_non_rounding_operations()
{
  const uint32_t rno = fp32(5);
  {
    fixture_t f;
    f.set_m(2, rno, 0xdeadbeef);
    execute_ztt(f.proc, ztt_opcode_t::mzero_2d_m, operands(2));
    assert(f.proc.ZTU.amestatus() == 0);
    f.expect_m(2, 0);
  }
  {
    fixture_t f;
    f.set_m(2, fp32(0), 0xdeadbeef);
    f.set_m(4, rno, 0x3f800000);
    execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(2, 4));
    assert(f.proc.ZTU.amestatus() == 0);
    f.expect_m(2, 0x3f800000);
  }
}

void test_exact_ldexp_rounding_and_flags()
{
  struct scale_case_t {
    unsigned mode;
    uint32_t input;
    int32_t exponent;
    uint32_t expected;
    uint8_t flags;
  };
  const scale_case_t cases[] = {
    {0, 0x00800000, -1, 0x00400000, 0},
    {0, 0x00000001, -1, 0x00000000,
      uint8_t(softfloat_flag_underflow | softfloat_flag_inexact)},
    {3, 0x00000001, -1, 0x00000001,
      uint8_t(softfloat_flag_underflow | softfloat_flag_inexact)},
    {0, 0x7f7fffff, 1, 0x7f800000,
      uint8_t(softfloat_flag_overflow | softfloat_flag_inexact)},
    {1, 0x7f7fffff, 1, 0x7f7fffff,
      uint8_t(softfloat_flag_overflow | softfloat_flag_inexact)},
    {2, 0xff7fffff, 1, 0xff800000,
      uint8_t(softfloat_flag_overflow | softfloat_flag_inexact)},
    {3, 0xff7fffff, 1, 0xff7fffff,
      uint8_t(softfloat_flag_overflow | softfloat_flag_inexact)},
  };
  for (const auto& c : cases) {
    fixture_t f;
    const uint32_t dtype = fp32(c.mode);
    f.set_m(1, dtype, 0);
    f.set_m(2, dtype, c.input);
    f.set_m(3, kI32, uint32_t(c.exponent));
    execute_ztt(f.proc, ztt_opcode_t::mldexp_ew, operands(1, 2, 3));
    f.expect_m(1, c.expected);
    assert(f.proc.ZTU.amefflags() == c.flags);
  }
}

void test_round_pack_midpoints_and_tininess()
{
  struct rounding_case_t {
    unsigned mode;
    uint32_t positive;
    uint32_t negative;
  };
  const rounding_case_t cases[] = {
    {0, 0x3f800000, 0xbf800000}, // RNE
    {1, 0x3f800000, 0xbf800000}, // RTZ
    {2, 0x3f800000, 0xbf800001}, // RDN
    {3, 0x3f800001, 0xbf800000}, // RUP
    {4, 0x3f800001, 0xbf800001}, // RMM
  };
  for (const auto& c : cases) {
    ztt_exact_fp_t exact;
    // 1 + 2^-24, exactly halfway between two adjacent FP32 values.
    exact.significand = (boost::multiprecision::cpp_int(1) << 24) + 1;
    exact.exponent = -24;
    ztt_fp_result_t result = ztt_round_pack(exact, fp32(c.mode));
    assert(result.bits == c.positive);
    assert(result.flags == softfloat_flag_inexact);

    exact.sign = true;
    result = ztt_round_pack(exact, fp32(c.mode));
    assert(result.bits == c.negative);
    assert(result.flags == softfloat_flag_inexact);
  }

  ztt_exact_fp_t rounds_to_normal;
  rounds_to_normal.significand =
    (boost::multiprecision::cpp_int(1) << 24) - 1;
  rounds_to_normal.exponent = -150;
  ztt_fp_result_t result = ztt_round_pack(rounds_to_normal, fp32(0));
  assert(result.bits == 0x00800000);
  assert(result.flags == softfloat_flag_inexact);

  ztt_exact_fp_t remains_tiny;
  remains_tiny.significand =
    (boost::multiprecision::cpp_int(1) << 24) - 3;
  remains_tiny.exponent = -150;
  result = ztt_round_pack(remains_tiny, fp32(0));
  assert(result.bits == 0x007ffffe);
  assert(result.flags ==
         (softfloat_flag_underflow | softfloat_flag_inexact));
}

void test_ldexpacc_single_rounding()
{
  fixture_t f;
  const uint32_t dtype = fp32(0);
  f.set_m(1, dtype, 0x00000001); // Existing destination: 1 * 2^-149.
  f.set_m(2, dtype, 0x00000001); // Scaled source contributes 1 * 2^-150.
  f.set_m(3, kI32, uint32_t(-1));
  execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew, operands(1, 2, 3));
  // Exact sum is 1.5 * 2^-149.  RNE rounds once to two subnormal quanta.
  f.expect_m(1, 0x00000002);
  assert(f.proc.ZTU.amefflags() ==
         (softfloat_flag_underflow | softfloat_flag_inexact));
}

void test_ldexpacc_exact_cancellation_and_invalid_infinity()
{
  for (unsigned mode : {0u, 2u}) {
    fixture_t f;
    const uint32_t dtype = fp32(mode);
    f.set_m(1, dtype, 0x3f800000); // Existing destination: +1.
    f.set_m(2, dtype, 0xbf800000); // Scaled source: -1.
    f.set_m(3, kI32, 0);
    execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew, operands(1, 2, 3));
    f.expect_m(1, mode == 2 ? 0x80000000 : 0x00000000);
    assert(f.proc.ZTU.amefflags() == 0);
  }

  const ztt_fp_result_t invalid = ztt_fp_scale_acc_exact(
    0x7f800000, 0xff800000, fp32(0), 0);
  assert(invalid.bits == 0x7fc00000);
  assert(invalid.flags == softfloat_flag_invalid);
}

void test_exact_scale_special_values()
{
  ztt_fp_result_t result =
    ztt_fp_scale_exact(0x7f812345, fp32(0), 100);
  assert(result.bits == 0x7fc00000);
  assert(result.flags == softfloat_flag_invalid);

  result = ztt_fp_scale_exact(0x80000000, fp32(0), -4096);
  assert(result.bits == 0x80000000);
  assert(result.flags == 0);

  result = ztt_fp_scale_exact(0xff800000, fp32(0), 4096);
  assert(result.bits == 0xff800000);
  assert(result.flags == 0);
}

void test_rno_is_explicitly_unsupported()
{
  fixture_t f;
  const uint32_t dtype = fp32(5);
  f.set_m(1, dtype, 0x40400000);
  f.set_m(2, dtype, 0x3f800000);
  f.set_m(3, dtype, 0x3f800000);
  f.proc.ZTU.write_amestatus(0);
  execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(1, 2, 3));
  assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
  f.expect_m(1, 0x40400000);
}

void test_mcos_fp32_rne_results_and_flags()
{
  struct cos_case_t {
    uint32_t input;
    uint32_t expected;
    uint8_t flags;
  };
  const cos_case_t cases[] = {
    {0x00000000, 0x3f800000, 0},
    {0x80000000, 0x3f800000, 0},
    {0x7f800000, 0x7fc00000, softfloat_flag_invalid},
    {0xff800000, 0x7fc00000, softfloat_flag_invalid},
    {0x7fc12345, 0x7fc00000, 0},
    {0x7f812345, 0x7fc00000, softfloat_flag_invalid},
    {0x3f800000, 0x3f0a5140, softfloat_flag_inexact},
  };
  for (const auto& c : cases) {
    fixture_t f;
    const uint32_t dtype = fp32(0);
    f.set_m(1, dtype, 0xdeadbeef);
    f.set_m(2, dtype, c.input);
    execute_ztt(f.proc, ztt_opcode_t::mcos_ew, operands(1, 2));
    f.expect_m(1, c.expected);
    assert(f.proc.ZTU.amefflags() == c.flags);
  }
}

void test_mcos_other_formats_rne_results_and_flags()
{
  struct format_case_t {
    uint32_t dtype;
    uint64_t negative_zero;
    uint64_t positive_infinity;
    uint64_t negative_infinity;
    uint64_t quiet_nan;
    uint64_t signaling_nan;
    uint64_t canonical_nan;
    uint64_t one;
    uint64_t cos_one;
  };
  const format_case_t formats[] = {
    {fp16(0), UINT64_C(0x8000), UINT64_C(0x7c00), UINT64_C(0xfc00),
     UINT64_C(0x7e55), UINT64_C(0x7d55), UINT64_C(0x7e00),
     UINT64_C(0x3c00), UINT64_C(0x3853)},
    {bf16(0), UINT64_C(0x8000), UINT64_C(0x7f80), UINT64_C(0xff80),
     UINT64_C(0x7fd1), UINT64_C(0x7f91), UINT64_C(0x7fc0),
     UINT64_C(0x3f80), UINT64_C(0x3f0a)},
    {fp64(0), UINT64_C(0x8000000000000000),
     UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
     UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff0123456789abc),
     UINT64_C(0x7ff8000000000000), UINT64_C(0x3ff0000000000000),
     UINT64_C(0x3fe14a280fb5068c)},
  };

  for (const auto& f : formats) {
    expect_mcos(f.dtype, 0, f.one, 0);
    expect_mcos(f.dtype, f.negative_zero, f.one, 0);
    expect_mcos(f.dtype, f.positive_infinity, f.canonical_nan,
                softfloat_flag_invalid);
    expect_mcos(f.dtype, f.negative_infinity, f.canonical_nan,
                softfloat_flag_invalid);
    expect_mcos(f.dtype, f.quiet_nan, f.canonical_nan, 0);
    expect_mcos(f.dtype, f.signaling_nan, f.canonical_nan,
                softfloat_flag_invalid);
    expect_mcos(f.dtype, f.one, f.cos_one, softfloat_flag_inexact);
  }
}

void test_transcendental_host_environment_is_restored()
{
  std::fenv_t original_env;
  assert(std::fegetenv(&original_env) == 0);
  assert(std::fesetround(FE_UPWARD) == 0);
  assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
  assert(std::feraiseexcept(FE_DIVBYZERO) == 0);
  const int expected_exceptions = std::fetestexcept(FE_ALL_EXCEPT);

  const uint32_t dtype = fp32(0);
  const ztt_fp_result_t result = ztt_fp_transcendental(
    ztt_fp_trans_op::cos, dtype, dtype, 0x3f800000);
  assert(result.bits == 0x3f0a5140);
  assert(result.flags == softfloat_flag_inexact);
  assert(std::fegetround() == FE_UPWARD);
  assert(std::fetestexcept(FE_ALL_EXCEPT) == expected_exceptions);

  assert(std::fesetenv(&original_env) == 0);
}

void test_mcos_ieee_rounding_modes()
{
  struct format_case_t {
    unsigned bits;
    unsigned exponent;
    uint64_t one;
    uint64_t expected[4];
  };
  const format_case_t formats[] = {
    {16, 5, UINT64_C(0x3c00),
     {UINT64_C(0x3853), UINT64_C(0x3852),
      UINT64_C(0x3852), UINT64_C(0x3853)}},
    {16, 8, UINT64_C(0x3f80),
     {UINT64_C(0x3f0a), UINT64_C(0x3f0a),
      UINT64_C(0x3f0a), UINT64_C(0x3f0b)}},
    {32, 8, UINT64_C(0x3f800000),
     {UINT64_C(0x3f0a5140), UINT64_C(0x3f0a5140),
      UINT64_C(0x3f0a5140), UINT64_C(0x3f0a5141)}},
    {64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3fe14a280fb5068c), UINT64_C(0x3fe14a280fb5068b),
      UINT64_C(0x3fe14a280fb5068b), UINT64_C(0x3fe14a280fb5068c)}},
  };
  for (const auto& f : formats)
    for (unsigned mode = 0; mode < 4; ++mode)
      expect_mcos(standard_float(f.bits, f.exponent, mode), f.one,
                  f.expected[mode], softfloat_flag_inexact);
}

void test_mcos_fp16_underflow()
{
  // CORE-MATH identifies 0x598c as one of the FP16 inputs whose cosine
  // rounds to a subnormal result.  ZTT supplies the architectural flags.
  expect_mcos(fp16(0), UINT64_C(0x598c), UINT64_C(0x80fd),
              softfloat_flag_underflow | softfloat_flag_inexact);
}

void test_transcendental_fp32_special_values()
{
  const uint32_t dtype = fp32(0);

  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x00000000, 0x00000000, 0);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x80000000, 0x80000000, 0);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x7f800000, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0xff800000, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x7fc12345, 0x7fc00000, 0);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x7f812345, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::msin_ew, dtype, 0x3f800000, 0x3f576aa4,
                  softfloat_flag_inexact);

  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x00000000, 0x00000000, 0);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x80000000, 0x80000000, 0);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x7f800000, 0x3f800000, 0);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0xff800000, 0xbf800000, 0);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x7fc12345, 0x7fc00000, 0);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x7f812345, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::mtanh_ew, dtype, 0x3f800000, 0x3f42f7d6,
                  softfloat_flag_inexact);

  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x00000000, 0xff800000,
                  softfloat_flag_infinite);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x80000000, 0xff800000,
                  softfloat_flag_infinite);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0xbf800000, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0xff800000, 0x7fc00000,
                  softfloat_flag_invalid);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x7f800000, 0x7f800000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x3f800000, 0x00000000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x40000000, 0x3f800000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x40800000, 0x40000000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x00000001, 0xc3150000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x40400000, 0x3fcae00d,
                  softfloat_flag_inexact);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x7fc12345, 0x7fc00000, 0);
  expect_fp_unary(ztt_opcode_t::mlog2_ew, dtype, 0x7f812345, 0x7fc00000,
                  softfloat_flag_invalid);

  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x00000000, 0x3f800000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x80000000, 0x3f800000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x7f800000, 0x7f800000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0xff800000, 0x00000000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x3f800000, 0x40000000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0xbf800000, 0x3f000000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x3f000000, 0x3fb504f3,
                  softfloat_flag_inexact);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x43000000, 0x7f800000,
                  softfloat_flag_overflow | softfloat_flag_inexact);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0xc3160000, 0x00000000,
                  softfloat_flag_underflow | softfloat_flag_inexact);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0xc3150000, 0x00000001, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x7fc12345, 0x7fc00000, 0);
  expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x7f812345, 0x7fc00000,
                  softfloat_flag_invalid);
}

void test_transcendental_other_formats_special_values()
{
  struct format_case_t {
    uint32_t dtype;
    uint64_t negative_zero;
    uint64_t positive_infinity;
    uint64_t negative_infinity;
    uint64_t quiet_nan;
    uint64_t signaling_nan;
    uint64_t canonical_nan;
    uint64_t one;
    uint64_t negative_one;
    uint64_t two;
    uint64_t half;
    uint64_t min_subnormal;
    uint64_t log2_min_subnormal;
  };
  const format_case_t formats[] = {
    {fp16(0), 0x8000, 0x7c00, 0xfc00, 0x7e55, 0x7d55, 0x7e00,
     0x3c00, 0xbc00, 0x4000, 0x3800, 0x0001, 0xce00},
    {bf16(0), 0x8000, 0x7f80, 0xff80, 0x7fd1, 0x7f91, 0x7fc0,
     0x3f80, 0xbf80, 0x4000, 0x3f00, 0x0001, 0xc305},
    {fp64(0), UINT64_C(0x8000000000000000),
     UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
     UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff0123456789abc),
     UINT64_C(0x7ff8000000000000), UINT64_C(0x3ff0000000000000),
     UINT64_C(0xbff0000000000000), UINT64_C(0x4000000000000000),
     UINT64_C(0x3fe0000000000000), UINT64_C(0x0000000000000001),
     UINT64_C(0xc090c80000000000)},
  };

  for (const auto& f : formats) {
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, 0, 0, 0);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.negative_zero,
                    f.negative_zero, 0);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.positive_infinity,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.negative_infinity,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.quiet_nan,
                    f.canonical_nan, 0);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.signaling_nan,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::msin_ew, f.dtype, f.min_subnormal,
                    f.min_subnormal,
                    softfloat_flag_underflow | softfloat_flag_inexact);

    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, 0, 0, 0);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.negative_zero,
                    f.negative_zero, 0);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.positive_infinity,
                    f.one, 0);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.negative_infinity,
                    f.negative_one, 0);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.quiet_nan,
                    f.canonical_nan, 0);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.signaling_nan,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::mtanh_ew, f.dtype, f.min_subnormal,
                    f.min_subnormal,
                    softfloat_flag_underflow | softfloat_flag_inexact);

    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, 0,
                    f.negative_infinity, softfloat_flag_infinite);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.negative_zero,
                    f.negative_infinity, softfloat_flag_infinite);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.negative_one,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.negative_infinity,
                    f.canonical_nan, softfloat_flag_invalid);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.positive_infinity,
                    f.positive_infinity, 0);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.one, 0, 0);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.two, f.one, 0);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.min_subnormal,
                    f.log2_min_subnormal, 0);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.quiet_nan,
                    f.canonical_nan, 0);
    expect_fp_unary(ztt_opcode_t::mlog2_ew, f.dtype, f.signaling_nan,
                    f.canonical_nan, softfloat_flag_invalid);

    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, 0, f.one, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.negative_zero,
                    f.one, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.positive_infinity,
                    f.positive_infinity, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.negative_infinity,
                    0, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.one, f.two, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.negative_one,
                    f.half, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.quiet_nan,
                    f.canonical_nan, 0);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, f.dtype, f.signaling_nan,
                    f.canonical_nan, softfloat_flag_invalid);
  }
}

void test_transcendental_ieee_rounding_modes()
{
  struct rounding_case_t {
    ztt_fp_trans_op op;
    unsigned bits;
    unsigned exponent;
    uint64_t input;
    uint64_t expected[4];
  };
  const rounding_case_t cases[] = {
    {ztt_fp_trans_op::sin, 16, 5, 0x3c00,
     {0x3abb, 0x3abb, 0x3abb, 0x3abc}},
    {ztt_fp_trans_op::tanh, 16, 5, 0x3c00,
     {0x3a18, 0x3a17, 0x3a17, 0x3a18}},
    {ztt_fp_trans_op::log2, 16, 5, 0x4200,
     {0x3e57, 0x3e57, 0x3e57, 0x3e58}},
    {ztt_fp_trans_op::exp2, 16, 5, 0x3800,
     {0x3da8, 0x3da8, 0x3da8, 0x3da9}},
    {ztt_fp_trans_op::sin, 16, 8, 0x3f80,
     {0x3f57, 0x3f57, 0x3f57, 0x3f58}},
    {ztt_fp_trans_op::tanh, 16, 8, 0x3f80,
     {0x3f43, 0x3f42, 0x3f42, 0x3f43}},
    {ztt_fp_trans_op::log2, 16, 8, 0x4040,
     {0x3fcb, 0x3fca, 0x3fca, 0x3fcb}},
    {ztt_fp_trans_op::exp2, 16, 8, 0x3f00,
     {0x3fb5, 0x3fb5, 0x3fb5, 0x3fb6}},
    {ztt_fp_trans_op::sin, 32, 8, 0x3f800000,
     {0x3f576aa4, 0x3f576aa4, 0x3f576aa4, 0x3f576aa5}},
    {ztt_fp_trans_op::tanh, 32, 8, 0x3f800000,
     {0x3f42f7d6, 0x3f42f7d5, 0x3f42f7d5, 0x3f42f7d6}},
    {ztt_fp_trans_op::log2, 32, 8, 0x40400000,
     {0x3fcae00d, 0x3fcae00d, 0x3fcae00d, 0x3fcae00e}},
    {ztt_fp_trans_op::exp2, 32, 8, 0x3f000000,
     {0x3fb504f3, 0x3fb504f3, 0x3fb504f3, 0x3fb504f4}},
    {ztt_fp_trans_op::sin, 64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3feaed548f090cee), UINT64_C(0x3feaed548f090cee),
      UINT64_C(0x3feaed548f090cee), UINT64_C(0x3feaed548f090cef)}},
    {ztt_fp_trans_op::tanh, 64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3fe85efab514f394), UINT64_C(0x3fe85efab514f394),
      UINT64_C(0x3fe85efab514f394), UINT64_C(0x3fe85efab514f395)}},
    {ztt_fp_trans_op::log2, 64, 11, UINT64_C(0x4008000000000000),
     {UINT64_C(0x3ff95c01a39fbd68), UINT64_C(0x3ff95c01a39fbd68),
      UINT64_C(0x3ff95c01a39fbd68), UINT64_C(0x3ff95c01a39fbd69)}},
    {ztt_fp_trans_op::exp2, 64, 11, UINT64_C(0x3fe0000000000000),
     {UINT64_C(0x3ff6a09e667f3bcd), UINT64_C(0x3ff6a09e667f3bcc),
      UINT64_C(0x3ff6a09e667f3bcc), UINT64_C(0x3ff6a09e667f3bcd)}},
  };

  for (const auto& c : cases)
    for (unsigned mode = 0; mode < 4; ++mode)
      expect_transcendental(c.op,
        standard_float(c.bits, c.exponent, mode), c.input,
        c.expected[mode], softfloat_flag_inexact);
}

// Independent fixed-golden ULP validation for the five transcendental
// instructions.  The expected result bits are embedded test data: they are
// not calculated by ztt_fp_transcendental(), CORE-MATH, or another function
// in the implementation under test at runtime.  Every case is finite and
// positive, so adjacent encodings are adjacent representable values and their
// unsigned encoding distance is the distance in destination-format ulps from
// the correctly rounded golden result.
//
// Exact agreement with the golden result is a zero-ulp distance from that
// rounded oracle.  It establishes the documented real-result error bound for
// these samples: at most 0.5 ulp under RNE and less than 1 ulp under each
// directed rounding mode.
void test_transcendental_independent_ulp_goldens()
{
  struct ulp_case_t {
    ztt_opcode_t opcode;
    unsigned bits;
    unsigned exponent;
    uint64_t input;
    uint64_t expected[4];
  };

  const ulp_case_t cases[] = {
    {ztt_opcode_t::mcos_ew, 16, 5, 0x3c00,
     {0x3853, 0x3852, 0x3852, 0x3853}},
    {ztt_opcode_t::msin_ew, 16, 5, 0x3c00,
     {0x3abb, 0x3abb, 0x3abb, 0x3abc}},
    {ztt_opcode_t::mtanh_ew, 16, 5, 0x3c00,
     {0x3a18, 0x3a17, 0x3a17, 0x3a18}},
    {ztt_opcode_t::mlog2_ew, 16, 5, 0x4200,
     {0x3e57, 0x3e57, 0x3e57, 0x3e58}},
    {ztt_opcode_t::mexp2_ew, 16, 5, 0x3800,
     {0x3da8, 0x3da8, 0x3da8, 0x3da9}},

    {ztt_opcode_t::mcos_ew, 16, 8, 0x3f80,
     {0x3f0a, 0x3f0a, 0x3f0a, 0x3f0b}},
    {ztt_opcode_t::msin_ew, 16, 8, 0x3f80,
     {0x3f57, 0x3f57, 0x3f57, 0x3f58}},
    {ztt_opcode_t::mtanh_ew, 16, 8, 0x3f80,
     {0x3f43, 0x3f42, 0x3f42, 0x3f43}},
    {ztt_opcode_t::mlog2_ew, 16, 8, 0x4040,
     {0x3fcb, 0x3fca, 0x3fca, 0x3fcb}},
    {ztt_opcode_t::mexp2_ew, 16, 8, 0x3f00,
     {0x3fb5, 0x3fb5, 0x3fb5, 0x3fb6}},

    {ztt_opcode_t::mcos_ew, 32, 8, UINT64_C(0x3f800000),
     {UINT64_C(0x3f0a5140), UINT64_C(0x3f0a5140),
      UINT64_C(0x3f0a5140), UINT64_C(0x3f0a5141)}},
    {ztt_opcode_t::msin_ew, 32, 8, UINT64_C(0x3f800000),
     {UINT64_C(0x3f576aa4), UINT64_C(0x3f576aa4),
      UINT64_C(0x3f576aa4), UINT64_C(0x3f576aa5)}},
    {ztt_opcode_t::mtanh_ew, 32, 8, UINT64_C(0x3f800000),
     {UINT64_C(0x3f42f7d6), UINT64_C(0x3f42f7d5),
      UINT64_C(0x3f42f7d5), UINT64_C(0x3f42f7d6)}},
    {ztt_opcode_t::mlog2_ew, 32, 8, UINT64_C(0x40400000),
     {UINT64_C(0x3fcae00d), UINT64_C(0x3fcae00d),
      UINT64_C(0x3fcae00d), UINT64_C(0x3fcae00e)}},
    {ztt_opcode_t::mexp2_ew, 32, 8, UINT64_C(0x3f000000),
     {UINT64_C(0x3fb504f3), UINT64_C(0x3fb504f3),
      UINT64_C(0x3fb504f3), UINT64_C(0x3fb504f4)}},

    {ztt_opcode_t::mcos_ew, 64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3fe14a280fb5068c), UINT64_C(0x3fe14a280fb5068b),
      UINT64_C(0x3fe14a280fb5068b), UINT64_C(0x3fe14a280fb5068c)}},
    {ztt_opcode_t::msin_ew, 64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3feaed548f090cee), UINT64_C(0x3feaed548f090cee),
      UINT64_C(0x3feaed548f090cee), UINT64_C(0x3feaed548f090cef)}},
    {ztt_opcode_t::mtanh_ew, 64, 11, UINT64_C(0x3ff0000000000000),
     {UINT64_C(0x3fe85efab514f394), UINT64_C(0x3fe85efab514f394),
      UINT64_C(0x3fe85efab514f394), UINT64_C(0x3fe85efab514f395)}},
    {ztt_opcode_t::mlog2_ew, 64, 11, UINT64_C(0x4008000000000000),
     {UINT64_C(0x3ff95c01a39fbd68), UINT64_C(0x3ff95c01a39fbd68),
      UINT64_C(0x3ff95c01a39fbd68), UINT64_C(0x3ff95c01a39fbd69)}},
    {ztt_opcode_t::mexp2_ew, 64, 11, UINT64_C(0x3fe0000000000000),
     {UINT64_C(0x3ff6a09e667f3bcd), UINT64_C(0x3ff6a09e667f3bcc),
      UINT64_C(0x3ff6a09e667f3bcc), UINT64_C(0x3ff6a09e667f3bcd)}},
  };

  uint64_t maximum_ulp_distance = 0;
  unsigned checked = 0;
  for (const auto& c : cases)
    for (unsigned mode = 0; mode < 4; ++mode) {
      fixture_t f;
      const uint32_t dtype = standard_float(c.bits, c.exponent, mode);
      f.set_m(2, dtype, 0);
      f.set_m(4, dtype, c.input);
      execute_ztt(f.proc, c.opcode, operands(2, 4));

      std::vector<elem_t> results;
      assert(f.proc.ZTU.read_m(2, results));
      assert(!results.empty());
      for (elem_t result : results) {
        const uint64_t actual = uint64_t(result);
        const uint64_t expected = c.expected[mode];
        const uint64_t ulp_distance = actual >= expected
                                    ? actual - expected : expected - actual;
        maximum_ulp_distance = std::max(maximum_ulp_distance, ulp_distance);
        assert(ulp_distance == 0);
      }
      assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
      ++checked;
    }

  // Five functions x four formats x four supported rounding modes.
  assert(checked == 80);
  assert(maximum_ulp_distance == 0);
}

void test_exp2_format_boundaries()
{
  struct boundary_case_t {
    uint32_t dtype;
    uint64_t overflow_input;
    uint64_t infinity;
    uint64_t below_min_input;
    uint64_t min_input;
    uint64_t min_subnormal;
  };
  const boundary_case_t cases[] = {
    {fp16(0), 0x4c00, 0x7c00, 0xce40, 0xce00, 0x0001},
    {bf16(0), 0x4300, 0x7f80, 0xc306, 0xc305, 0x0001},
    {fp32(0), 0x43000000, 0x7f800000, 0xc3160000, 0xc3150000, 0x00000001},
    {fp64(0), UINT64_C(0x4090000000000000),
     UINT64_C(0x7ff0000000000000), UINT64_C(0xc090cc0000000000),
     UINT64_C(0xc090c80000000000), UINT64_C(0x0000000000000001)},
  };
  for (const auto& c : cases) {
    expect_fp_unary(ztt_opcode_t::mexp2_ew, c.dtype, c.overflow_input,
                    c.infinity,
                    softfloat_flag_overflow | softfloat_flag_inexact);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, c.dtype, c.below_min_input, 0,
                    softfloat_flag_underflow | softfloat_flag_inexact);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, c.dtype, c.min_input,
                    c.min_subnormal, 0);
  }
}

void test_exp2_fp32_directed_boundaries()
{
  const uint64_t overflow_result[] = {
    0x7f800000, 0x7f7fffff, 0x7f7fffff, 0x7f800000,
  };
  const uint64_t underflow_result[] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000001,
  };
  for (unsigned mode = 0; mode < 4; ++mode) {
    const uint32_t dtype = fp32(mode);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0x43000000,
                    overflow_result[mode],
                    softfloat_flag_overflow | softfloat_flag_inexact);
    expect_fp_unary(ztt_opcode_t::mexp2_ew, dtype, 0xc3160000,
                    underflow_result[mode],
                    softfloat_flag_underflow | softfloat_flag_inexact);
  }
}

void test_transcendental_non_ieee_rounding_is_unsupported()
{
  const ztt_opcode_t opcodes[] = {
    ztt_opcode_t::mcos_ew,
    ztt_opcode_t::msin_ew,
    ztt_opcode_t::mtanh_ew,
    ztt_opcode_t::mlog2_ew,
    ztt_opcode_t::mexp2_ew,
  };
  struct format_case_t {
    unsigned bits;
    unsigned exponent;
    uint64_t input;
    uint64_t sentinel;
  };
  const format_case_t formats[] = {
    {16, 5, 0x3c00, 0x4200},
    {16, 8, 0x3f80, 0x4040},
    {32, 8, 0x3f800000, 0x40400000},
    {64, 11, UINT64_C(0x3ff0000000000000),
     UINT64_C(0x4008000000000000)},
  };
  for (ztt_opcode_t opcode : opcodes)
    for (const auto& c : formats)
      for (unsigned mode : {4u, 5u}) {
        fixture_t f;
        const uint32_t dtype = standard_float(c.bits, c.exponent, mode);
        f.set_m(2, dtype, c.sentinel);
        f.set_m(4, dtype, c.input);
        f.proc.ZTU.write_amestatus(0);
        f.proc.ZTU.write_amefflags(softfloat_flag_inexact);
        execute_ztt(f.proc, opcode, operands(2, 4));
        assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
        f.expect_m(2, c.sentinel);
        assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
      }
}

void test_transcendental_fp8_is_unsupported()
{
  const ztt_opcode_t unary_opcodes[] = {
    ztt_opcode_t::mcos_ew,
    ztt_opcode_t::msin_ew,
    ztt_opcode_t::mtanh_ew,
    ztt_opcode_t::mlog2_ew,
    ztt_opcode_t::mexp2_ew,
  };
  const uint32_t formats[] = {fp8e4m3(0), fp8e5m2(0)};
  for (ztt_opcode_t opcode : unary_opcodes)
    for (uint32_t dtype : formats) {
      fixture_t f;
      f.set_m(2, dtype, 0x40);
      f.set_m(4, dtype, dtype == fp8e4m3(0) ? 0x38 : 0x3c);
      f.proc.ZTU.write_amestatus(0);
      f.proc.ZTU.write_amefflags(softfloat_flag_inexact);
      execute_ztt(f.proc, opcode, operands(2, 4));
      assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
      f.expect_m(2, 0x40);
      assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
    }
}

void expect_transcendental_state(fixture_t& fixture, const ztt_unit_t& before,
                                 unsigned destination_span = 0)
{
  const auto& state = fixture.proc.ZTU;
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
    assert(state.m_datatype(reg) == before.m_datatype(reg));
    if (reg < 2 || reg >= 2 + destination_span)
      assert(state.m_register(reg) == before.m_register(reg));
  }
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    assert(state.acc_datatype(reg) == before.acc_datatype(reg));
    assert(state.accumulator(reg) == before.accumulator(reg));
  }
  assert(state.amefflags() == before.amefflags());
  assert(state.amexsat() == before.amexsat());
  assert(state.amestype() == before.amestype());
  assert(state.owned() == before.owned());
}

void seed_transcendental_state(fixture_t& fixture)
{
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg)
    fixture.set_m(reg, kI32, 0x12340000 + reg);
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    fixture.set_acc(reg, kI32, 0);
    assert(fixture.proc.ZTU.write_acc_as(reg, 128,
      std::vector<elem_t>(ztt::kNumElements, ~elem_t(reg))));
  }
  fixture.proc.ZTU.write_amefflags(21);
  fixture.proc.ZTU.write_amexsat(1);
}

void test_setter_to_transcendental_support_matrix()
{
  struct format_case_t {
    uint32_t dtype;
    uint64_t one;
    uint64_t four;
    unsigned span;
    bool transcendental;
  };
  const format_case_t formats[] = {
    {fp8e4m3(0), 0x38, 0x48, 1, false},
    {fp8e5m2(0), 0x3c, 0x44, 1, false},
    {fp16(0), 0x3c00, 0x4400, 1, true},
    {bf16(0), 0x3f80, 0x4080, 1, true},
    {fp32(0), 0x3f800000, 0x40800000, 1, true},
    {fp64(0), UINT64_C(0x3ff0000000000000), UINT64_C(0x4010000000000000), 2, true},
  };
  const ztt_opcode_t opcodes[] = {
    ztt_opcode_t::msin_ew, ztt_opcode_t::mcos_ew,
    ztt_opcode_t::mtanh_ew, ztt_opcode_t::mlog2_ew, ztt_opcode_t::mexp2_ew,
    ztt_opcode_t::mlog2sub_ew, ztt_opcode_t::msublog2_ew,
    ztt_opcode_t::mlog2sub_ew_x, ztt_opcode_t::msublog2_ew_x,
  };
  for (const auto& format : formats)
    for (unsigned mode = 0; mode < 6; ++mode)
      for (ztt_opcode_t opcode : opcodes) {
        fixture_t fixture;
        seed_transcendental_state(fixture);
        auto& proc = fixture.proc;
        const uint32_t dtype = format.dtype | (mode << 22);
        proc.get_state()->XPR.write(8, UINT64_C(0xdeadbeef00000000) | dtype);
        for (unsigned base : {2u, 4u, 6u}) {
          execute_ztt(proc, ztt_opcode_t::msettyp, operands(base, 8));
          assert(proc.ZTU.amestatus() == 0);
          assert(proc.ZTU.m_datatype(base) == dtype);
          for (unsigned reg = base; reg < base + format.span; ++reg)
            assert(proc.ZTU.m_register(reg) == ztt_unit_t::m_register_t{});
        }
        execute_ztt(proc, ztt_opcode_t::asettyp, operands(1, 8));
        assert(proc.ZTU.amestatus() == 0);
        assert(proc.ZTU.acc_datatype(1) == dtype);
        assert(proc.ZTU.accumulator(1) == ztt_unit_t::accumulator_t{});
        assert(proc.ZTU.amefflags() == 21);
        assert(proc.ZTU.amexsat() == 1);
        const bool scalar = opcode == ztt_opcode_t::mlog2sub_ew_x ||
                            opcode == ztt_opcode_t::msublog2_ew_x;
        const bool compound = scalar || opcode == ztt_opcode_t::mlog2sub_ew ||
                              opcode == ztt_opcode_t::msublog2_ew;
        const auto fill = [&](unsigned base, uint64_t value) {
          assert(proc.ZTU.write_m_as(base, dtype,
            std::vector<elem_t>(ztt_unit_t::element_count(dtype), value)));
        };
        fill(2, format.four);
        fill(4, compound ? format.four : opcode == ztt_opcode_t::mlog2_ew ? format.one : 0);
        fill(6, scalar ? format.four : format.one);
        proc.ZTU.write_amestype(dtype);
        proc.get_state()->XPR.write(4, format.one);
        const ztt_unit_t before = proc.ZTU;
        const auto old_xpr = proc.get_state()->XPR;
        execute_ztt(proc, opcode, operands(2, 4, compound ? 6 : 0));
        const bool supported = format.transcendental && mode < 4;
        assert(proc.ZTU.amestatus() == (supported ? 0 : ztt::kAmestatusUn));
        expect_transcendental_state(fixture, before, supported ? format.span : 0);
        if (supported) {
          uint64_t expected = compound || opcode == ztt_opcode_t::mcos_ew ||
            opcode == ztt_opcode_t::mexp2_ew ? format.one : 0;
          if (opcode == ztt_opcode_t::msublog2_ew || opcode == ztt_opcode_t::msublog2_ew_x)
            expected |= UINT64_C(1) << ((dtype & 255) - 1);
          fixture.expect_m(2, expected);
        }
        for (unsigned reg = 0; reg < 32; ++reg)
          assert(proc.get_state()->XPR[reg] == old_xpr[reg]);
      }
}

void test_transcendental_mixed_and_unknown_tuples()
{
  const uint32_t formats[] = {
    fp8e4m3(0), fp8e5m2(0), fp16(0), bf16(0), fp32(0), fp64(0), fp32(1),
    0, 0x80000020u, fp32(0) | (1u << 9),
  };
  for (ztt_opcode_t opcode : {ztt_opcode_t::msin_ew, ztt_opcode_t::mcos_ew,
       ztt_opcode_t::mtanh_ew, ztt_opcode_t::mlog2_ew, ztt_opcode_t::mexp2_ew})
    for (uint32_t destination : formats)
      for (uint32_t source : formats) {
        if (destination == source)
          continue;
        fixture_t fixture;
        seed_transcendental_state(fixture);
        assert(fixture.proc.ZTU.set_m_datatype(30, destination));
        assert(fixture.proc.ZTU.set_m_datatype(4, source));
        const ztt_unit_t before = fixture.proc.ZTU;
        execute_ztt(fixture.proc, opcode, operands(30, 4));
        assert(fixture.proc.ZTU.amestatus() == ztt::kAmestatusUn);
        expect_transcendental_state(fixture, before);
      }
}

void test_compound_log2_uses_transcendental_backend()
{
  const uint32_t dtype = fp32(0);
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(4, dtype, 0xc0800000); // -4
    f.set_m(6, dtype, 0x3f800000); // 1
    execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew, operands(2, 4, 6));
    f.expect_m(2, 0x3f800000); // log2(abs(-4)) - 1 = 1
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(4, dtype, 0xc0800000); // -4
    f.set_m(6, dtype, 0x40400000); // 3
    execute_ztt(f.proc, ztt_opcode_t::msublog2_ew, operands(2, 4, 6));
    f.expect_m(2, 0x3f800000); // 3 - log2(abs(-4)) = 1
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(6, dtype, 0xc0800000); // -4
    f.proc.ZTU.write_amestype(dtype);
    f.proc.get_state()->XPR.write(4, 0x3f800000); // 1
    execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew_x, operands(2, 4, 6));
    f.expect_m(2, 0x3f800000); // log2(abs(-4)) - 1 = 1
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(6, dtype, 0xc0800000); // -4
    f.proc.ZTU.write_amestype(dtype);
    f.proc.get_state()->XPR.write(4, 0x40400000); // 3
    execute_ztt(f.proc, ztt_opcode_t::msublog2_ew_x, operands(2, 4, 6));
    f.expect_m(2, 0x3f800000); // 3 - log2(abs(-4)) = 1
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(4, dtype, 0xff812345); // negative sNaN, abs remains sNaN
    f.set_m(6, dtype, 0x3f800000);
    execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew, operands(2, 4, 6));
    f.expect_m(2, 0x7fc00000);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_invalid);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(4, dtype, 0xc0800000);
    f.set_m(6, dtype, 0x7fc12345); // qNaN bias
    execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew, operands(2, 4, 6));
    f.expect_m(2, 0x7fc00000);
    assert(f.proc.ZTU.amefflags() == 0);
  }
  {
    fixture_t f;
    f.set_m(2, dtype, 0);
    f.set_m(4, dtype, 0x80000000); // abs(-0) = +0
    f.set_m(6, dtype, 0x3f800000);
    execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew, operands(2, 4, 6));
    f.expect_m(2, 0xff800000);
    assert(f.proc.ZTU.amefflags() == softfloat_flag_infinite);
  }
}

void test_compound_log2_non_ieee_rounding_is_unsupported()
{
  const ztt_opcode_t matrix_opcodes[] = {
    ztt_opcode_t::mlog2sub_ew,
    ztt_opcode_t::msublog2_ew,
  };
  const ztt_opcode_t scalar_opcodes[] = {
    ztt_opcode_t::mlog2sub_ew_x,
    ztt_opcode_t::msublog2_ew_x,
  };
  struct format_case_t {
    unsigned bits;
    unsigned exponent;
    uint64_t one;
    uint64_t four;
    uint64_t sentinel;
  };
  const format_case_t formats[] = {
    {16, 5, 0x3c00, 0x4400, 0x4200},
    {16, 8, 0x3f80, 0x4080, 0x4040},
    {32, 8, 0x3f800000, 0x40800000, 0x40400000},
    {64, 11, UINT64_C(0x3ff0000000000000),
     UINT64_C(0x4010000000000000), UINT64_C(0x4008000000000000)},
  };
  for (const auto& c : formats)
    for (unsigned mode : {4u, 5u}) {
      const uint32_t dtype = standard_float(c.bits, c.exponent, mode);
      for (ztt_opcode_t opcode : matrix_opcodes) {
        fixture_t f;
        f.set_m(2, dtype, c.sentinel);
        f.set_m(4, dtype, c.four);
        f.set_m(6, dtype, c.one);
        f.proc.ZTU.write_amestatus(0);
        f.proc.ZTU.write_amefflags(softfloat_flag_inexact);
        execute_ztt(f.proc, opcode, operands(2, 4, 6));
        assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
        f.expect_m(2, c.sentinel);
        assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
      }
      for (ztt_opcode_t opcode : scalar_opcodes) {
        fixture_t f;
        f.set_m(2, dtype, c.sentinel);
        f.set_m(6, dtype, c.four);
        f.proc.ZTU.write_amestype(dtype);
        f.proc.get_state()->XPR.write(4, c.one);
        f.proc.ZTU.write_amestatus(0);
        f.proc.ZTU.write_amefflags(softfloat_flag_inexact);
        execute_ztt(f.proc, opcode, operands(2, 4, 6));
        assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
        f.expect_m(2, c.sentinel);
        assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
      }
    }
}

void test_compound_log2_fp8_is_unsupported()
{
  const ztt_opcode_t opcodes[] = {
    ztt_opcode_t::mlog2sub_ew,
    ztt_opcode_t::msublog2_ew,
  };
  const uint32_t formats[] = {fp8e4m3(0), fp8e5m2(0)};
  for (ztt_opcode_t opcode : opcodes)
    for (uint32_t dtype : formats) {
      fixture_t f;
      const elem_t one = dtype == fp8e4m3(0) ? 0x38 : 0x3c;
      f.set_m(2, dtype, 0x40);
      f.set_m(4, dtype, one);
      f.set_m(6, dtype, one);
      f.proc.ZTU.write_amestatus(0);
      f.proc.ZTU.write_amefflags(softfloat_flag_inexact);
      execute_ztt(f.proc, opcode, operands(2, 4, 6));
      assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
      f.expect_m(2, 0x40);
      assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
    }
}

struct ieee_format_case_t {
  unsigned bits;
  unsigned exponent;
  uint64_t zero;
  uint64_t negative_zero;
  uint64_t half;
  uint64_t one;
  uint64_t negative_one;
  uint64_t one_point_five;
  uint64_t two;
  uint64_t two_point_five;
  uint64_t half_ulp_at_one;
  uint64_t next_one;
  uint64_t previous_one;
  uint64_t min_subnormal;
  uint64_t max_finite;
  uint64_t infinity;
  uint64_t quiet_nan;
  uint64_t signaling_nan;
};

const ieee_format_case_t kIeeeFormats[] = {
  {16, 5, 0x0000, 0x8000, 0x3800, 0x3c00, 0xbc00, 0x3e00,
   0x4000, 0x4100, 0x1000, 0x3c01, 0x3bff, 0x0001, 0x7bff,
   0x7c00, 0x7e00, 0x7d00},
  {32, 8, 0x00000000, 0x80000000, 0x3f000000, 0x3f800000,
   0xbf800000, 0x3fc00000, 0x40000000, 0x40200000, 0x33800000,
   0x3f800001, 0x3f7fffff, 0x00000001, 0x7f7fffff, 0x7f800000,
   0x7fc00000, 0x7f800001},
  {64, 11, UINT64_C(0x0000000000000000),
   UINT64_C(0x8000000000000000), UINT64_C(0x3fe0000000000000),
   UINT64_C(0x3ff0000000000000), UINT64_C(0xbff0000000000000),
   UINT64_C(0x3ff8000000000000), UINT64_C(0x4000000000000000),
   UINT64_C(0x4004000000000000), UINT64_C(0x3ca0000000000000),
   UINT64_C(0x3ff0000000000001), UINT64_C(0x3fefffffffffffff),
   UINT64_C(0x0000000000000001), UINT64_C(0x7fefffffffffffff),
   UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff8000000000000),
   UINT64_C(0x7ff0000000000001)},
};

uint32_t ieee_dtype(const ieee_format_case_t& format, unsigned rounding)
{
  return standard_float(format.bits, format.exponent, rounding);
}

uint32_t signed_integer_dtype(unsigned bits, unsigned rounding = 0)
{
  return (1u << 30) | (rounding << 27) | bits;
}

void expect_binary_ieee(ztt_opcode_t opcode, uint32_t dtype, uint64_t source1,
                        uint64_t source2)
{
  const char op = opcode == ztt_opcode_t::madd_ew ? '+'
                : opcode == ztt_opcode_t::msub_ew ? '-' : '*';
  const uint64_t lhs = opcode == ztt_opcode_t::msub_ew ? source2 : source1;
  const uint64_t rhs = opcode == ztt_opcode_t::msub_ew ? source1 : source2;
  const ztt_fp_result_t expected =
    softfloat_binary_reference(dtype, lhs, rhs, op);
  fixture_t f;
  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, source1);
  f.set_m(6, dtype, source2);
  execute_ztt(f.proc, opcode, operands(2, 4, 6));
  f.expect_m(2, expected.bits);
  assert(f.proc.ZTU.amefflags() == expected.flags);
}

// IEEE 754 basic operations: all standard Ztt storage formats, all five Ztt
// IEEE rounding modes, finite boundaries, signed zero, subnormal, infinity,
// and both quiet and signaling NaNs.  Berkeley SoftFloat supplies both the
// result-bit and exception-flag oracle.
void test_basic_arithmetic_ieee_matrix()
{
  for (const auto& c : kIeeeFormats)
    for (unsigned mode = 0; mode <= 4; ++mode) {
      const uint32_t dtype = ieee_dtype(c, mode);
      const std::pair<uint64_t, uint64_t> add_cases[] = {
        {c.one, c.two}, {c.one, c.half_ulp_at_one},
        {c.negative_zero, c.zero}, {c.min_subnormal, c.min_subnormal},
        {c.max_finite, c.max_finite}, {c.infinity, c.infinity ^
          (UINT64_C(1) << (c.bits - 1))},
        {c.quiet_nan, c.one}, {c.signaling_nan, c.one},
      };
      for (const auto& v : add_cases)
        expect_binary_ieee(ztt_opcode_t::madd_ew, dtype, v.first, v.second);

      // msub.ew computes source2 - source1.
      const std::pair<uint64_t, uint64_t> sub_cases[] = {
        {c.half_ulp_at_one, c.one}, {c.zero, c.negative_zero},
        {c.min_subnormal, c.min_subnormal}, {c.max_finite ^
          (UINT64_C(1) << (c.bits - 1)), c.max_finite},
        {c.infinity, c.infinity}, {c.quiet_nan, c.one},
        {c.signaling_nan, c.one},
      };
      for (const auto& v : sub_cases)
        expect_binary_ieee(ztt_opcode_t::msub_ew, dtype, v.first, v.second);

      const std::pair<uint64_t, uint64_t> mul_cases[] = {
        {c.one, c.two}, {c.next_one, c.previous_one},
        {c.negative_zero, c.two}, {c.min_subnormal, c.half},
        {c.max_finite, c.two}, {c.zero, c.infinity},
        {c.quiet_nan, c.one}, {c.signaling_nan, c.one},
      };
      for (const auto& v : mul_cases)
        expect_binary_ieee(ztt_opcode_t::mmul_ew, dtype, v.first, v.second);
    }
}

void expect_fma_ieee(uint32_t dtype, uint64_t old_dest, uint64_t addend,
                     uint64_t factor)
{
  // mmuladd.ew is source1 + source2 * old-destination.
  const ztt_fp_result_t expected =
    softfloat_fma_reference(dtype, factor, old_dest, addend);
  fixture_t f;
  f.set_m(2, dtype, old_dest);
  f.set_m(4, dtype, addend);
  f.set_m(6, dtype, factor);
  execute_ztt(f.proc, ztt_opcode_t::mmuladd_ew, operands(2, 4, 6));
  f.expect_m(2, expected.bits);
  assert(f.proc.ZTU.amefflags() == expected.flags);
}

void test_fma_ieee_matrix()
{
  for (const auto& c : kIeeeFormats)
    for (unsigned mode = 0; mode <= 4; ++mode) {
      const uint32_t dtype = ieee_dtype(c, mode);
      expect_fma_ieee(dtype, c.half_ulp_at_one, c.one, c.one);
      expect_fma_ieee(dtype, c.next_one, c.negative_one, c.previous_one);
      expect_fma_ieee(dtype, c.max_finite, c.zero, c.two);
      expect_fma_ieee(dtype, c.min_subnormal, c.zero, c.half);
      expect_fma_ieee(dtype, c.zero, c.one, c.infinity);
      expect_fma_ieee(dtype, c.infinity, c.infinity ^
                      (UINT64_C(1) << (c.bits - 1)), c.one);
      expect_fma_ieee(dtype, c.one, c.quiet_nan, c.one);
      expect_fma_ieee(dtype, c.one, c.signaling_nan, c.one);

      if (mode == 0) {
        const ztt_fp_result_t fused = softfloat_fma_reference(
          dtype, c.previous_one, c.next_one, c.negative_one);
        const ztt_fp_result_t product = softfloat_binary_reference(
          dtype, c.previous_one, c.next_one, '*');
        const ztt_fp_result_t non_fused = softfloat_binary_reference(
          dtype, product.bits, c.negative_one, '+');
        assert(fused.bits != non_fused.bits);
        assert(fused.bits != c.zero);
        assert(non_fused.bits == c.zero);
      }
    }
}

void test_sqrt_ieee_matrix()
{
  for (const auto& c : kIeeeFormats)
    for (unsigned mode = 0; mode <= 4; ++mode) {
      const uint32_t dtype = ieee_dtype(c, mode);
      const uint64_t inputs[] = {
        c.one, c.two, c.min_subnormal, c.negative_one, c.zero,
        c.negative_zero, c.infinity, c.quiet_nan, c.signaling_nan,
      };
      for (uint64_t input : inputs) {
        const ztt_fp_result_t expected =
          softfloat_sqrt_reference(dtype, input);
        expect_fp_unary(ztt_opcode_t::msqrt_ew, dtype, input,
                        expected.bits, expected.flags);
      }
    }
}

void expect_float_conversion(uint32_t source_dtype, uint32_t dest_dtype,
                             uint64_t input)
{
  const ztt_fp_result_t expected =
    softfloat_convert_reference(source_dtype, dest_dtype, input);
  const std::size_t instruction_squares = std::max(
    ztt_unit_t::square_count(source_dtype),
    ztt_unit_t::square_count(dest_dtype));
  fixture_t f;
  f.set_m_span(0, dest_dtype, 0, instruction_squares);
  f.set_m_span(8, source_dtype, input, instruction_squares);
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(0, 8));
  f.expect_m_span(0, dest_dtype, expected.bits, instruction_squares);
  assert(f.proc.ZTU.amefflags() == expected.flags);
}

void test_round_to_integral_ieee()
{
  struct rounding_opcode_t {
    ztt_opcode_t opcode;
    uint_fast8_t rounding;
  };
  const rounding_opcode_t operations[] = {
    {ztt_opcode_t::mfrintm_ew, softfloat_round_min},
    {ztt_opcode_t::mfrintn_ew, softfloat_round_near_even},
    {ztt_opcode_t::mfrintp_ew, softfloat_round_max},
    {ztt_opcode_t::mfrintz_ew, softfloat_round_minMag},
  };
  for (const auto& c : kIeeeFormats) {
    const uint32_t dtype = ieee_dtype(c, 0);
    const uint64_t sign = UINT64_C(1) << (c.bits - 1);
    const uint64_t inputs[] = {
      c.one_point_five, c.two_point_five,
      c.one_point_five | sign, c.two_point_five | sign,
      c.zero, c.negative_zero, c.infinity, c.quiet_nan, c.signaling_nan,
    };
    for (const auto& op : operations)
      for (uint64_t input : inputs) {
        const ztt_fp_result_t expected = softfloat_round_to_int_reference(
          dtype, input, op.rounding);
        expect_fp_unary(op.opcode, dtype, input, expected.bits,
                        expected.flags);
      }
  }
}

void test_conversion_ieee_matrix()
{
  for (const auto& source : kIeeeFormats)
    for (const auto& dest : kIeeeFormats) {
      if (source.bits == dest.bits)
        continue;
      for (unsigned mode = 0; mode <= 4; ++mode) {
        const uint32_t source_dtype = ieee_dtype(source, 0);
        const uint32_t dest_dtype = ieee_dtype(dest, mode);
        const uint64_t inputs[] = {
          source.zero, source.negative_zero, source.one,
          source.min_subnormal, source.max_finite, source.infinity,
          source.quiet_nan, source.signaling_nan,
        };
        for (uint64_t input : inputs)
          expect_float_conversion(source_dtype, dest_dtype, input);
      }
    }

  // Narrowing midpoints distinguish all directed modes and catch accidental
  // intermediate rounding in FP32 -> FP16 and FP64 -> FP32.
  for (unsigned mode = 0; mode <= 4; ++mode) {
    expect_float_conversion(fp32(0), fp16(mode), 0x3f801000);
    expect_float_conversion(fp64(0), fp32(mode),
                            UINT64_C(0x3ff0000010000000));
  }

  const struct int_case_t {
    const ieee_format_case_t* format;
    int64_t positive_midpoint;
  } int_cases[] = {
    {&kIeeeFormats[0], INT64_C(2049)},
    {&kIeeeFormats[1], INT64_C(16777217)},
    {&kIeeeFormats[2], INT64_C(9007199254740993)},
  };
  constexpr uint32_t i64 = (1u << 30) | 64u;
  for (const auto& c : int_cases)
    for (unsigned mode = 0; mode <= 4; ++mode)
      for (int64_t input : {c.positive_midpoint, -c.positive_midpoint}) {
        const uint32_t dest_dtype = ieee_dtype(*c.format, mode);
        const ztt_fp_result_t expected =
          softfloat_integer_to_float_reference(dest_dtype, input);
        const std::size_t squares = std::max(
          ztt_unit_t::square_count(i64),
          ztt_unit_t::square_count(dest_dtype));
        fixture_t f;
        f.set_m_span(0, dest_dtype, 0, squares);
        f.set_m_span(8, i64, uint64_t(input), squares);
        execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(0, 8));
        f.expect_m_span(0, dest_dtype, expected.bits, squares);
        assert(f.proc.ZTU.amefflags() == expected.flags);
      }

  // Float-to-integer conversion uses the destination integer rounding mode.
  // Integer mode 1 is RNE in Ztt v0.6.
  constexpr uint32_t i32_rne = (1u << 30) | (1u << 27) | 32u;
  for (const auto& c : kIeeeFormats) {
    const uint32_t source_dtype = ieee_dtype(c, 0);
    const uint64_t sign = UINT64_C(1) << (c.bits - 1);
    const struct float_int_case_t {
      uint64_t input;
      uint32_t expected;
      uint8_t flags;
    } cases[] = {
      {c.one_point_five, 2, softfloat_flag_inexact},
      {c.two_point_five, 2, softfloat_flag_inexact},
      {c.one_point_five | sign, UINT32_C(0xfffffffe),
       softfloat_flag_inexact},
      {c.two_point_five | sign, UINT32_C(0xfffffffe),
       softfloat_flag_inexact},
      {c.infinity, UINT32_C(0x7fffffff), softfloat_flag_invalid},
      {c.infinity | sign, UINT32_C(0x80000000), softfloat_flag_invalid},
      {c.quiet_nan, 0, softfloat_flag_invalid},
    };
    for (const auto& v : cases) {
      fixture_t f;
      f.set_m(0, i32_rne, 0);
      f.set_m(4, source_dtype, v.input);
      execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(0, 4));
      f.expect_m(0, v.expected);
      assert(f.proc.ZTU.amefflags() == v.flags);
    }
  }

  test_round_to_integral_ieee();
}

void test_comparison_ieee_specials()
{
  for (const auto& c : kIeeeFormats) {
    const uint32_t dtype = ieee_dtype(c, 0);
    const uint32_t result_dtype = signed_integer_dtype(c.bits);
    const elem_t true_value = ztt_unit_t::element_mask(c.bits);
    const struct compare_case_t {
      uint64_t lhs;
      uint64_t rhs;
      bool less;
      bool greater_equal;
      uint8_t flags;
    } cases[] = {
      {c.zero, c.negative_zero, false, true, 0},
      {c.negative_zero, c.zero, false, true, 0},
      {c.infinity ^ (UINT64_C(1) << (c.bits - 1)), c.infinity,
       true, false, 0},
      {c.infinity, c.max_finite, false, true, 0},
      {c.quiet_nan, c.one, false, false, 0},
      {c.signaling_nan, c.one, false, false, softfloat_flag_invalid},
    };
    for (const auto& v : cases)
      for (ztt_opcode_t opcode : {ztt_opcode_t::mcmplt_ew,
                                  ztt_opcode_t::mcmpge_ew}) {
        fixture_t f;
        f.set_m(0, result_dtype, 0);
        f.set_m(4, dtype, v.lhs);
        f.set_m(8, dtype, v.rhs);
        execute_ztt(f.proc, opcode, operands(0, 4, 8));
        const bool result = opcode == ztt_opcode_t::mcmplt_ew
                          ? v.less : v.greater_equal;
        f.expect_m(0, result ? true_value : 0);
        assert(f.proc.ZTU.amefflags() == v.flags);
      }
  }
}

void test_architectural_flags_ieee()
{
  fixture_t f;
  const uint32_t dtype = fp32(0);
  const reg_t initial_fflags = f.proc.get_csr(CSR_FFLAGS);
  assert(f.proc.get_csr(CSR_FFLAGS) == initial_fflags);

  // NX: exact midpoint rounded to FP32.
  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0x3f800000);
  f.set_m(6, dtype, 0x33800000);
  execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(2, 4, 6));
  assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);

  // UF (and NX): half the minimum subnormal.
  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0x00000001);
  f.set_m(6, dtype, 0x3f000000);
  execute_ztt(f.proc, ztt_opcode_t::mmul_ew, operands(2, 4, 6));
  assert(f.proc.ZTU.amefflags() ==
         (softfloat_flag_underflow | softfloat_flag_inexact));

  // OF (and NX), then NV, then DZ.  amefflags must retain every prior bit.
  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0x7f7fffff);
  f.set_m(6, dtype, 0x40000000);
  execute_ztt(f.proc, ztt_opcode_t::mmul_ew, operands(2, 4, 6));
  assert(f.proc.ZTU.amefflags() ==
         (softfloat_flag_overflow | softfloat_flag_underflow |
          softfloat_flag_inexact));

  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0xbf800000);
  execute_ztt(f.proc, ztt_opcode_t::msqrt_ew, operands(2, 4));
  assert(f.proc.ZTU.amefflags() ==
         (softfloat_flag_invalid | softfloat_flag_overflow |
          softfloat_flag_underflow | softfloat_flag_inexact));

  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0);
  execute_ztt(f.proc, ztt_opcode_t::mrec_ew, operands(2, 4));
  assert(f.proc.ZTU.amefflags() == ztt::kAmeFlagMask);
  assert(f.proc.get_csr(CSR_FFLAGS) == initial_fflags);

  // An exact operation cannot clear already accrued architectural flags.
  f.set_m(2, dtype, 0);
  f.set_m(4, dtype, 0x3f800000);
  f.set_m(6, dtype, 0x40000000);
  execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(2, 4, 6));
  assert(f.proc.ZTU.amefflags() == ztt::kAmeFlagMask);
  assert(f.proc.get_csr(CSR_FFLAGS) == initial_fflags);
}

} // namespace

int main()
{
  test_basic_arithmetic_ieee_matrix();
  test_fma_ieee_matrix();
  test_sqrt_ieee_matrix();
  test_conversion_ieee_matrix();
  test_comparison_ieee_specials();
  test_architectural_flags_ieee();
  test_softfloat_rounding_mapping();
  test_integer_to_float_is_direct();
  test_fp64_to_fp8_rounds_directly();
  test_fp8_exact_operation_boundaries();
  test_fp8_instruction_paths();
  test_fp8_exhaustive_against_softfloat_reference();
  test_compound_float_expressions_round_once();
  test_matrix_float_precision_contract();
  test_integer_ldexpacc_rounds_complete_expression_once();
  test_bf16_quiet_nan_compare_does_not_set_invalid();
  test_log2_compound_integer_bias();
  test_rno_is_allowed_for_non_rounding_operations();
  test_exact_ldexp_rounding_and_flags();
  test_round_pack_midpoints_and_tininess();
  test_ldexpacc_single_rounding();
  test_ldexpacc_exact_cancellation_and_invalid_infinity();
  test_exact_scale_special_values();
  test_rno_is_explicitly_unsupported();
  test_mcos_fp32_rne_results_and_flags();
  test_mcos_other_formats_rne_results_and_flags();
  test_transcendental_host_environment_is_restored();
  test_mcos_ieee_rounding_modes();
  test_mcos_fp16_underflow();
  test_transcendental_fp32_special_values();
  test_transcendental_other_formats_special_values();
  test_transcendental_ieee_rounding_modes();
  test_transcendental_independent_ulp_goldens();
  test_exp2_format_boundaries();
  test_exp2_fp32_directed_boundaries();
  test_transcendental_non_ieee_rounding_is_unsupported();
  test_transcendental_fp8_is_unsupported();
  test_setter_to_transcendental_support_matrix();
  test_transcendental_mixed_and_unknown_tuples();
  test_compound_log2_uses_transcendental_backend();
  test_compound_log2_non_ieee_rounding_is_unsupported();
  test_compound_log2_fp8_is_unsupported();
  return 0;
}
