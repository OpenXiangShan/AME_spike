// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
// Targeted regressions for the Spike-side review items reported against the
// gem5 Ztt implementation.  Items which describe gem5 scheduling metadata or
// an offline ULP certification are represented by executable Spike invariants
// and documented in the corresponding test below.

#include "cfg.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "softfloat.h"
#include "trap.h"
#include "ztt_execute.h"
#include "ztt_fp.h"
#include "ztt_state.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <sstream>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;

constexpr uint32_t fp_dtype(unsigned exponent, unsigned rounding,
                            bool infinity, unsigned bits)
{
  return (exponent << 26) | (rounding << 22) |
         (uint32_t(infinity) << 21) | (1u << 20) | (1u << 8) | bits;
}

constexpr uint32_t kU8 = 8;
constexpr uint32_t kU16 = 16;
constexpr uint32_t kU32 = 32;
constexpr uint32_t kU64 = 64;
constexpr uint32_t kU128 = 128;
constexpr uint32_t kI8 = (1u << 30) | 8u;
constexpr uint32_t kI16 = (1u << 30) | 16u;
constexpr uint32_t kI32 = (1u << 30) | 32u;
constexpr uint32_t kI64 = (1u << 30) | 64u;
constexpr uint32_t kI128 = (1u << 30) | 128u;
constexpr uint32_t kF16 = (5u << 26) | (1u << 21) | (1u << 20) |
                           (1u << 8) | 16u;
constexpr uint32_t kBF16 = (8u << 26) | (1u << 21) | (1u << 20) |
                            (1u << 8) | 16u;
constexpr uint32_t kF32 = fp_dtype(8, 0, true, 32);
constexpr uint32_t kF64 = fp_dtype(11, 0, true, 64);
constexpr uint32_t kE4M3 = fp_dtype(4, 0, false, 8);
constexpr uint32_t kE5M2 = fp_dtype(5, 0, true, 8);

insn_t operands(unsigned rd, unsigned rs1 = 0, unsigned rs2 = 0)
{
  return insn_t((insn_bits_t(rs2) << 20) |
                (insn_bits_t(rs1) << 15) |
                (insn_bits_t(rd) << 7));
}

class test_sim_t final : public simif_t {
 public:
  test_sim_t() : memory(16384, 0), wrapped(16384, 0), wrapped_high(16384, 0)
    { debug_mmu = nullptr; }

  char* addr_to_mem(reg_t address) override
  {
    // The low-address window is used to make RV32 XLEN wrapping observable;
    // the normal DRAM window is kept separate so aliasing is not accidental.
    if (address < wrapped.size())
      return reinterpret_cast<char*>(wrapped.data() + address);
    constexpr reg_t high_base = UINT64_C(0xffffffe0);
    if (address >= high_base && address - high_base < wrapped_high.size())
      return reinterpret_cast<char*>(wrapped_high.data() + address - high_base);
    if (address >= DRAM_BASE && address - DRAM_BASE < memory.size())
      return reinterpret_cast<char*>(memory.data() + address - DRAM_BASE);
    return nullptr;
  }
  bool mmio_load(reg_t, size_t, uint8_t*) override { return false; }
  bool mmio_store(reg_t, size_t, const uint8_t*) override { return false; }
  void proc_reset(unsigned) override {}
  const cfg_t& get_cfg() const override { return cfg; }
  const std::map<size_t, processor_t*>& get_harts() const override
    { return harts; }
  const char* get_symbol(uint64_t) override { return nullptr; }

  cfg_t cfg;
  std::vector<uint8_t> memory;
  std::vector<uint8_t> wrapped;
  std::vector<uint8_t> wrapped_high;
  std::map<size_t, processor_t*> harts;
};

class fixture_t {
 public:
  explicit fixture_t(const char* isa = "rv64i_ztt")
    : proc(isa, "M", &sim.cfg, &sim, 0, false, nullptr, output)
  {
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(reg_t(1) << 25);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1));
    assert(proc.ZTU.owned());
  }

  test_sim_t sim;
  std::ostringstream output;
  processor_t proc;
};

std::vector<elem_t> filled(uint32_t dtype, elem_t value)
{
  return std::vector<elem_t>(ztt_unit_t::element_count(dtype), value);
}

std::vector<elem_t> sequence(uint32_t dtype, elem_t first)
{
  std::vector<elem_t> values(ztt_unit_t::element_count(dtype));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = first + i;
  return values;
}

void set_m(processor_t& proc, unsigned reg, uint32_t dtype,
           const std::vector<elem_t>& values)
{
  assert(proc.ZTU.set_m_datatype(reg, dtype));
  assert(proc.ZTU.write_m(reg, values));
}

std::vector<elem_t> get_m(processor_t& proc, unsigned reg)
{
  std::vector<elem_t> values;
  assert(proc.ZTU.read_m(reg, values));
  return values;
}

void set_acc(processor_t& proc, unsigned reg, uint32_t dtype, elem_t value)
{
  assert(proc.ZTU.set_acc_datatype(reg, dtype));
  // Accumulators always contain one architectural square.  Narrow M
  // datatypes span multiple packed squares, so element_count(dtype) is only
  // appropriate for M-register helpers.
  assert(proc.ZTU.write_acc(
    reg, std::vector<elem_t>(ztt::kNumElements, value)));
}

void test_01_dedicated_status_contract()
{
  fixture_t f;
  f.proc.get_state()->XPR.write(4, reg_t(1) << 6);
  execute_ztt(f.proc, ztt_opcode_t::ame_acquire, operands(5, 4));
  assert(f.proc.get_state()->XPR[5] == 0x08); // BAD_DESC in rd[...]
  f.proc.get_state()->XPR.write(4, 0);
  execute_ztt(f.proc, ztt_opcode_t::ame_acquire, operands(5, 4));
  assert(f.proc.get_state()->XPR[5] == 1);
  assert(f.proc.ZTU.owned());
}

void test_02_strided_xlen_wrap()
{
  // In RV32, segment addresses are computed modulo 2^32.  Segment zero is
  // in the DRAM window; the next transposed segment wraps into low memory.
  fixture_t f("rv32i_ztt");
  set_m(f.proc, 0, kU32, sequence(kU32, 0x100));
  f.proc.get_state()->XPR.write(4, DRAM_BASE + 128);
  f.proc.get_state()->XPR.write(5, UINT32_C(0x80000040));
  execute_ztt(f.proc, ztt_opcode_t::mss_tst, operands(0, 4, 5));
  set_m(f.proc, 1, kU32, filled(kU32, 0));
  execute_ztt(f.proc, ztt_opcode_t::mls_tst, operands(1, 4, 5));
  assert(get_m(f.proc, 1) == get_m(f.proc, 0));
}

void test_03_contiguous_xlen_wrap()
{
  // A 64-byte packed-u8 transfer starts at 0xffffffe0, so its final
  // 32 bytes use addresses 0x00000000-0x0000001f in RV32.
  fixture_t f("rv32i_ztt");
  const auto values = sequence(kU8, 1);
  set_m(f.proc, 0, kU8, values);
  const reg_t base = UINT32_C(0xffffffe0);
  f.proc.get_state()->XPR.write(4, base);

  execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));
  set_m(f.proc, 1, kU8, filled(kU8, 0));
  execute_ztt(f.proc, ztt_opcode_t::mls_rm, operands(1, 4));
  assert(get_m(f.proc, 1) == values);
}

void test_03_left_shift_is_converted_at_destination_width()
{
  const uint32_t sat_u8 = (1u << 29) | 8u;
  {
    fixture_t f;
    set_m(f.proc, 0, sat_u8, filled(sat_u8, 0x80));
    set_m(f.proc, 1, sat_u8, filled(sat_u8, 1));
    set_m(f.proc, 2, sat_u8, filled(sat_u8, 0));
    execute_ztt(f.proc, ztt_opcode_t::msll_ew, operands(2, 0, 1));
    assert(get_m(f.proc, 2).front() == 0xff);
    assert(f.proc.ZTU.amexsat());
  }
  {
    fixture_t f;
    set_m(f.proc, 0, kU8, filled(kU8, 0x80));
    set_m(f.proc, 4, kU16, filled(kU16, 0));
    f.proc.get_state()->XPR.write(3, 1);
    execute_ztt(f.proc, ztt_opcode_t::msll_ew_x, operands(4, 3, 0));
    assert(get_m(f.proc, 4).front() == 0x0100);
  }
  {
    fixture_t f;
    set_m(f.proc, 0, kI32, filled(kI32, UINT32_C(0xffffffff)));
    set_m(f.proc, 1, kI32, filled(kI32, 0));
    set_m(f.proc, 2, kI32, filled(kI32, 0));
    execute_ztt(f.proc, ztt_opcode_t::msll_ew, operands(2, 0, 1));
    assert(get_m(f.proc, 2).front() == UINT32_C(0xffffffff));
    assert(f.proc.ZTU.amexsat() == 0);
  }
  {
    fixture_t f;
    set_m(f.proc, 0, kI32, filled(kI32, UINT32_C(0xffffffff)));
    set_m(f.proc, 4, kI32, filled(kI32, 0));
    f.proc.get_state()->XPR.write(3, 1);
    execute_ztt(f.proc, ztt_opcode_t::msll_ew_x, operands(4, 3, 0));
    assert(get_m(f.proc, 4).front() == UINT32_C(0xfffffffe));
    assert(f.proc.ZTU.amexsat() == 0);
  }
}

void test_mfrintn_preserves_negative_zero()
{
  fixture_t f;
  set_m(f.proc, 0, kF32, filled(kF32, 0));
  set_m(f.proc, 2, kF32, filled(kF32, UINT32_C(0xbf000000))); // -0.5
  execute_ztt(f.proc, ztt_opcode_t::mfrintn_ew, operands(0, 2));
  assert(get_m(f.proc, 0).front() == UINT32_C(0x80000000));
  assert(f.proc.ZTU.amefflags() == softfloat_flag_inexact);
}

void test_04_half_difference_operand_order()
{
  fixture_t f;
  set_m(f.proc, 0, kI32, filled(kI32, 0));
  set_m(f.proc, 2, kI32, filled(kI32, 6)); // A
  set_m(f.proc, 4, kI32, filled(kI32, 10)); // B
  execute_ztt(f.proc, ztt_opcode_t::mhdiff_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 2);

  set_m(f.proc, 0, kI32, filled(kI32, 0));
  f.proc.ZTU.write_amestype(kI32);
  f.proc.get_state()->XPR.write(5, 6); // c
  set_m(f.proc, 4, kI32, filled(kI32, 10)); // B
  execute_ztt(f.proc, ztt_opcode_t::mhdiff_ew_x, operands(0, 5, 4));
  assert(get_m(f.proc, 0).front() == 2);
}

void test_05_softfloat_format_paths()
{
  const struct format_case { uint32_t dtype; elem_t one; elem_t two; elem_t three; } cases[] = {
    {kF16, 0x3c00, 0x4000, 0x4200},
    {kBF16, 0x3f80, 0x4000, 0x4040},
    {kF32, 0x3f800000, 0x40000000, 0x40400000},
    {kF64, UINT64_C(0x3ff0000000000000), UINT64_C(0x4000000000000000),
     UINT64_C(0x4008000000000000)},
    {kE4M3, 0x38, 0x40, 0x44},
    {kE5M2, 0x3c, 0x40, 0x42},
  };
  for (const auto& c : cases) {
    fixture_t f;
    set_m(f.proc, 0, c.dtype, filled(c.dtype, 0));
    set_m(f.proc, 2, c.dtype, filled(c.dtype, c.one));
    set_m(f.proc, 4, c.dtype, filled(c.dtype, c.two));
    execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(0, 2, 4));
    assert(get_m(f.proc, 0).front() == c.three);
  }
}

void test_06_debug_move_lane_boundaries()
{
  fixture_t f;
  const struct lane_case { ztt_opcode_t to_m; ztt_opcode_t from_m; unsigned pos; uint64_t value; } cases[] = {
    {ztt_opcode_t::mmove8_m_x, ztt_opcode_t::mmove8_x_m, 15, 0xa5},
    {ztt_opcode_t::mmove16_m_x, ztt_opcode_t::mmove16_x_m, 7, 0xa55a},
    {ztt_opcode_t::mmove32_m_x, ztt_opcode_t::mmove32_x_m, 3, 0xa55aa55a},
    {ztt_opcode_t::mmove64_m_x, ztt_opcode_t::mmove64_x_m, 1,
     UINT64_C(0x0123456789abcdef)},
  };
  for (const auto& c : cases) {
    f.proc.get_state()->XPR.write(4, c.value);
    f.proc.get_state()->XPR.write(5, c.pos);
    execute_ztt(f.proc, c.to_m, operands(0, 4, 5));
    execute_ztt(f.proc, c.from_m, operands(6, 0, 5));
    assert(f.proc.get_state()->XPR[6] == c.value);
  }
}

void test_07_one_register_memory_is_ordered()
{
  fixture_t f;
  const reg_t address = DRAM_BASE + 256;
  set_m(f.proc, 0, kU32, sequence(kU32, 1));
  f.proc.get_state()->XPR.write(4, address);
  execute_ztt(f.proc, ztt_opcode_t::mss_1r, operands(0, 4));
  set_m(f.proc, 1, kU32, filled(kU32, 0));
  execute_ztt(f.proc, ztt_opcode_t::mls_1r, operands(1, 4));
  assert(get_m(f.proc, 1) == sequence(kU32, 1));
}

void test_08_matrix_nan_and_signaling_nan()
{
  {
    fixture_t f;
    set_m(f.proc, 2, kE4M3, filled(kE4M3, 0x7f));
    set_m(f.proc, 4, kE4M3, filled(kE4M3, 0x38));
    set_acc(f.proc, 0, kE4M3, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 4));
    std::vector<elem_t> actual;
    assert(f.proc.ZTU.read_acc(0, actual));
    assert(actual.front() == 0x7f);
  }
  {
    fixture_t f;
    set_m(f.proc, 2, kBF16, filled(kBF16, 0x7f81)); // BF16 sNaN
    set_m(f.proc, 4, kBF16, filled(kBF16, 0x3f80));
    set_acc(f.proc, 0, kBF16, 0);
    execute_ztt(f.proc, ztt_opcode_t::mmulacc_2d, operands(0, 2, 4));
    std::vector<elem_t> actual;
    assert(f.proc.ZTU.read_acc(0, actual));
    assert(actual.front() == 0x7fc0);
    assert(f.proc.ZTU.amefflags() & softfloat_flag_invalid);
  }
}

void test_09_integer_ldexp_family()
{
  fixture_t f;
  set_m(f.proc, 0, kI32, filled(kI32, 0));
  set_m(f.proc, 2, kI32, filled(kI32, 3));
  set_m(f.proc, 4, kI32, filled(kI32, 2));
  execute_ztt(f.proc, ztt_opcode_t::mldexp_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 12);
  set_m(f.proc, 0, kI32, filled(kI32, 1));
  execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 13);
  set_m(f.proc, 2, kI32, filled(kI32, 8));
  set_m(f.proc, 0, kI32, filled(kI32, 0));
  execute_ztt(f.proc, ztt_opcode_t::mrdexp_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 2);
  set_m(f.proc, 0, kI32, filled(kI32, 1));
  execute_ztt(f.proc, ztt_opcode_t::mrdexpacc_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 3);
  f.proc.ZTU.write_amestype(kI32);
  f.proc.get_state()->XPR.write(5, 2);
  set_m(f.proc, 0, kI32, filled(kI32, 0));
  execute_ztt(f.proc, ztt_opcode_t::mldexp_ew_x, operands(0, 5, 2));
  assert(get_m(f.proc, 0).front() == 32);
  set_m(f.proc, 0, kI32, filled(kI32, 1));
  execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew_x, operands(0, 5, 2));
  assert(get_m(f.proc, 0).front() == 33);
}

void test_10_wide_matrix_exponents()
{
  fixture_t f;
  set_m(f.proc, 0, kI128, filled(kI128, 0));
  set_m(f.proc, 4, kI128, filled(kI128, 1));
  set_m(f.proc, 8, kI128, filled(kI128, 64));
  execute_ztt(f.proc, ztt_opcode_t::mldexp_ew, operands(0, 4, 8));
  assert(get_m(f.proc, 0).front() == (elem_t(1) << 64));
  set_m(f.proc, 8, kI128, filled(kI128, elem_t(-64)));
  set_m(f.proc, 4, kI128, filled(kI128, 1));
  execute_ztt(f.proc, ztt_opcode_t::mrdexp_ew, operands(0, 4, 8));
  assert(get_m(f.proc, 0).front() == (elem_t(1) << 64));
}

void test_11_scalar_max_integer_datatype_zero_extension()
{
  fixture_t f;
  constexpr uint32_t sat_i128 = (1u << 30) | (1u << 29) | 128u;
  set_m(f.proc, 0, sat_i128, filled(sat_i128, 1));
  f.proc.get_state()->XPR.write(5, UINT64_C(0x8000000000000000));
  execute_ztt(f.proc, ztt_opcode_t::mldexp_ew_x, operands(0, 5, 0));
  // AME_MAX_INT_DTYPE is signed Int128, but the RV64 register bit pattern is
  // zero-extended before interpretation.  The exponent is therefore +2^63,
  // which saturates positive; the old sign-extension bug treated it as
  // -2^63 and produced zero instead.
  assert(get_m(f.proc, 0).front() == ((elem_t(1) << 127) - 1));
  assert(f.proc.ZTU.amexsat());
}

void test_12_integer_log_bias_and_test_13_axis_shift()
{
  fixture_t f;
  set_m(f.proc, 0, kF32, filled(kF32, 0));
  set_m(f.proc, 2, kF32, filled(kF32, 0x41000000)); // 8
  set_m(f.proc, 4, kI32, filled(kI32, UINT32_C(0xfffffffe))); // -2
  execute_ztt(f.proc, ztt_opcode_t::mlog2sub_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 0x40a00000); // 5

  set_m(f.proc, 0, kI32, sequence(kI32, 1));
  f.proc.get_state()->XPR.write(5, UINT32_C(0x80000000));
  execute_ztt(f.proc, ztt_opcode_t::mrowshift_ew_x, operands(0, 5, 0));
  for (elem_t value : get_m(f.proc, 0))
    assert(value == 0);
}

void test_14_15_invalid_descriptors_are_rejected_before_read()
{
  fixture_t f;
  constexpr uint32_t invalid = fp_dtype(8, 0, false, 32);
  assert(f.proc.ZTU.set_m_datatype(0, invalid));
  assert(f.proc.ZTU.set_m_datatype(2, invalid));
  f.proc.ZTU.write_m_register(0, {});
  f.proc.ZTU.write_m_register(2, {});
  execute_ztt(f.proc, ztt_opcode_t::mabs_ew, operands(0, 2));
  assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
}

void test_16_optional_cross_class_conversion()
{
  fixture_t f;
  set_m(f.proc, 0, kI32, filled(kI32, 0));
  set_m(f.proc, 2, kF32, filled(kF32, 0x40600000)); // 3.5 -> 4 RNE
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(0, 2));
  assert(get_m(f.proc, 0).front() == 4);

  f.proc.get_state()->XPR.write(4, 0x40400000); // 3.0 as FP32
  f.proc.get_state()->XPR.write(5, kF32);
  set_m(f.proc, 4, kI32, filled(kI32, 0));
  execute_ztt(f.proc, ztt_opcode_t::mbcast_m_x, operands(4, 4, 5));
  assert(get_m(f.proc, 4).front() == 3);
}

void test_17_scatter_mixed_types_and_test_20_float_tuple_policy()
{
  std::vector<elem_t> indices(ztt::kNumElements);
  for (unsigned row = 0; row < ztt::kTileLength; ++row)
    for (unsigned col = 0; col < ztt::kTileLength; ++col)
      indices[row * ztt::kTileLength + col] = col;
  fixture_t f;
  set_m(f.proc, 0, kU32, filled(kU32, 10));
  set_m(f.proc, 2, kI64, filled(kI64, UINT64_MAX));
  set_m(f.proc, 4, kI32, indices);
  execute_ztt(f.proc, ztt_opcode_t::mcolscatadd_ew, operands(0, 2, 4));
  assert(get_m(f.proc, 0).front() == 9);

  fixture_t rejected;
  set_m(rejected.proc, 0, kF32, filled(kF32, 0));
  set_m(rejected.proc, 2, kF64, filled(kF64, UINT64_C(0x3ff0000000000000)));
  set_m(rejected.proc, 4, kI32, indices);
  execute_ztt(rejected.proc, ztt_opcode_t::madd_ew, operands(0, 2, 2));
  // This is intentionally a green test for the current implementation
  // contract: mixed floating EW arithmetic is rejected.  The spreadsheet's
  // item 20 asks whether it should instead be accepted and rounded only at
  // the destination; that policy remains open and is called out in the audit
  // document rather than silently treating rejection as a semantic fix.
  assert(rejected.proc.ZTU.amestatus() == ztt::kAmestatusUn);
}

void test_18_math_reference_points()
{
  const struct case_t { ztt_opcode_t op; elem_t input; elem_t expected; } cases[] = {
    {ztt_opcode_t::mcos_ew, 0, 0x3f800000},
    {ztt_opcode_t::mexp2_ew, 0x3f800000, 0x40000000},
    {ztt_opcode_t::mlog2_ew, 0x3f800000, 0},
    {ztt_opcode_t::mrec_ew, 0x3f800000, 0x3f800000},
    {ztt_opcode_t::mrsqrt_ew, 0x3f800000, 0x3f800000},
    {ztt_opcode_t::msin_ew, 0, 0},
    {ztt_opcode_t::mtanh_ew, 0, 0},
  };
  for (const auto& c : cases) {
    fixture_t f;
    set_m(f.proc, 0, kF32, filled(kF32, 0));
    set_m(f.proc, 2, kF32, filled(kF32, c.input));
    execute_ztt(f.proc, c.op, operands(0, 2));
    assert(get_m(f.proc, 0).front() == c.expected);
  }
}

void test_19_wide_scalar_is_zero_extended()
{
  fixture_t f;
  set_m(f.proc, 0, kI128, filled(kI128, 0));
  set_m(f.proc, 4, kI128, filled(kI128, 1));
  f.proc.ZTU.write_amestype(kI128);
  f.proc.get_state()->XPR.write(5, UINT64_C(0x8000000000000000));
  execute_ztt(f.proc, ztt_opcode_t::madd_ew_x, operands(0, 5, 4));
  assert(get_m(f.proc, 0).front() == (elem_t(1) + (elem_t(1) << 63)));
}

void test_21_22_directed_rounding_of_exact_expressions()
{
  for (unsigned mode : {2u, 3u}) {
    fixture_t f;
    const uint32_t dtype = fp_dtype(8, mode, true, 32);
    set_m(f.proc, 0, dtype, filled(dtype, 0));
    // |2^-25 - 1| is exactly halfway between the two FP32 values below/at
    // one.  Rounding the negative subtraction before abs reverses the RDN
    // and RUP answers, which is the bug this row of the spreadsheet reports.
    set_m(f.proc, 2, dtype, filled(dtype, 0x33000000)); // 2^-25
    set_m(f.proc, 4, dtype, filled(dtype, 0x3f800000)); // 1
    execute_ztt(f.proc, ztt_opcode_t::mabsdiff_ew, operands(0, 2, 4));
    assert(get_m(f.proc, 0).front() ==
           (mode == 2 ? 0x3f7fffff : 0x3f800000));
  }

  for (unsigned mode : {2u, 3u}) {
    fixture_t f;
    const uint32_t dtype = fp_dtype(8, mode, true, 32);
    set_m(f.proc, 0, dtype, filled(dtype, 0));
    set_m(f.proc, 2, dtype, filled(dtype, 0x3f8ccccd)); // 1.1
    set_m(f.proc, 4, dtype, filled(dtype, 0x3f8ccccd)); // 1.1
    execute_ztt(f.proc, ztt_opcode_t::mmulneg_ew, operands(0, 2, 4));
    softfloat_roundingMode = mode == 2 ? softfloat_round_min
                                       : softfloat_round_max;
    const float32_t expected = f32_mul(float32_t{0xbf8ccccd},
                                       float32_t{0x3f8ccccd});
    assert(get_m(f.proc, 0).front() == expected.v);
  }
}

void test_23_24_single_rounding_and_flag_isolation()
{
  fixture_t f;
  set_m(f.proc, 0, kI32, filled(kI32, 1));
  set_m(f.proc, 2, kI32, filled(kI32, 1));
  set_m(f.proc, 4, kI32, filled(kI32, UINT32_MAX));
  execute_ztt(f.proc, ztt_opcode_t::mldexpacc_ew, operands(0, 2, 4));
  // Integer -1 exponent: D + A*2^-1 = 1 + 0.5.  The complete expression
  // is rounded once using the datatype's RNU mode, so the result is 2.
  assert(get_m(f.proc, 0).front() == 2);

  fixture_t flags;
  set_m(flags.proc, 0, kE4M3, filled(kE4M3, 0x40));
  set_m(flags.proc, 2, kE4M3, filled(kE4M3, 0x40));
  set_m(flags.proc, 4, kE4M3, filled(kE4M3, 0x38));
  flags.proc.ZTU.write_amefflags(softfloat_flag_inexact);
  execute_ztt(flags.proc, ztt_opcode_t::mlog2sub_ew, operands(0, 2, 4));
  assert(flags.proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(flags.proc.ZTU.amefflags() == softfloat_flag_inexact);
}

} // namespace

int main()
{
  test_01_dedicated_status_contract();
  test_02_strided_xlen_wrap();
  test_03_contiguous_xlen_wrap();
  test_03_left_shift_is_converted_at_destination_width();
  test_mfrintn_preserves_negative_zero();
  test_04_half_difference_operand_order();
  test_05_softfloat_format_paths();
  test_06_debug_move_lane_boundaries();
  test_07_one_register_memory_is_ordered();
  test_08_matrix_nan_and_signaling_nan();
  test_09_integer_ldexp_family();
  test_10_wide_matrix_exponents();
  test_11_scalar_max_integer_datatype_zero_extension();
  test_12_integer_log_bias_and_test_13_axis_shift();
  test_14_15_invalid_descriptors_are_rejected_before_read();
  test_16_optional_cross_class_conversion();
  test_17_scatter_mixed_types_and_test_20_float_tuple_policy();
  test_18_math_reference_points();
  test_19_wide_scalar_is_zero_extended();
  test_21_22_directed_rounding_of_exact_expressions();
  test_23_24_single_rounding_and_flag_isolation();
  return 0;
}
