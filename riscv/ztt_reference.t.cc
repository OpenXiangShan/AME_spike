// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

// Reference tests for the Ztt v0.6 execution layer.  These tests deliberately
// use small, fixed operands so the expected values are independent of the
// implementation's internal helper functions.
#include "cfg.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "trap.h"
#include "ztt_execute.h"
#include "ztt_state.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;
constexpr uint32_t kI32 = (1u << 30) | 32u;
constexpr uint32_t kI8 = (1u << 30) | 8u;
constexpr uint32_t kU32 = 32u;
constexpr uint32_t kF32 = (8u << 26) | (1u << 21) | (1u << 20) |
                           (1u << 8) | 32u;

std::set<unsigned> covered;
unsigned current_opcode = 0;

class test_sim_t final : public simif_t {
 public:
  test_sim_t() : memory(16384, 0) { debug_mmu = nullptr; }
  char* addr_to_mem(reg_t address) override
  {
    if (address < DRAM_BASE || address - DRAM_BASE >= memory.size())
      return nullptr;
    return reinterpret_cast<char*>(memory.data() + address - DRAM_BASE);
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
  std::map<size_t, processor_t*> harts;
};

insn_t operands(unsigned rd, unsigned rs1 = 0, unsigned rs2 = 0)
{
  return insn_t((insn_bits_t(rs2) << 20) |
                (insn_bits_t(rs1) << 15) |
                (insn_bits_t(rd) << 7));
}

elem_t i32(int64_t value) { return elem_t(uint32_t(value)); }
elem_t f32(uint32_t value) { return elem_t(value); }

std::vector<elem_t> filled(uint32_t dtype, elem_t value)
{
  return std::vector<elem_t>(ztt_unit_t::element_count(dtype), value);
}

std::vector<elem_t> sequence(uint32_t dtype, int first = 0)
{
  std::vector<elem_t> result(ztt_unit_t::element_count(dtype));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = elem_t(first + int(i));
  return result;
}

class fixture_t {
 public:
  fixture_t() : proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output)
  {
    sim.harts.emplace(0, &proc);
  }

  void prepare(uint32_t dtype)
  {
    proc.ZTU.reset();
    proc.get_state()->mstatus->write(reg_t(1) << 25);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1));
    assert(proc.ZTU.owned());
    for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
      assert(proc.ZTU.set_m_datatype(reg, dtype));
      assert(proc.ZTU.write_m(reg, filled(dtype,
          ztt_unit_t::datatype_floating(dtype) ? f32(0x3f800000) : i32(1))));
    }
    for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
      assert(proc.ZTU.set_acc_datatype(reg, dtype));
      assert(proc.ZTU.write_acc(reg, filled(dtype,
          ztt_unit_t::datatype_floating(dtype) ? f32(0x3f800000) : i32(1))));
    }
    proc.ZTU.write_amestype(dtype);
    proc.get_state()->XPR.write(2, DRAM_BASE + 256);
    proc.get_state()->XPR.write(3, 32);
    proc.get_state()->XPR.write(4, 4);
  }

  test_sim_t sim;
  std::ostringstream output;
  processor_t proc;
};

void set_m(fixture_t& f, unsigned reg, uint32_t dtype,
           const std::vector<elem_t>& values)
{
  assert(f.proc.ZTU.set_m_datatype(reg, dtype));
  assert(f.proc.ZTU.write_m(reg, values));
}

void expect_m(fixture_t& f, unsigned reg, const std::vector<elem_t>& expected)
{
  std::vector<elem_t> actual;
  assert(f.proc.ZTU.read_m(reg, actual));
  if (actual != expected) {
    std::fprintf(stderr, "opcode %u vector mismatch got=%llx expected=%llx\n",
                 current_opcode,
                 static_cast<unsigned long long>(actual.empty() ? 0 : actual[0]),
                 static_cast<unsigned long long>(expected.empty() ? 0 : expected[0]));
    assert(false);
  }
}

void expect_m_all(fixture_t& f, unsigned reg, elem_t value)
{
  std::vector<elem_t> actual;
  assert(f.proc.ZTU.read_m(reg, actual));
  assert(!actual.empty());
  for (elem_t got : actual) {
    if (got != value) {
      std::fprintf(stderr, "opcode %u expected mismatch raw=%llx got=%llx expected=%llx\n",
                   current_opcode, static_cast<unsigned long long>(got),
                   static_cast<unsigned long long>(f.proc.ZTU.m_register(reg)[0]),
                   static_cast<unsigned long long>(value));
      assert(false);
    }
  }
}

void execute(fixture_t& f, ztt_opcode_t opcode, insn_t insn)
{
  covered.insert(static_cast<unsigned>(opcode));
  current_opcode = static_cast<unsigned>(opcode);
  try {
    execute_ztt(f.proc, opcode, insn);
  } catch (const trap_illegal_instruction&) {
    assert(false && "legal reference operand unexpectedly trapped");
  }
}

struct int_case {
  ztt_opcode_t opcode;
  int64_t expected;
  bool scalar = false;
};

void test_integer_elementwise()
{
  const int_case cases[] = {
    {ztt_opcode_t::mabs_ew, 6},
    {ztt_opcode_t::mabsdiff_ew, 4},
    {ztt_opcode_t::madd_ew, 8},
    {ztt_opcode_t::mand_ew, 2},
    {ztt_opcode_t::mandnot_ew, 4},
    {ztt_opcode_t::mcmpge_ew, int64_t(UINT32_MAX)},
    {ztt_opcode_t::mcmplt_ew, 0},
    {ztt_opcode_t::mhdiff_ew, -2},
    {ztt_opcode_t::mldexp_ew, 24},
    {ztt_opcode_t::mldexpacc_ew, 34},
    {ztt_opcode_t::mmax_ew, 6},
    {ztt_opcode_t::mmean_ew, 4},
    {ztt_opcode_t::mmin_ew, 2},
    {ztt_opcode_t::mmul_ew, 12},
    {ztt_opcode_t::mmulacc_ew, 22},
    {ztt_opcode_t::mmulaccneg_ew, -2},
    {ztt_opcode_t::mmuladd_ew, 26},
    {ztt_opcode_t::mmulneg_ew, -12},
    {ztt_opcode_t::mmulsub_ew, -14},
    {ztt_opcode_t::mrdexp_ew, 2},
    {ztt_opcode_t::mrdexpacc_ew, 12},
    {ztt_opcode_t::msll_ew_x, 32, true},
    {ztt_opcode_t::msra_ew_x, 0, true},
    {ztt_opcode_t::msrl_ew_x, 0, true},
    {ztt_opcode_t::mornot_ew, int64_t(UINT32_MAX)},
    {ztt_opcode_t::mor_ew, 6},
    {ztt_opcode_t::msll_ew, 32},
    {ztt_opcode_t::msra_ew, 2},
    {ztt_opcode_t::msrl_ew, 2},
    {ztt_opcode_t::msub_ew, -4},
    {ztt_opcode_t::mxor_ew, 4},
    {ztt_opcode_t::mabsdiff_ew_x, 2, true},
    {ztt_opcode_t::madd_ew_x, 6, true},
    {ztt_opcode_t::mand_ew_x, 0, true},
    {ztt_opcode_t::mandnot_ew_x, 4, true},
    {ztt_opcode_t::mhdiff_ew_x, -1, true},
    {ztt_opcode_t::mldexp_ew_x, 32, true},
    {ztt_opcode_t::mldexpacc_ew_x, 42, true},
    {ztt_opcode_t::mmax_ew_x, 4, true},
    {ztt_opcode_t::mmean_ew_x, 3, true},
    {ztt_opcode_t::mmin_ew_x, 2, true},
    {ztt_opcode_t::mmul_ew_x, 8, true},
    {ztt_opcode_t::mmulacc_ew_x, 18, true},
    {ztt_opcode_t::mmulaccneg_ew_x, 2, true},
    {ztt_opcode_t::mmuladd_ew_x, 24, true},
    {ztt_opcode_t::mmulneg_ew_x, -8, true},
    {ztt_opcode_t::mmulsub_ew_x, -16, true},
    {ztt_opcode_t::mor_ew_x, 6, true},
    {ztt_opcode_t::mornot_ew_x, int64_t(UINT32_MAX - 2), true},
    {ztt_opcode_t::msub_ew_x, -2, true},
    {ztt_opcode_t::mxor_ew_x, 6, true},
  };
  for (const auto& c : cases) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(10)));
    set_m(f, 2, kI32, filled(kI32, i32(6)));
    set_m(f, 3, kI32, filled(kI32, i32(2)));
    f.proc.get_state()->XPR.write(4, 4);
    if (c.opcode == ztt_opcode_t::msll_ew ||
        c.opcode == ztt_opcode_t::msra_ew ||
        c.opcode == ztt_opcode_t::msrl_ew) {
      set_m(f, 2, kI32, filled(kI32, i32(8)));
      set_m(f, 3, kI32, filled(kI32, i32(2)));
      // Vector shifts use the per-element amount in rs2.
      f.proc.get_state()->XPR.write(4, 1);
    }
    execute(f, c.opcode, c.scalar ? operands(1, 4, 3) : operands(1, 2, 3));
    expect_m_all(f, 1, i32(c.expected));
  }
}

struct fp_case {
  ztt_opcode_t opcode;
  uint32_t expected;
  bool scalar = false;
  bool exponent_vector = false;
};

void test_float_elementwise()
{
  const fp_case cases[] = {
    {ztt_opcode_t::mabs_ew, 0x3f800000},
    {ztt_opcode_t::mabsdiff_ew, 0x3f800000},
    {ztt_opcode_t::madd_ew, 0x40400000},
    {ztt_opcode_t::mcos_ew, 0x3f800000},
    {ztt_opcode_t::mexp2_ew, 0x40000000},
    {ztt_opcode_t::mfrintm_ew, 0x3f800000},
    {ztt_opcode_t::mfrintn_ew, 0x3f800000},
    {ztt_opcode_t::mfrintp_ew, 0x3f800000},
    {ztt_opcode_t::mfrintz_ew, 0x3f800000},
    {ztt_opcode_t::mrec_ew, 0x3f800000},
    {ztt_opcode_t::mhdiff_ew, 0x3f000000},
    {ztt_opcode_t::mldexp_ew, 0x40000000, false, true},
    {ztt_opcode_t::mldexpacc_ew, 0x40800000, false, true},
    {ztt_opcode_t::mlog2_ew, 0},
    {ztt_opcode_t::mlog2sub_ew, 0xc0000000},
    {ztt_opcode_t::mmax_ew, 0x40000000},
    {ztt_opcode_t::mmean_ew, 0x3fc00000},
    {ztt_opcode_t::mmin_ew, 0x3f800000},
    {ztt_opcode_t::mmul_ew, 0x40000000},
    {ztt_opcode_t::mmulacc_ew, 0x40800000},
    {ztt_opcode_t::mmulaccneg_ew, 0},
    {ztt_opcode_t::mmuladd_ew, 0x40a00000},
    {ztt_opcode_t::mmulneg_ew, 0xc0000000},
    {ztt_opcode_t::mmulsub_ew, 0xc0400000},
    {ztt_opcode_t::mrdexp_ew, 0x3f000000, false, true},
    {ztt_opcode_t::mrdexpacc_ew, 0x40200000, false, true},
    {ztt_opcode_t::msin_ew, 0},
    {ztt_opcode_t::msqrt_ew, 0x3f800000},
    {ztt_opcode_t::mrsqrt_ew, 0x3f800000},
    {ztt_opcode_t::msub_ew, 0x3f800000},
    {ztt_opcode_t::msublog2_ew, 0x40000000},
    {ztt_opcode_t::mtanh_ew, 0},
  };
  for (const auto& c : cases) {
    fixture_t f;
    f.prepare(kF32);
    set_m(f, 1, kF32, filled(kF32, f32(0x40000000)));
    set_m(f, 2, kF32, filled(kF32, f32(0x3f800000)));
    set_m(f, 3, kF32, filled(kF32, f32(0x40000000)));
    if (c.opcode == ztt_opcode_t::mcos_ew ||
        c.opcode == ztt_opcode_t::msin_ew ||
        c.opcode == ztt_opcode_t::mtanh_ew)
      set_m(f, 2, kF32, filled(kF32, f32(0)));
    const bool scalar_exponent = c.opcode == ztt_opcode_t::mldexp_ew_x ||
                                 c.opcode == ztt_opcode_t::mldexpacc_ew_x;
    f.proc.get_state()->XPR.write(4, scalar_exponent ? 1 : 0x3f800000);
    if (c.exponent_vector)
      set_m(f, 3, kI32, filled(kI32, i32(1)));
    execute(f, c.opcode, c.scalar ? operands(1, 4, 3) : operands(1, 2, 3));
    expect_m_all(f, 1, f32(c.expected));
  }
  const std::pair<ztt_opcode_t, uint32_t> scalar_reference[] = {
    {ztt_opcode_t::mabsdiff_ew_x, 0x40000000},
    {ztt_opcode_t::madd_ew_x, 0x40000000},
    {ztt_opcode_t::mhdiff_ew_x, 0x3f800000},
    {ztt_opcode_t::mldexp_ew_x, 0x40000000},
    {ztt_opcode_t::mldexpacc_ew_x, 0x40800000},
    {ztt_opcode_t::mlog2sub_ew_x, 0x3f800000},
    {ztt_opcode_t::mmax_ew_x, 0x40000000},
    {ztt_opcode_t::mmean_ew_x, 0x3f800000},
    {ztt_opcode_t::mmin_ew_x, 0},
    {ztt_opcode_t::mmul_ew_x, 0},
    {ztt_opcode_t::mmulacc_ew_x, 0x40000000},
    {ztt_opcode_t::mmulaccneg_ew_x, 0x40000000},
    {ztt_opcode_t::mmuladd_ew_x, 0x40800000},
    {ztt_opcode_t::mmulneg_ew_x, 0x80000000},
    {ztt_opcode_t::mmulsub_ew_x, 0xc0800000},
    {ztt_opcode_t::msub_ew_x, 0x40000000},
    {ztt_opcode_t::msublog2_ew_x, 0xbf800000},
  };
  for (const auto& scalar_case : scalar_reference) {
    fixture_t f;
    f.prepare(kF32);
    set_m(f, 1, kF32, filled(kF32, f32(0x40000000)));
    set_m(f, 3, kF32, filled(kF32, f32(0x40000000)));
    f.proc.get_state()->XPR.write(4, 0);
    execute(f, scalar_case.first, operands(1, 4, 3));
    expect_m_all(f, 1, f32(scalar_case.second));
  }

  for (auto opcode : {ztt_opcode_t::mcmpge_ew,
                      ztt_opcode_t::mcmplt_ew}) {
    fixture_t f;
    f.prepare(kF32);
    set_m(f, 1, kI32, filled(kI32, 0));
    set_m(f, 2, kF32, filled(kF32, f32(0x3f800000)));
    set_m(f, 3, kF32, filled(kF32, f32(0x40000000)));
    execute(f, opcode, operands(1, 2, 3));
    expect_m_all(f, 1, opcode == ztt_opcode_t::mcmpge_ew ? 0 : UINT32_MAX);
  }
  for (auto opcode : {ztt_opcode_t::mcmpge_ew_x,
                      ztt_opcode_t::mcmplt_ew_x}) {
    fixture_t f;
    f.prepare(kF32);
    set_m(f, 1, kI32, filled(kI32, 0));
    set_m(f, 3, kF32, filled(kF32, f32(0x40000000)));
    f.proc.get_state()->XPR.write(4, 0x3f800000);
    execute(f, opcode, operands(1, 4, 3));
    expect_m_all(f, 1, opcode == ztt_opcode_t::mcmpge_ew_x ? 0 : UINT32_MAX);
  }
}

void test_control_and_moves()
{
  fixture_t f;
  f.prepare(kI32);
  f.proc.get_state()->XPR.write(4, kI32);
  execute(f, ztt_opcode_t::asettyp, operands(0, 4));
  assert(f.proc.ZTU.acc_datatype(0) == kI32);
  execute(f, ztt_opcode_t::agettyp, operands(5, 0));
  assert(f.proc.get_state()->XPR[5] == kI32);
  f.proc.get_state()->XPR.write(4, kI32);
  execute(f, ztt_opcode_t::msettyp, operands(6, 4));
  execute(f, ztt_opcode_t::mgettyp, operands(5, 6));
  assert(f.proc.get_state()->XPR[5] == kI32);

  set_m(f, 6, kI32, filled(kI32, i32(9)));
  execute(f, ztt_opcode_t::mzero_2d_m, operands(6));
  expect_m_all(f, 6, i32(0));
  set_m(f, 7, kI32, filled(kI32, i32(4)));
  execute(f, ztt_opcode_t::mmov_m_m, operands(6, 7));
  expect_m_all(f, 6, i32(4));
  set_m(f, 7, kI32, sequence(kI32, 1));
  execute(f, ztt_opcode_t::mmov_a_m, operands(0, 7));
  set_m(f, 6, kI32, filled(kI32, i32(0)));
  execute(f, ztt_opcode_t::mmov_m_a, operands(6, 0));
  expect_m(f, 6, sequence(kI32, 1));
  execute(f, ztt_opcode_t::mzero_2d_acc, operands(0));
  for (uint64_t word : f.proc.ZTU.accumulator(0))
    assert(word == 0);

  // The debug moves address one 32-bit lane at a time in a raw M register.
  f.proc.get_state()->XPR.write(4, UINT64_C(0xdeadbeef));
  f.proc.get_state()->XPR.write(3, 2);
  execute(f, ztt_opcode_t::mmove32_m_x, operands(6, 4, 3));
  execute(f, ztt_opcode_t::mmove32_x_m, operands(5, 6, 3));
  assert(f.proc.get_state()->XPR[5] == UINT64_C(0xdeadbeef));
  f.proc.get_state()->XPR.write(4, UINT64_C(0x0123456789abcdef));
  f.proc.get_state()->XPR.write(3, 1);
  execute(f, ztt_opcode_t::mmove64_m_x, operands(6, 4, 3));
  execute(f, ztt_opcode_t::mmove64_x_m, operands(5, 6, 3));
  assert(f.proc.get_state()->XPR[5] == UINT64_C(0x0123456789abcdef));
  for (auto opcode : {ztt_opcode_t::mmove8_m_x,
                      ztt_opcode_t::mmove16_m_x,
                      ztt_opcode_t::mmove8_x_m,
                      ztt_opcode_t::mmove16_x_m}) {
    f.proc.get_state()->XPR.write(4, 0x55);
    execute(f, opcode, operands(6, 4, 0));
  }
}

std::vector<elem_t> matrix_reference(const std::vector<elem_t>& a,
                                     const std::vector<elem_t>& b,
                                     bool ta, bool tb, bool negate)
{
  std::vector<elem_t> result(ztt::kNumElements);
  for (unsigned r = 0; r < ztt::kTileLength; ++r)
    for (unsigned c = 0; c < ztt::kTileLength; ++c) {
      int64_t sum = 0;
      for (unsigned k = 0; k < ztt::kTileLength; ++k) {
        const unsigned ia = ta ? k * ztt::kTileLength + r
                               : r * ztt::kTileLength + k;
        const unsigned ib = tb ? c * ztt::kTileLength + k
                               : k * ztt::kTileLength + c;
        const int64_t av = int64_t(uint32_t(a[ia]));
        const int64_t bv = int64_t(uint32_t(b[ib]));
        sum += negate ? -av * bv : av * bv;
      }
      result[r * ztt::kTileLength + c] = i32(sum);
    }
  return result;
}

void test_matrix_and_conversion()
{
  const auto a = sequence(kI32, 1);
  const auto b = sequence(kI32, 17);
  const ztt_opcode_t matrix_ops[] = {
    ztt_opcode_t::mmulacc_2d, ztt_opcode_t::mmulaccneg_2d,
    ztt_opcode_t::mmulatacc_2d, ztt_opcode_t::mmulataccneg_2d,
    ztt_opcode_t::mmulbtacc_2d, ztt_opcode_t::mmulbtaccneg_2d,
  };
  for (unsigned i = 0; i < 6; ++i) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 2, kI32, a);
    set_m(f, 3, kI32, b);
    set_m(f, 4, kI32, filled(kI32, i32(0)));
    f.proc.ZTU.set_acc_datatype(0, kI32);
    f.proc.ZTU.write_acc(0, filled(kI32, i32(0)));
    execute(f, matrix_ops[i], operands(0, 2, 3));
    const bool ta = i == 2 || i == 3;
    const bool tb = i == 4 || i == 5;
    const bool neg = i == 1 || i == 3 || i == 5;
    std::vector<elem_t> actual;
    assert(f.proc.ZTU.read_acc(0, actual));
    assert(actual == matrix_reference(a, b, ta, tb, neg));
  }

  // Int8 is four packed squares; mconv expands them to four Int32 M regs.
  {
    fixture_t f;
    f.prepare(kI32);
    const auto src = sequence(kI8, 1);
    set_m(f, 0, kI8, src);
    for (unsigned r = 12; r < 16; ++r)
      set_m(f, r, kI32, filled(kI32, i32(0)));
    execute(f, ztt_opcode_t::mconv_ew, operands(12, 0));
    for (unsigned square = 0; square < 4; ++square) {
      std::vector<elem_t> expected(ztt::kNumElements);
      for (unsigned e = 0; e < ztt::kNumElements; ++e)
        expected[e] = i32(square * ztt::kNumElements + e + 1);
      expect_m(f, 12 + square, expected);
    }
  }
  {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 20, kI8, filled(kI8, i32(0)));
    set_m(f, 21, kI32, sequence(kI32, 1));
    f.proc.get_state()->XPR.write(4, 2);
    execute(f, ztt_opcode_t::mpack_ew_x, operands(20, 4, 21));
    std::vector<elem_t> packed;
    assert(f.proc.ZTU.read_m(20, packed));
    for (unsigned e = 0; e < ztt::kNumElements; ++e)
      assert(packed[2 * ztt::kNumElements + e] == i32(e + 1));
    set_m(f, 22, kI32, filled(kI32, i32(0)));
    f.proc.get_state()->XPR.write(4, 2);
    execute(f, ztt_opcode_t::munpack_ew_x, operands(22, 4, 20));
    expect_m(f, 22, sequence(kI32, 1));
  }
}

void test_axis_and_reductions()
{
  const auto src = sequence(kI32, 0);
  const auto indices = [] {
    std::vector<elem_t> result(ztt::kNumElements);
    for (unsigned r = 0; r < ztt::kTileLength; ++r)
      for (unsigned c = 0; c < ztt::kTileLength; ++c)
        result[r * ztt::kTileLength + c] = (3 - c) & 3;
    return result;
  }();

  for (auto opcode : {ztt_opcode_t::mcolid_ew, ztt_opcode_t::mrowid_ew}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(99)));
    execute(f, opcode, operands(1));
    std::vector<elem_t> expected(ztt::kNumElements);
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned c = 0; c < 4; ++c)
        expected[r * 4 + c] = i32(opcode == ztt_opcode_t::mrowid_ew ? r : c);
    expect_m(f, 1, expected);
  }

  for (auto item : {std::pair{ztt_opcode_t::mcolbcast_ew_x, false},
                    std::pair{ztt_opcode_t::mrowbcast_ew_x, true},
                    std::pair{ztt_opcode_t::mcolshift_ew_x, false},
                    std::pair{ztt_opcode_t::mrowshift_ew_x, true}}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(99)));
    set_m(f, 3, kI32, src);
    f.proc.get_state()->XPR.write(4, item.second ? 1 : 1);
    execute(f, item.first, operands(1, 4, 3));
    std::vector<elem_t> expected(ztt::kNumElements);
    const bool broadcast = item.first == ztt_opcode_t::mcolbcast_ew_x ||
                           item.first == ztt_opcode_t::mrowbcast_ew_x;
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned c = 0; c < 4; ++c) {
        int sr = int(r), sc = int(c);
        if (broadcast)
          (item.second ? sr : sc) = 1;
        else
          (item.second ? sr : sc) += 1;
        expected[r * 4 + c] = (sr < 4 && sc < 4) ? src[sr * 4 + sc] : i32(0);
      }
    expect_m(f, 1, expected);
  }

  for (auto item : {std::pair{ztt_opcode_t::mcolgather_ew, true},
                    std::pair{ztt_opcode_t::mrowgather_ew, false}}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(0)));
    set_m(f, 2, kI32, src);
    set_m(f, 3, kI32, indices);
    execute(f, item.first, operands(1, 2, 3));
    std::vector<elem_t> expected(ztt::kNumElements);
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned c = 0; c < 4; ++c) {
        const unsigned selected = 3 - c;
        expected[r * 4 + c] = item.second ? src[r * 4 + selected]
                                           : src[selected * 4 + c];
      }
    expect_m(f, 1, expected);
  }

  for (auto item : {std::pair{ztt_opcode_t::mcolzip_ew, true},
                    std::pair{ztt_opcode_t::mcolunzip_ew, false},
                    std::pair{ztt_opcode_t::mrowzip_ew, true},
                    std::pair{ztt_opcode_t::mrowunzip_ew, false}}) {
    fixture_t f;
    f.prepare(kI32);
    const auto left = sequence(kI32, 0);
    const auto right = sequence(kI32, 100);
    set_m(f, 1, kI32, left);
    set_m(f, 2, kI32, right);
    execute(f, item.first, operands(0, 1, 2));
    std::vector<elem_t> out0(ztt::kNumElements), out1(ztt::kNumElements);
    assert(f.proc.ZTU.read_m(1, out0));
    assert(f.proc.ZTU.read_m(2, out1));
    fixture_t g;
    g.prepare(kI32);
    set_m(g, 1, kI32, out0);
    set_m(g, 2, kI32, out1);
    const ztt_opcode_t inverse = item.second
      ? (item.first == ztt_opcode_t::mcolzip_ew
          ? ztt_opcode_t::mcolunzip_ew : ztt_opcode_t::mrowunzip_ew)
      : (item.first == ztt_opcode_t::mcolunzip_ew
          ? ztt_opcode_t::mcolzip_ew : ztt_opcode_t::mrowzip_ew);
    execute(g, inverse, operands(0, 1, 2));
    std::vector<elem_t> round0, round1;
    assert(g.proc.ZTU.read_m(1, round0));
    assert(g.proc.ZTU.read_m(2, round1));
    assert(round0 == left && round1 == right);
  }

  for (auto item : {std::tuple{ztt_opcode_t::mprefixadd_col, true, false, false},
                    std::tuple{ztt_opcode_t::mprefixadd_row, false, false, false},
                    std::tuple{ztt_opcode_t::mprefixmax_col, true, true, false},
                    std::tuple{ztt_opcode_t::mprefixmax_row, false, true, false},
                    std::tuple{ztt_opcode_t::mreduceadd_col, true, false, true},
                    std::tuple{ztt_opcode_t::mreduceadd_row, false, false, true},
                    std::tuple{ztt_opcode_t::mreducemax_col, true, true, true},
                    std::tuple{ztt_opcode_t::mreducemax_row, false, true, true},
                    std::tuple{ztt_opcode_t::mreducemin_col, true, false, true},
                    std::tuple{ztt_opcode_t::mreducemin_row, false, false, true}}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(0)));
    set_m(f, 2, kI32, src);
    execute(f, std::get<0>(item), operands(1, 2));
    const bool col = std::get<1>(item), maximum = std::get<2>(item);
    const bool reduce = std::get<3>(item);
    std::vector<elem_t> expected(ztt::kNumElements);
    for (unsigned outer = 0; outer < 4; ++outer) {
      int running = 0;
      for (unsigned inner = 0; inner < 4; ++inner) {
        const unsigned index = col ? inner * 4 + outer : outer * 4 + inner;
        const int value = int(src[index]);
        if (inner == 0) running = value;
        else if (maximum) running = std::max(running, value);
        else if (std::get<0>(item) == ztt_opcode_t::mreducemin_col ||
                 std::get<0>(item) == ztt_opcode_t::mreducemin_row)
          running = std::min(running, value);
        else running += value;
        expected[index] = i32(running);
      }
      if (reduce)
        for (unsigned inner = 0; inner < 4; ++inner)
          expected[col ? inner * 4 + outer : outer * 4 + inner] = i32(running);
    }
    expect_m(f, 1, expected);
  }

  for (auto opcode : {ztt_opcode_t::mcolscatadd_ew,
                      ztt_opcode_t::mrowscatadd_ew,
                      ztt_opcode_t::mcolscatmax_ew,
                      ztt_opcode_t::mrowscatmax_ew}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(10)));
    set_m(f, 2, kI32, sequence(kI32, 1));
    set_m(f, 3, kI32, indices);
    execute(f, opcode, operands(1, 2, 3));
    const bool col = opcode == ztt_opcode_t::mcolscatadd_ew ||
                     opcode == ztt_opcode_t::mcolscatmax_ew;
    const bool maximum = opcode == ztt_opcode_t::mcolscatmax_ew ||
                         opcode == ztt_opcode_t::mrowscatmax_ew;
    std::vector<elem_t> expected = filled(kI32, i32(10));
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned c = 0; c < 4; ++c) {
        const unsigned dst = col ? r * 4 + (3 - c) : (3 - c) * 4 + c;
        const int value = int(r * 4 + c + 1);
        if (maximum) expected[dst] = i32(std::max(int(expected[dst]), value));
        else expected[dst] = i32(int(expected[dst]) + value);
      }
    expect_m(f, 1, expected);
  }
}

void test_memory()
{
  const auto expected = sequence(kI32, 1);
  const std::pair<ztt_opcode_t, ztt_opcode_t> pairs[] = {
    {ztt_opcode_t::mls_1r, ztt_opcode_t::mss_1r},
    {ztt_opcode_t::mls_rm, ztt_opcode_t::mss_rm},
    {ztt_opcode_t::mls_cm, ztt_opcode_t::mss_cm},
    {ztt_opcode_t::mls_st, ztt_opcode_t::mss_st},
    {ztt_opcode_t::mls_tst, ztt_opcode_t::mss_tst},
  };
  for (const auto& pair : pairs) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, expected);
    f.proc.get_state()->XPR.write(2, DRAM_BASE + 512);
    f.proc.get_state()->XPR.write(3, 32);
    execute(f, pair.second, operands(1, 2, 3));
    set_m(f, 1, kI32, filled(kI32, i32(0)));
    execute(f, pair.first, operands(1, 2, 3));
    expect_m(f, 1, expected);
  }
}

void test_conditional_and_broadcast()
{
  const auto pred = std::vector<elem_t>{i32(-1), i32(0), i32(1), i32(-2),
                                        i32(-1), i32(0), i32(1), i32(-2),
                                        i32(-1), i32(0), i32(1), i32(-2),
                                        i32(-1), i32(0), i32(1), i32(-2)};
  const auto source = sequence(kI32, 20);
  for (auto opcode : {ztt_opcode_t::mcmovge_ew,
                      ztt_opcode_t::mcmovlt_ew,
                      ztt_opcode_t::mselge_ew,
                      ztt_opcode_t::msellt_ew}) {
    fixture_t f;
    f.prepare(kI32);
    set_m(f, 1, kI32, filled(kI32, i32(99)));
    set_m(f, 2, kI32, pred);
    set_m(f, 3, kI32, source);
    execute(f, opcode, operands(1, 2, 3));
    std::vector<elem_t> expected(ztt::kNumElements);
    const bool negative = opcode == ztt_opcode_t::mcmovlt_ew ||
                          opcode == ztt_opcode_t::msellt_ew;
    const bool preserve = opcode == ztt_opcode_t::mcmovge_ew ||
                          opcode == ztt_opcode_t::mcmovlt_ew;
    for (unsigned i = 0; i < ztt::kNumElements; ++i) {
      const bool selected = negative ? int32_t(uint32_t(pred[i])) < 0
                                     : int32_t(uint32_t(pred[i])) >= 0;
      expected[i] = selected ? source[i] : (preserve ? i32(99) : i32(0));
    }
    expect_m(f, 1, expected);
  }

  fixture_t f;
  f.prepare(kI32);
  set_m(f, 1, kI32, filled(kI32, i32(0)));
  f.proc.get_state()->XPR.write(4, 7);
  f.proc.get_state()->XPR.write(5, kI32);
  execute(f, ztt_opcode_t::mbcast_m_x, operands(1, 4, 5));
  expect_m_all(f, 1, i32(7));
}

void test_unsupported_and_ownership()
{
  fixture_t f;
  f.prepare(kI32);
  f.proc.ZTU.write_amestatus(0);
  f.proc.get_state()->XPR.write(4, 0);
  f.proc.ZTU.write_amexsat(1);
  f.proc.ZTU.write_amefflags(21);
  execute(f, ztt_opcode_t::msettyp, operands(6, 4));
  assert(f.proc.ZTU.m_datatype(6) == kI32);
  expect_m_all(f, 6, i32(1));
  assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(f.proc.ZTU.amexsat() == 1);
  assert(f.proc.ZTU.amefflags() == 21);
  f.proc.ZTU.write_amestatus(0);
  assert(f.proc.ZTU.set_m_datatype(6, 0));
  const auto before = f.proc.ZTU.m_register(6);
  execute(f, ztt_opcode_t::madd_ew, operands(6, 6, 6));
  assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(f.proc.ZTU.m_register(6) == before);
  assert(f.proc.ZTU.amexsat() == 1);
  assert(f.proc.ZTU.amefflags() == 21);
  f.proc.ZTU.write_amestatus(0);
  assert(f.proc.ZTU.amestatus() == 0);
  execute(f, ztt_opcode_t::ame_release, operands(0));
  assert(!f.proc.ZTU.owned());
  assert(f.proc.get_state()->XPR[1] == 1);
  covered.insert(static_cast<unsigned>(ztt_opcode_t::ame_acquire));
}

void test_all_reference_semantics()
{
  test_control_and_moves();
  test_integer_elementwise();
  test_float_elementwise();
  test_conditional_and_broadcast();
  test_axis_and_reductions();
  test_matrix_and_conversion();
  test_memory();
  test_unsupported_and_ownership();

  // The enum is intentionally contiguous.  This assertion prevents adding a
  // new Ztt opcode without also adding a reference test above.
  constexpr unsigned opcode_count =
    static_cast<unsigned>(ztt_opcode_t::mzero_2d_m) + 1;
  static_assert(opcode_count == 138);
  for (unsigned i = 0; i < opcode_count; ++i)
    if (!covered.count(i))
      std::fprintf(stderr, "uncovered opcode %u\n", i);
  assert(covered.size() == opcode_count);
}

} // namespace

int main()
{
  test_all_reference_semantics();
  return 0;
}
