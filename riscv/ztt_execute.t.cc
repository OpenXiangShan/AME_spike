// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "softfloat.h"
#include "trap.h"
#include "ztt_execute.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <sstream>
#include <vector>

namespace {
using elem_t = ztt_unit_t::element_t;

constexpr reg_t kMsInitial = reg_t(1) << 25;
constexpr reg_t kMsClean = reg_t(2) << 25;
constexpr reg_t kMsDirty = reg_t(3) << 25;
constexpr reg_t kAcquireGranted = 1;
constexpr reg_t kAcquireBadDesc = reg_t(0x04) << 1;

constexpr uint32_t fp_dtype(unsigned exponent, unsigned rounding,
                            bool infinity, unsigned bits)
{
  return (exponent << 26) | (rounding << 22) |
         (uint32_t(infinity) << 21) | (1u << 20) | (1u << 8) | bits;
}

class test_sim_t final : public simif_t {
 public:
  test_sim_t() : memory(8192, 0) { debug_mmu = nullptr; }
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
  const std::map<size_t, processor_t*>& get_harts() const override { return harts; }
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

class fixture_t {
 public:
  fixture_t() : proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output)
  {
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(reg_t(1) << 25);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1, 0));
    assert(proc.get_state()->XPR[1] == 1);
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

std::vector<elem_t> bounded_sequence(uint32_t dtype, elem_t first)
{
  std::vector<elem_t> values(ztt_unit_t::element_count(dtype));
  const elem_t mask = ztt_unit_t::element_mask(
    ztt_unit_t::datatype_bits(dtype));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = (first + i) & mask;
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
  std::vector<elem_t> result;
  assert(proc.ZTU.read_m(reg, result));
  return result;
}

struct ame_snapshot_t {
  std::array<ztt_unit_t::m_register_t, ztt::kNumMRegisters> m;
  std::array<uint32_t, ztt::kNumMRegisters> md;
  std::array<ztt_unit_t::accumulator_t, ztt::kNumAccRegisters> acc;
  std::array<uint32_t, ztt::kNumAccRegisters> ad;
  uint64_t amestatus;
  uint32_t amestype;
  uint64_t amefflags;
  uint64_t amexsat;
  bool owned;
  reg_t ms;
};

ame_snapshot_t snapshot(processor_t& proc)
{
  ame_snapshot_t result{};
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
    result.m[reg] = proc.ZTU.m_register(reg);
    result.md[reg] = proc.ZTU.m_datatype(reg);
  }
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    result.acc[reg] = proc.ZTU.accumulator(reg);
    result.ad[reg] = proc.ZTU.acc_datatype(reg);
  }
  result.amestatus = proc.ZTU.amestatus();
  result.amestype = proc.ZTU.amestype();
  result.amefflags = proc.ZTU.amefflags();
  result.amexsat = proc.ZTU.amexsat();
  result.owned = proc.ZTU.owned();
  result.ms = proc.get_state()->mstatus->read() & MSTATUS_MS;
  return result;
}

void assert_same(const ame_snapshot_t& lhs, const ame_snapshot_t& rhs)
{
  assert(lhs.m == rhs.m);
  assert(lhs.md == rhs.md);
  assert(lhs.acc == rhs.acc);
  assert(lhs.ad == rhs.ad);
  assert(lhs.amestatus == rhs.amestatus);
  assert(lhs.amestype == rhs.amestype);
  assert(lhs.amefflags == rhs.amefflags);
  assert(lhs.amexsat == rhs.amexsat);
  assert(lhs.owned == rhs.owned);
  assert(lhs.ms == rhs.ms);
}

void assert_same_backend_contents(const ame_snapshot_t& lhs,
                                  const ame_snapshot_t& rhs)
{
  assert(lhs.m == rhs.m);
  assert(lhs.md == rhs.md);
  assert(lhs.acc == rhs.acc);
  assert(lhs.ad == rhs.ad);
  assert(lhs.amestatus == rhs.amestatus);
  assert(lhs.amestype == rhs.amestype);
  assert(lhs.amefflags == rhs.amefflags);
  assert(lhs.amexsat == rhs.amexsat);
}

void expect_illegal(processor_t& proc, ztt_opcode_t opcode, insn_t insn)
{
  bool trapped = false;
  try {
    execute_ztt(proc, opcode, insn);
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
}

insn_t csr_insn(unsigned csr, unsigned funct3, unsigned rs1)
{
  return insn_t((insn_bits_t(csr) << 20) |
                (insn_bits_t(rs1) << 15) |
                (insn_bits_t(funct3) << 12) | 0x73);
}

void expect_csr_write_illegal(processor_t& proc, unsigned csr,
                              unsigned funct3, unsigned rs1)
{
  const ame_snapshot_t before = snapshot(proc);
  bool trapped = false;
  try {
    (void)proc.get_csr(csr, csr_insn(csr, funct3, rs1), true, false);
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
  assert_same(before, snapshot(proc));
}

void seed_backend(processor_t& proc, elem_t value)
{
  constexpr uint32_t u32 = 32;
  set_m(proc, 0, u32, filled(u32, value));
  assert(proc.ZTU.set_acc_datatype(0, u32));
  assert(proc.ZTU.write_acc(0, std::vector<elem_t>(ztt::kNumElements, value)));
  proc.ZTU.write_amestype(u32);
  proc.ZTU.set_unsupported();
  proc.ZTU.write_amefflags(0x15);
  proc.ZTU.set_amexsat();
}

void assert_initial_backend(processor_t& proc)
{
  const ztt_unit_t::m_register_t zero_m{};
  const ztt_unit_t::accumulator_t zero_acc{};
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
    assert(proc.ZTU.m_register(reg) == zero_m);
    assert(proc.ZTU.m_datatype(reg) == 0);
  }
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    assert(proc.ZTU.accumulator(reg) == zero_acc);
    assert(proc.ZTU.acc_datatype(reg) == 0);
  }
  assert(proc.ZTU.amestatus() == 0);
  assert(proc.ZTU.amestype() == 0);
  assert(proc.ZTU.amefflags() == 0);
  assert(proc.ZTU.amexsat() == 0);
}

void test_chapter5_ownership_semantics()
{
  // MS=Off makes every AME instruction illegal, including acquire/release.
  {
    test_sim_t sim;
    std::ostringstream output;
    processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output);
    sim.harts.emplace(0, &proc);
    assert((proc.get_state()->mstatus->read() & MSTATUS_MS) == 0);
    const ame_snapshot_t before = snapshot(proc);
    expect_illegal(proc, ztt_opcode_t::ame_acquire, operands(1, 0));
    expect_illegal(proc, ztt_opcode_t::ame_release, insn_t(0));
    expect_illegal(proc, ztt_opcode_t::madd_ew, operands(0, 1, 2));
    assert_same(before, snapshot(proc));
  }

  // With state enabled but no owner, ordinary operations are illegal and a
  // release is a no-op that preserves MS and all backend state.
  {
    test_sim_t sim;
    std::ostringstream output;
    processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output);
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(kMsClean);
    seed_backend(proc, 0x11);
    const ame_snapshot_t before = snapshot(proc);
    expect_illegal(proc, ztt_opcode_t::madd_ew, operands(0, 1, 2));
    assert_same(before, snapshot(proc));
    execute_ztt(proc, ztt_opcode_t::ame_release, insn_t(0));
    assert_same(before, snapshot(proc));
  }

  // A new dedicated-backend acquisition installs Initial AME state.
  {
    test_sim_t sim;
    std::ostringstream output;
    processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output);
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(kMsDirty);
    seed_backend(proc, 0x22);
    proc.get_state()->XPR.write(4, 0);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(5, 4));
    assert(proc.get_state()->XPR[5] == kAcquireGranted);
    assert(proc.ZTU.owned());
    assert((proc.get_state()->mstatus->read() & MSTATUS_MS) == kMsInitial);
    assert_initial_backend(proc);
    assert(proc.get_csr(ztt::kCsrAmestype, insn_t(0), false, false) == 0);
    assert(proc.get_csr(ztt::kCsrAmefflags, insn_t(0), false, false) == 0);
    assert(proc.get_csr(ztt::kCsrAmexsat, insn_t(0), false, false) == 0);
    assert(proc.get_csr(ztt::kCsrAmestatus, insn_t(0), false, false) == 0);
  }

  // Reacquiring an already-owned backend grants without reinitializing it.
  // An invalid descriptor returns BAD_DESC and has the same atomicity rule.
  {
    fixture_t f;
    seed_backend(f.proc, 0x33);
    f.proc.get_state()->mstatus->write(kMsDirty);
    f.proc.get_state()->XPR.write(4, 0);
    const ame_snapshot_t before_reacquire = snapshot(f.proc);
    execute_ztt(f.proc, ztt_opcode_t::ame_acquire, operands(5, 4));
    assert(f.proc.get_state()->XPR[5] == kAcquireGranted);
    assert_same(before_reacquire, snapshot(f.proc));

    f.proc.get_state()->XPR.write(4, reg_t(1) << 6);
    const ame_snapshot_t before_bad_desc = snapshot(f.proc);
    execute_ztt(f.proc, ztt_opcode_t::ame_acquire, operands(5, 4));
    assert(f.proc.get_state()->XPR[5] == kAcquireBadDesc);
    assert_same(before_bad_desc, snapshot(f.proc));
  }

  // A failed first acquisition must not change unowned backend state or MS.
  {
    test_sim_t sim;
    std::ostringstream output;
    processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                     nullptr, output);
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(kMsClean);
    seed_backend(proc, 0x44);
    proc.get_state()->XPR.write(4, reg_t(1) << 6);
    const ame_snapshot_t before = snapshot(proc);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(5, 4));
    assert(proc.get_state()->XPR[5] == kAcquireBadDesc);
    assert_same(before, snapshot(proc));
  }

  // An owner's release changes only ownership and MS; backend contents are
  // not implicitly reset.  A subsequent non-owner release is a complete no-op.
  {
    fixture_t f;
    seed_backend(f.proc, 0x55);
    f.proc.get_state()->mstatus->write(kMsDirty);
    const ame_snapshot_t before = snapshot(f.proc);
    execute_ztt(f.proc, ztt_opcode_t::ame_release, insn_t(0));
    const ame_snapshot_t after = snapshot(f.proc);
    assert_same_backend_contents(before, after);
    assert(!after.owned);
    assert(after.ms == kMsInitial);

    f.proc.get_state()->mstatus->write(kMsClean);
    const ame_snapshot_t before_nonowner_release = snapshot(f.proc);
    execute_ztt(f.proc, ztt_opcode_t::ame_release, insn_t(0));
    assert_same(before_nonowner_release, snapshot(f.proc));
  }

  // Architectural CSR writes may not turn the active owner's MS field Off.
  // Cover register CSRRW/CSRRC and immediate CSRRWI forms.
  {
    fixture_t f;
    seed_backend(f.proc, 0x66);
    f.proc.get_state()->mstatus->write(kMsDirty);
    f.proc.get_state()->XPR.write(2, 0);
    expect_csr_write_illegal(f.proc, CSR_MSTATUS, 1, 2);
    f.proc.get_state()->XPR.write(2, MSTATUS_MS);
    expect_csr_write_illegal(f.proc, CSR_MSTATUS, 3, 2);
    expect_csr_write_illegal(f.proc, CSR_MSTATUS, 5, 0);
    assert(f.proc.ZTU.owned());
  }

  // sstatus protects the host context.  When V=1, sstatus and vsstatus both
  // address the active guest context and must protect its MS field as well.
  {
    test_sim_t sim;
    std::ostringstream output;
    processor_t proc("rv64imafdch_ztt", "MSU", &sim.cfg, &sim, 0, false,
                     nullptr, output);
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(kMsInitial);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1, 0));
    proc.get_state()->XPR.write(2, 0);
    expect_csr_write_illegal(proc, CSR_SSTATUS, 1, 2);

    proc.get_state()->vsstatus->write(kMsInitial);
    proc.get_state()->v = true;
    expect_csr_write_illegal(proc, CSR_SSTATUS, 1, 2);
    expect_csr_write_illegal(proc, CSR_VSSTATUS, 1, 2);
    assert((proc.get_state()->vsstatus->read() & SSTATUS_MS) == kMsInitial);
    assert(proc.ZTU.owned());
  }
}

void test_dedicated_backend_is_per_hart()
{
  constexpr uint32_t u32 = 32;
  test_sim_t sim;
  std::ostringstream output0;
  std::ostringstream output1;
  std::ostringstream output2;
  processor_t hart0("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                    nullptr, output0);
  processor_t hart1("rv64i_ztt", "M", &sim.cfg, &sim, 1, false,
                    nullptr, output1);
  processor_t hart2("rv64i_ztt", "M", &sim.cfg, &sim, 2, false,
                    nullptr, output2);
  sim.harts.emplace(0, &hart0);
  sim.harts.emplace(1, &hart1);
  sim.harts.emplace(2, &hart2);

  assert(&hart0.ZTU != &hart1.ZTU);
  assert(&hart0.ZTU != &hart2.ZTU);
  assert(&hart1.ZTU != &hart2.ZTU);

  processor_t* const harts[] = {&hart0, &hart1, &hart2};
  for (unsigned i = 0; i < 3; ++i) {
    processor_t& hart = *harts[i];
    hart.get_state()->mstatus->write(kMsClean);
    hart.get_state()->XPR.write(4, 0);
    execute_ztt(hart, ztt_opcode_t::ame_acquire, operands(5, 4));
    assert(hart.get_state()->XPR[5] == kAcquireGranted);
    set_m(hart, 0, u32, filled(u32, i + 1));
    assert(hart.ZTU.set_acc_datatype(0, u32));
    assert(hart.ZTU.write_acc(
      0, std::vector<elem_t>(ztt::kNumElements, 0x100 + i)));
    hart.ZTU.write_amestype(8u << i);
    hart.ZTU.write_amefflags(1u << i);
    assert(hart.get_csr(ztt::kCsrAmeown, insn_t(0), false, false) == 1);
  }

  for (unsigned i = 0; i < 3; ++i) {
    processor_t& hart = *harts[i];
    assert(get_m(hart, 0) == filled(u32, i + 1));
    std::vector<elem_t> acc;
    assert(hart.ZTU.read_acc(0, acc));
    assert(acc == std::vector<elem_t>(ztt::kNumElements, 0x100 + i));
    assert(hart.ZTU.amestype() == (8u << i));
    assert(hart.ZTU.amefflags() == (1u << i));
  }

  const ame_snapshot_t hart1_before = snapshot(hart1);
  const ame_snapshot_t hart2_before = snapshot(hart2);
  execute_ztt(hart0, ztt_opcode_t::ame_release, insn_t(0));
  assert(!hart0.ZTU.owned());
  assert_same(hart1_before, snapshot(hart1));
  assert_same(hart2_before, snapshot(hart2));

  hart0.reset();
  assert(!hart0.ZTU.owned());
  assert_initial_backend(hart0);
  assert_same(hart1_before, snapshot(hart1));
  assert_same(hart2_before, snapshot(hart2));
}

void test_enablement_and_csrs()
{
  test_sim_t sim;
  std::ostringstream output;
  processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false, nullptr, output);
  bool trapped = false;
  try { execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1, 0)); }
  catch (const trap_illegal_instruction&) { trapped = true; }
  assert(trapped);

  proc.get_state()->mstatus->write(reg_t(1) << 25);
  assert(proc.get_csr(ztt::kCsrAmenlen, insn_t(0), false, false) == ztt::kTileLength);
  assert(proc.get_csr(ztt::kCsrAmeown, insn_t(0), false, false) == 0);
  trapped = false;
  try { (void)proc.get_csr(ztt::kCsrAmestype, insn_t(0), false, false); }
  catch (const trap_illegal_instruction&) { trapped = true; }
  assert(trapped);

  execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1, 0));
  assert(proc.get_csr(ztt::kCsrAmeown, insn_t(0), false, false) == 1);

  proc.get_state()->XPR.write(2, 0);
  const insn_t clear_ms((insn_bits_t(CSR_MSTATUS) << 20) |
                        (insn_bits_t(2) << 15) | (insn_bits_t(1) << 12) | 0x73);
  trapped = false;
  try { (void)proc.get_csr(CSR_MSTATUS, clear_ms, true, false); }
  catch (const trap_illegal_instruction&) { trapped = true; }
  assert(trapped);

  execute_ztt(proc, ztt_opcode_t::ame_release, insn_t(0));
  assert(!proc.ZTU.owned());

  // With virtualization inactive, vsstatus belongs to an inactive guest
  // context.  Its non-applicable MS field may therefore be written Off even
  // while the host owns AME; mstatus.MS above remains protected.
  test_sim_t h_sim;
  std::ostringstream h_output;
  processor_t h_proc("rv64imafdch_ztt", "MSU", &h_sim.cfg, &h_sim, 0, false,
                     nullptr, h_output);
  h_proc.get_state()->mstatus->write(reg_t(1) << 25);
  execute_ztt(h_proc, ztt_opcode_t::ame_acquire, operands(1, 0));
  assert(h_proc.ZTU.owned());
  assert(!h_proc.get_state()->v);
  h_proc.get_state()->XPR.write(2, 0);
  const insn_t clear_vs_ms((insn_bits_t(CSR_VSSTATUS) << 20) |
                           (insn_bits_t(2) << 15) |
                           (insn_bits_t(1) << 12) | 0x73);
  (void)h_proc.get_csr(CSR_VSSTATUS, clear_vs_ms, true, false);
}

void test_types_scalars_rounding_and_flags()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u16 = 16;
  constexpr uint32_t sat_u8 = (1u << 29) | 8;

  proc.get_state()->XPR.write(2, u16);
  execute_ztt(proc, ztt_opcode_t::msettyp, operands(4, 2));
  proc.get_state()->XPR.write(3, 0xff);
  proc.get_state()->XPR.write(5, (1u << 30) | 8u);
  execute_ztt(proc, ztt_opcode_t::mbcast_m_x, operands(4, 3, 5));
  const auto broadcast_scalar = get_m(proc, 4);
  assert(std::all_of(broadcast_scalar.begin(), broadcast_scalar.end(),
                     [](elem_t value) { return value == 0xffff; }));

  set_m(proc, 6, u8, filled(u8, 10));
  set_m(proc, 7, u8, filled(u8, 0));
  proc.ZTU.write_amestype(u8);
  proc.get_state()->XPR.write(8, 3);
  execute_ztt(proc, ztt_opcode_t::madd_ew_x, operands(7, 8, 6));
  assert(get_m(proc, 7).front() == 13);

  set_m(proc, 9, sat_u8, filled(sat_u8, 250));
  set_m(proc, 10, sat_u8, filled(sat_u8, 10));
  set_m(proc, 11, sat_u8, filled(sat_u8, 0));
  execute_ztt(proc, ztt_opcode_t::madd_ew, operands(11, 9, 10));
  assert(get_m(proc, 11).front() == 255);
  assert(proc.ZTU.amexsat() == 1);

  constexpr uint32_t rne_u8 = (1u << 27) | 8;
  set_m(proc, 12, rne_u8, filled(rne_u8, 2));
  set_m(proc, 13, rne_u8, filled(rne_u8, 3));
  set_m(proc, 14, rne_u8, filled(rne_u8, 0));
  execute_ztt(proc, ztt_opcode_t::mmean_ew, operands(14, 12, 13));
  assert(get_m(proc, 14).front() == 2);

  constexpr uint32_t f32 = fp_dtype(8, 0, true, 32);
  set_m(proc, 15, f32, filled(f32, 0));
  set_m(proc, 16, f32, filled(f32, 0));
  execute_ztt(proc, ztt_opcode_t::mrec_ew, operands(16, 15));
  assert((uint32_t(get_m(proc, 16).front()) & 0x7f800000u) == 0x7f800000u);
  assert(proc.ZTU.amefflags() & softfloat_flag_infinite);

  set_m(proc, 15, f32, filled(f32, 0x7f800001u));
  execute_ztt(proc, ztt_opcode_t::mrec_ew, operands(16, 15));
  assert(proc.ZTU.amefflags() & softfloat_flag_invalid);

  constexpr uint32_t u32 = 32;
  set_m(proc, 17, u32, filled(u32, 3));
  set_m(proc, 18, f32, filled(f32, 0));
  execute_ztt(proc, ztt_opcode_t::mconv_ew, operands(18, 17));
  assert(uint32_t(get_m(proc, 18).front()) == 0x40400000u);

  set_m(proc, 19, u32, filled(u32, 0xffffffffu));
  const auto old_m19 = proc.ZTU.m_register(19);
  const auto old_sat = proc.ZTU.amexsat();
  const auto old_fflags = proc.ZTU.amefflags();
  proc.ZTU.write_amestatus(0);
  proc.get_state()->XPR.write(2, 0);
  execute_ztt(proc, ztt_opcode_t::msettyp, operands(19, 2));
  assert(proc.ZTU.m_datatype(19) == u32);
  assert(proc.ZTU.m_register(19) == old_m19);
  assert(proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(proc.ZTU.amexsat() == old_sat);
  assert(proc.ZTU.amefflags() == old_fflags);
  proc.ZTU.write_amestatus(0);

  set_m(proc, 20, u8, filled(u8, 0));
  set_m(proc, 21, u32, filled(u32, 7));
  proc.get_state()->XPR.write(3, 5); // Low two bits select packed slot 1.
  execute_ztt(proc, ztt_opcode_t::mpack_ew_x, operands(20, 3, 21));
  assert(get_m(proc, 20)[ztt::kNumElements] == 7);

  set_m(proc, 22, f32, filled(f32, 0));
  set_m(proc, 23, f32, filled(f32, 0x3fc00000u));
  set_m(proc, 24, u32, filled(u32, 2));
  execute_ztt(proc, ztt_opcode_t::mldexp_ew, operands(22, 23, 24));
  assert(uint32_t(get_m(proc, 22).front()) == 0x40c00000u);

  set_m(proc, 25, f32, filled(f32, 0xffffffffu));
  set_m(proc, 26, f32, filled(f32, 0x7fc00000u));
  set_m(proc, 27, f32, filled(f32, 0x3f800000u));
  execute_ztt(proc, ztt_opcode_t::mselge_ew, operands(25, 26, 27));
  assert(get_m(proc, 25).front() == 0);
  set_m(proc, 26, f32, filled(f32, 0x80000000u));
  execute_ztt(proc, ztt_opcode_t::msellt_ew, operands(25, 26, 27));
  assert(get_m(proc, 25).front() == 0);

  set_m(proc, 25, f32, filled(f32, 0));
  set_m(proc, 26, f32, filled(f32, 0x7fa00001u));
  execute_ztt(proc, ztt_opcode_t::mabs_ew, operands(25, 26));
  assert(get_m(proc, 25).front() == 0x7fc00000u);
}

void test_permute_shift_and_moves()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u32 = 32;
  std::vector<elem_t> values(16);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = i;
  set_m(proc, 0, u32, values);
  set_m(proc, 1, u32, filled(u32, 0));
  proc.get_state()->XPR.write(2, 1);
  execute_ztt(proc, ztt_opcode_t::mrowbcast_ew_x, operands(1, 2, 0));
  const auto broadcast = get_m(proc, 1);
  for (std::size_t row = 0; row < 4; ++row)
    for (std::size_t col = 0; col < 4; ++col)
      assert(broadcast[row * 4 + col] == values[4 + col]);

  execute_ztt(proc, ztt_opcode_t::mcolid_ew, operands(1));
  assert(get_m(proc, 1)[7] == 3);
  set_m(proc, 2, u32, filled(u32, 1));
  execute_ztt(proc, ztt_opcode_t::msll_ew, operands(1, 0, 2));
  assert(get_m(proc, 1)[5] == values[5] * 2);

  proc.get_state()->XPR.write(3, 0xdeadbeef);
  proc.get_state()->XPR.write(4, 5);
  execute_ztt(proc, ztt_opcode_t::mmove32_m_x, operands(5, 3, 4));
  execute_ztt(proc, ztt_opcode_t::mmove32_x_m, operands(6, 5, 4));
  assert(proc.get_state()->XPR[6] == 0xdeadbeef);
}

void test_shift_conversion_saturation_and_axis_extremes()
{
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u16 = 16;
  constexpr uint32_t sat_u8 = (1u << 29) | 8u;
  constexpr uint32_t i32 = (1u << 30) | 32u;

  {
    fixture_t f;
    set_m(f.proc, 0, sat_u8, filled(sat_u8, 0x80));
    set_m(f.proc, 1, sat_u8, filled(sat_u8, 1));
    set_m(f.proc, 2, sat_u8, filled(sat_u8, 0));
    execute_ztt(f.proc, ztt_opcode_t::msll_ew, operands(2, 0, 1));
    for (elem_t value : get_m(f.proc, 2))
      assert(value == 0xff);
    assert(f.proc.ZTU.amexsat() == 1);
  }

  {
    fixture_t f;
    set_m(f.proc, 0, u8, filled(u8, 0x80));
    set_m(f.proc, 4, u16, filled(u16, 0));
    set_m(f.proc, 5, u16, filled(u16, 0));
    f.proc.get_state()->XPR.write(3, 1);
    execute_ztt(f.proc, ztt_opcode_t::msll_ew_x, operands(4, 3, 0));
    for (unsigned reg : {4u, 5u}) {
      std::vector<elem_t> values;
      assert(f.proc.ZTU.read_m_as(reg, u16, values));
      for (elem_t value : values)
        assert(value == 0x0100);
    }
    assert(f.proc.ZTU.amexsat() == 0);
  }

  for (ztt_opcode_t opcode : {ztt_opcode_t::mcolshift_ew_x,
                              ztt_opcode_t::mrowshift_ew_x})
    for (uint32_t offset : {UINT32_C(0x7fffffff), UINT32_C(0x80000000)}) {
      fixture_t f;
      set_m(f.proc, 0, i32, bounded_sequence(i32, 1));
      set_m(f.proc, 1, i32, filled(i32, 0xdeadbeef));
      f.proc.get_state()->XPR.write(3, offset);
      execute_ztt(f.proc, opcode, operands(1, 3, 0));
      for (elem_t value : get_m(f.proc, 1))
        assert(value == 0);
    }
}

void test_mixed_scatter_datatypes()
{
  constexpr uint32_t u32 = 32;
  constexpr uint32_t i32 = (1u << 30) | 32u;
  constexpr uint32_t i64 = (1u << 30) | 64u;
  constexpr uint32_t f32 = fp_dtype(8, 0, true, 32);
  constexpr uint32_t f64 = fp_dtype(11, 0, true, 64);

  for (ztt_opcode_t opcode : {ztt_opcode_t::mcolscatadd_ew,
                              ztt_opcode_t::mrowscatadd_ew,
                              ztt_opcode_t::mcolscatmax_ew,
                              ztt_opcode_t::mrowscatmax_ew}) {
    const bool column = opcode == ztt_opcode_t::mcolscatadd_ew ||
                        opcode == ztt_opcode_t::mcolscatmax_ew;
    const bool maximum = opcode == ztt_opcode_t::mcolscatmax_ew ||
                         opcode == ztt_opcode_t::mrowscatmax_ew;
    std::vector<elem_t> indices(ztt::kNumElements);
    for (unsigned row = 0; row < ztt::kTileLength; ++row)
      for (unsigned col = 0; col < ztt::kTileLength; ++col)
        indices[row * ztt::kTileLength + col] = column ? col : row;

    {
      fixture_t f;
      set_m(f.proc, 0, u32, filled(u32, 10));
      set_m(f.proc, 2, i64, filled(i64, UINT64_MAX)); // signed -1
      set_m(f.proc, 4, i32, indices);
      execute_ztt(f.proc, opcode, operands(0, 2, 4));
      for (elem_t value : get_m(f.proc, 0))
        assert(value == (maximum ? 10 : 9));
    }

    {
      fixture_t f;
      set_m(f.proc, 0, f32, filled(f32, 0x3f800000)); // 1
      const elem_t source = maximum
        ? elem_t(UINT64_C(0x4000000000000000))       // 2
        : elem_t(UINT64_C(0x3e70000000400000));      // 2^-24 + 2^-54
      set_m(f.proc, 2, f64, filled(f64, source));
      set_m(f.proc, 4, i32, indices);
      execute_ztt(f.proc, opcode, operands(0, 2, 4));
      const elem_t expected = maximum ? 0x40000000 : 0x3f800001;
      for (elem_t value : get_m(f.proc, 0))
        assert(value == expected);
      assert(f.proc.ZTU.amefflags() ==
             (maximum ? 0 : softfloat_flag_inexact));
    }
  }

  // The relaxation is intentionally limited to datatypes in the same
  // arithmetic class; an integer/float Scatter tuple remains unsupported.
  {
    fixture_t f;
    set_m(f.proc, 0, u32, filled(u32, 10));
    set_m(f.proc, 2, f32, filled(f32, 0x3f800000));
    set_m(f.proc, 4, i32, filled(i32, 0));
    const auto before = get_m(f.proc, 0);
    execute_ztt(f.proc, ztt_opcode_t::mcolscatadd_ew, operands(0, 2, 4));
    assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
    assert(get_m(f.proc, 0) == before);
  }
  {
    fixture_t f;
    set_m(f.proc, 0, u32, filled(u32, 10));
    set_m(f.proc, 2, i64, filled(i64, UINT64_MAX));
    set_m(f.proc, 4, i32, filled(i32, 0));
    const auto before = get_m(f.proc, 0);
    execute_ztt(f.proc, ztt_opcode_t::mcolgather_ew, operands(0, 2, 4));
    assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
    assert(get_m(f.proc, 0) == before);
  }
}

void test_memory_and_atomic_store()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u8 = 8;
  const reg_t address = DRAM_BASE + 128;
  std::vector<elem_t> packed(64);
  for (std::size_t i = 0; i < packed.size(); ++i) packed[i] = i;
  set_m(proc, 0, u8, packed);
  proc.get_state()->XPR.write(3, address);
  execute_ztt(proc, ztt_opcode_t::mss_rm, operands(0, 3));
  for (std::size_t i = 0; i < packed.size(); ++i)
    assert(f.sim.memory[128 + i] == i);
  set_m(proc, 1, u8, filled(u8, 0));
  execute_ztt(proc, ztt_opcode_t::mls_rm, operands(1, 3));
  assert(get_m(proc, 1) == packed);

  proc.get_state()->XPR.write(3, address + 128);
  execute_ztt(proc, ztt_opcode_t::mss_cm, operands(0, 3));
  for (std::size_t square = 0; square < 4; ++square)
    for (std::size_t col = 0; col < 4; ++col)
      for (std::size_t row = 0; row < 4; ++row)
        assert(f.sim.memory[256 + square * 16 + col * 4 + row] ==
               packed[square * 16 + row * 4 + col]);

  std::fill(f.sim.memory.begin(), f.sim.memory.end(), 0);
  proc.get_state()->XPR.write(3, address);
  proc.get_state()->XPR.write(4, 20);
  execute_ztt(proc, ztt_opcode_t::mss_st, operands(0, 3, 4));
  for (std::size_t row = 0; row < 4; ++row)
    for (std::size_t square = 0; square < 4; ++square)
      for (std::size_t col = 0; col < 4; ++col)
        assert(f.sim.memory[128 + row * 20 + square * 4 + col] ==
               packed[square * 16 + row * 4 + col]);

  const auto before = f.sim.memory;
  proc.get_state()->XPR.write(3, DRAM_BASE + f.sim.memory.size() - 16);
  proc.get_state()->XPR.write(4, 16);
  bool trapped = false;
  try { execute_ztt(proc, ztt_opcode_t::mss_st, operands(0, 3, 4)); }
  catch (const trap_store_access_fault&) { trapped = true; }
  assert(trapped);
  assert(f.sim.memory == before);
}

void test_all_integer_memory_shapes_roundtrip()
{
  const uint32_t dtypes[] = {4, 8, 16, 32, 64, 128};
  const struct mode_t {
    ztt_opcode_t store;
    ztt_opcode_t load;
    bool strided;
  } modes[] = {
    {ztt_opcode_t::mss_rm, ztt_opcode_t::mls_rm, false},
    {ztt_opcode_t::mss_cm, ztt_opcode_t::mls_cm, false},
    {ztt_opcode_t::mss_st, ztt_opcode_t::mls_st, true},
    {ztt_opcode_t::mss_tst, ztt_opcode_t::mls_tst, true},
  };

  for (uint32_t dtype : dtypes)
    for (const auto& mode : modes) {
      fixture_t f;
      const auto values = bounded_sequence(dtype, 1);
      set_m(f.proc, 0, dtype, values);
      set_m(f.proc, 16, dtype, filled(dtype, 0));
      const std::size_t alignment =
        (ztt_unit_t::datatype_bits(dtype) + 7) / 8;
      const reg_t address = DRAM_BASE + 1024 + alignment;
      f.proc.get_state()->XPR.write(3, address - address % alignment);
      f.proc.get_state()->XPR.write(4, 128);
      execute_ztt(f.proc, mode.store,
                  operands(0, 3, mode.strided ? 4 : 0));
      execute_ztt(f.proc, mode.load,
                  operands(16, 3, mode.strided ? 4 : 0));
      assert(get_m(f.proc, 16) == values);
    }
}

void test_conversion_across_packed_and_wide_spans()
{
  fixture_t f;
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u16 = 16;
  constexpr uint32_t u128 = 128;
  const auto packed = bounded_sequence(u8, 1);

  set_m(f.proc, 0, u8, packed);
  set_m(f.proc, 16, u128, filled(u128, 0));
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(16, 0));
  for (std::size_t square = 0; square < 4; ++square) {
    std::vector<elem_t> actual;
    assert(f.proc.ZTU.read_m_as(16 + square * 4, u128, actual));
    for (std::size_t i = 0; i < actual.size(); ++i)
      assert(actual[i] == packed[square * ztt::kNumElements + i]);
  }

  assert(f.proc.ZTU.set_m_datatype(0, u128));
  for (std::size_t square = 0; square < 4; ++square) {
    std::vector<elem_t> values(ztt::kNumElements);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = square * ztt::kNumElements + i + 3;
    assert(f.proc.ZTU.write_m_as(square * 4, u128, values));
  }
  set_m(f.proc, 16, u8, filled(u8, 0));
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(16, 0));
  const auto narrowed = get_m(f.proc, 16);
  for (std::size_t i = 0; i < narrowed.size(); ++i)
    assert(narrowed[i] == i + 3);

  set_m(f.proc, 0, u8, packed);
  set_m(f.proc, 4, u16, filled(u16, 0));
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(4, 0));
  std::vector<elem_t> converted;
  assert(f.proc.ZTU.read_m_as(4, u16, converted));
  std::vector<elem_t> second_group;
  assert(f.proc.ZTU.read_m_as(5, u16, second_group));
  converted.insert(converted.end(), second_group.begin(), second_group.end());
  assert(converted == packed);
}

void test_cross_page_memory_roundtrip()
{
  fixture_t f;
  constexpr uint32_t u8 = 8;
  const auto values = bounded_sequence(u8, 0x40);
  set_m(f.proc, 0, u8, values);
  set_m(f.proc, 1, u8, filled(u8, 0));
  const reg_t address = DRAM_BASE + 4096 - 17;
  f.proc.get_state()->XPR.write(3, address);
  execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 3));
  execute_ztt(f.proc, ztt_opcode_t::mls_rm, operands(1, 3));
  assert(get_m(f.proc, 1) == values);
}

} // namespace

int main()
{
  test_chapter5_ownership_semantics();
  test_dedicated_backend_is_per_hart();
  test_enablement_and_csrs();
  test_types_scalars_rounding_and_flags();
  test_permute_shift_and_moves();
  test_shift_conversion_saturation_and_axis_extremes();
  test_mixed_scatter_datatypes();
  test_memory_and_atomic_store();
  test_all_integer_memory_shapes_roundtrip();
  test_conversion_across_packed_and_wide_spans();
  test_cross_page_memory_roundtrip();
  return 0;
}
