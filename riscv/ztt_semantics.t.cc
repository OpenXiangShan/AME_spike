// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "trap.h"
#include "ztt_execute.h"
#include "ztt_state.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <sstream>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;

constexpr uint32_t kSigned32 = (1u << 30) | 32u;
constexpr uint32_t kFloat32 =
    (8u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 32u;
constexpr unsigned kOpcodeCount =
    static_cast<unsigned>(ztt_opcode_t::mzero_2d_m) + 1;

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
    proc.get_state()->XPR.write(0, 0);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1));

    const elem_t value = ztt_unit_t::datatype_floating(dtype)
                       ? elem_t(0x3f800000u) : elem_t(1);
    const std::vector<elem_t> values(ztt_unit_t::element_count(dtype), value);
    for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
      assert(proc.ZTU.set_m_datatype(reg, dtype));
      assert(proc.ZTU.write_m(reg, values));
    }
    for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
      assert(proc.ZTU.set_acc_datatype(reg, dtype));
      assert(proc.ZTU.write_acc(reg, values));
    }

    proc.ZTU.write_amestype(dtype);
    proc.get_state()->XPR.write(2, DRAM_BASE + 256);
    proc.get_state()->XPR.write(3, 16);
    proc.get_state()->XPR.write(4, 1);
  }

  test_sim_t sim;
  std::ostringstream output;
  processor_t proc;
};

bool is_floating_opcode(ztt_opcode_t opcode)
{
  switch (opcode) {
    case ztt_opcode_t::mcos_ew:
    case ztt_opcode_t::mexp2_ew:
    case ztt_opcode_t::mfrintm_ew:
    case ztt_opcode_t::mfrintn_ew:
    case ztt_opcode_t::mfrintp_ew:
    case ztt_opcode_t::mfrintz_ew:
    case ztt_opcode_t::mlog2_ew:
    case ztt_opcode_t::mlog2sub_ew:
    case ztt_opcode_t::mlog2sub_ew_x:
    case ztt_opcode_t::mrec_ew:
    case ztt_opcode_t::mrsqrt_ew:
    case ztt_opcode_t::msin_ew:
    case ztt_opcode_t::msqrt_ew:
    case ztt_opcode_t::msublog2_ew:
    case ztt_opcode_t::msublog2_ew_x:
    case ztt_opcode_t::mtanh_ew:
      return true;
    default:
      return false;
  }
}

bool is_vector_exponent_opcode(ztt_opcode_t opcode)
{
  switch (opcode) {
    case ztt_opcode_t::mldexp_ew:
    case ztt_opcode_t::mldexpacc_ew:
    case ztt_opcode_t::mrdexp_ew:
    case ztt_opcode_t::mrdexpacc_ew:
      return true;
    default:
      return false;
  }
}

bool expected_may_modify_state(ztt_opcode_t opcode)
{
  switch (opcode) {
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
    default:
      // This includes datatype-aware stores: an unsupported tuple modifies
      // amestatus.UN.  Conservatively dirtying successful supported forms is
      // permitted by Section 2.1.1.
      return true;
  }
}

insn_t test_operands(fixture_t& fixture, ztt_opcode_t opcode)
{
  switch (opcode) {
    case ztt_opcode_t::ame_acquire:
      return operands(1, 0);
    case ztt_opcode_t::ame_release:
      return operands(0, 0);
    case ztt_opcode_t::agettyp:
      return operands(1, 2, 3); // rd=XPR, rs1=accumulator index.
    case ztt_opcode_t::asettyp:
    case ztt_opcode_t::msettyp:
      fixture.proc.get_state()->XPR.write(4, kSigned32);
      return operands(1, 4, 3);
    case ztt_opcode_t::mzero_2d_acc:
      return operands(1);
    case ztt_opcode_t::mbcast_m_x:
      fixture.proc.get_state()->XPR.write(3, kSigned32);
      return operands(1, 4, 3);
    case ztt_opcode_t::mcolbcast_ew_x:
    case ztt_opcode_t::mrowbcast_ew_x:
    case ztt_opcode_t::mcolshift_ew_x:
    case ztt_opcode_t::mrowshift_ew_x:
      fixture.proc.get_state()->XPR.write(0, 0);
      return operands(1, 0, 3);
    case ztt_opcode_t::mmove8_m_x:
    case ztt_opcode_t::mmove16_m_x:
    case ztt_opcode_t::mmove32_m_x:
    case ztt_opcode_t::mmove64_m_x:
      fixture.proc.get_state()->XPR.write(3, 0);
      return operands(1, 4, 3);
    case ztt_opcode_t::mmove8_x_m:
    case ztt_opcode_t::mmove16_x_m:
    case ztt_opcode_t::mmove32_x_m:
    case ztt_opcode_t::mmove64_x_m:
      fixture.proc.get_state()->XPR.write(3, 0);
      return operands(1, 2, 3);
    case ztt_opcode_t::mls_1r:
    case ztt_opcode_t::mls_rm:
    case ztt_opcode_t::mls_cm:
    case ztt_opcode_t::mls_st:
    case ztt_opcode_t::mls_tst:
    case ztt_opcode_t::mss_1r:
    case ztt_opcode_t::mss_rm:
    case ztt_opcode_t::mss_cm:
    case ztt_opcode_t::mss_st:
    case ztt_opcode_t::mss_tst:
      return operands(1, 2, 3);
    case ztt_opcode_t::mldexp_ew_x:
    case ztt_opcode_t::mldexpacc_ew_x:
      return operands(1, 4, 3);
    default:
      return operands(1, 2, 3);
  }
}

void check_postconditions(fixture_t& fixture, ztt_opcode_t opcode,
                          uint32_t dtype)
{
  auto& proc = fixture.proc;
  switch (opcode) {
    case ztt_opcode_t::ame_acquire:
      assert(proc.ZTU.owned());
      assert(proc.get_state()->XPR[1] == 1);
      return;
    case ztt_opcode_t::ame_release:
      assert(!proc.ZTU.owned());
      return;
    case ztt_opcode_t::agettyp:
    case ztt_opcode_t::mgettyp:
      assert(proc.get_state()->XPR[1] == dtype);
      return;
    case ztt_opcode_t::msettyp:
      assert(proc.ZTU.m_datatype(1) == kSigned32);
      return;
    case ztt_opcode_t::mzero_2d_m: {
      std::vector<elem_t> values;
      assert(proc.ZTU.read_m(1, values));
      assert(!values.empty());
      for (elem_t value : values)
        assert(value == 0);
      return;
    }
    case ztt_opcode_t::mzero_2d_acc:
      for (uint64_t word : proc.ZTU.accumulator(1))
        assert(word == 0);
      return;
    case ztt_opcode_t::mcolid_ew: {
      std::vector<elem_t> values;
      assert(proc.ZTU.read_m(1, values));
      assert(values[0] == 0 && values[1] == 1 && values[2] == 2 &&
             values[3] == 3);
      return;
    }
    case ztt_opcode_t::mrowid_ew: {
      std::vector<elem_t> values;
      assert(proc.ZTU.read_m(1, values));
      assert(values[0] == 0 && values[4] == 1 && values[8] == 2 &&
             values[12] == 3);
      return;
    }
    case ztt_opcode_t::madd_ew: {
      std::vector<elem_t> values;
      assert(proc.ZTU.read_m(1, values));
      assert(values.front() == 2);
      return;
    }
    case ztt_opcode_t::mmove8_x_m:
    case ztt_opcode_t::mmove16_x_m:
    case ztt_opcode_t::mmove32_x_m:
    case ztt_opcode_t::mmove64_x_m:
      assert((proc.get_state()->XPR[1] & 0xffffffffu) == 1);
      return;
    default:
      return;
  }
}

void test_all_semantics()
{
  static_assert(kOpcodeCount == 138);
  fixture_t fixture;

  for (unsigned value = 0; value < kOpcodeCount; ++value) {
    const auto opcode = static_cast<ztt_opcode_t>(value);
    const uint32_t dtype = is_floating_opcode(opcode) ? kFloat32 : kSigned32;
    fixture.prepare(dtype);

    if (is_vector_exponent_opcode(opcode)) {
      // Vector exponent forms combine floating M operands with an integer
      // exponent operand; this is the legal mixed-dtype case in the spec.
      assert(fixture.proc.ZTU.set_m_datatype(3, kSigned32));
      assert(fixture.proc.ZTU.write_m(
          3, std::vector<elem_t>(ztt::kNumElements, elem_t(1))));
    }

    const insn_t insn = test_operands(fixture, opcode);
    const bool should_dirty = expected_may_modify_state(opcode);
    assert(ztt_opcode_may_modify_state(opcode) == should_dirty);
    fixture.proc.get_state()->mstatus->write(
      (fixture.proc.get_state()->mstatus->read() & ~MSTATUS_MS) |
      (reg_t(2) << 25));
    bool illegal = false;
    try {
      execute_ztt(fixture.proc, opcode, insn);
    } catch (const trap_illegal_instruction&) {
      illegal = true;
    }
    assert(!illegal);
    check_postconditions(fixture, opcode, dtype);
    const reg_t ms = fixture.proc.get_state()->mstatus->read() & MSTATUS_MS;
    if (opcode == ztt_opcode_t::ame_release)
      assert(ms == (reg_t(1) << 25));
    else if (should_dirty)
      assert(ms == (reg_t(3) << 25));
    else
      assert(ms == (reg_t(2) << 25));
    if (opcode != ztt_opcode_t::ame_release)
      assert(fixture.proc.ZTU.owned());
  }
}

} // namespace

int main()
{
  test_all_semantics();
  return 0;
}
