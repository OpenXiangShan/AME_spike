// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "processor.h"
#include "simif.h"
#include "trap.h"
#include "ztt_execute.h"
#include "ztt_validation.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <sstream>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;

class test_sim_t final : public simif_t {
 public:
  test_sim_t() { debug_mmu = nullptr; }
  char* addr_to_mem(reg_t) override { return nullptr; }
  bool mmio_load(reg_t, size_t, uint8_t*) override { return false; }
  bool mmio_store(reg_t, size_t, const uint8_t*) override { return false; }
  void proc_reset(unsigned) override {}
  const cfg_t& get_cfg() const override { return cfg; }
  const std::map<size_t, processor_t*>& get_harts() const override
    { return harts; }
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
  explicit fixture_t(const char* isa = "rv64i_ztt")
    : proc(isa, "M", &sim.cfg, &sim, 0, false,
                     nullptr, output)
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

void set_m(processor_t& proc, unsigned reg, uint32_t dtype,
           const std::vector<elem_t>& values)
{
  assert(proc.ZTU.set_m_datatype(reg, dtype));
  assert(proc.ZTU.write_m(reg, values));
}

void test_all_opcodes_have_validation_descriptors()
{
  for (unsigned value = 0;
       value <= unsigned(ztt_opcode_t::mzero_2d_m); ++value)
    (void)ztt_tuple_rule(ztt_opcode_t(value));
}

void test_tuple_precedes_formation()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u64 = 64;
  set_m(proc, 30, u64, filled(u64, 0x55));
  const auto before = proc.ZTU.m_register(30);

  execute_ztt(proc, ztt_opcode_t::mconv_ew, operands(30, 0));

  assert(proc.ZTU.amestatus() & ztt::kAmestatusUn);
  assert(proc.ZTU.m_register(30) == before);
}

void test_supported_tuple_checks_complete_span()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u64 = 64;
  set_m(proc, 0, u8, filled(u8, 3));
  set_m(proc, 26, u64, filled(u64, 0x55));
  const auto before = proc.ZTU.m_register(26);
  bool trapped = false;
  try {
    execute_ztt(proc, ztt_opcode_t::mconv_ew, operands(26, 0));
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
  assert(proc.ZTU.amestatus() == 0);
  assert(proc.ZTU.m_register(26) == before);
}

void test_conversion_ignores_derived_md()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u32 = 32;
  std::vector<elem_t> source(ztt_unit_t::element_count(u8));
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = i;
  set_m(proc, 0, u8, source);
  set_m(proc, 4, u32, filled(u32, 0));
  assert(proc.ZTU.m_datatype(5) == 0);

  execute_ztt(proc, ztt_opcode_t::mconv_ew, operands(4, 0));

  for (unsigned square = 0; square < 4; ++square) {
    std::vector<elem_t> result;
    assert(proc.ZTU.read_m_as(4 + square, u32, result));
    for (std::size_t element = 0; element < ztt::kNumElements; ++element)
      assert(result[element] == source[square * ztt::kNumElements + element]);
  }
}

void test_conversion_between_packed_widths()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t u8 = 8;
  constexpr uint32_t u16 = 16;
  std::vector<elem_t> source(ztt_unit_t::element_count(u8));
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = i;
  set_m(proc, 0, u8, source);
  set_m(proc, 4, u16, filled(u16, 0));

  execute_ztt(proc, ztt_opcode_t::mconv_ew, operands(4, 0));

  std::vector<elem_t> result;
  assert(proc.ZTU.read_m_as(4, u16, result));
  std::vector<elem_t> second;
  assert(proc.ZTU.read_m_as(5, u16, second));
  result.insert(result.end(), second.begin(), second.end());
  assert(result == source);
}

void test_wide_operand_base_alignment()
{
  fixture_t f;
  constexpr uint32_t u16 = 16;
  constexpr uint32_t u64 = 64;
  constexpr std::size_t two_squares = 2;

  // A u64 square uses a two-register group.  With N_sq=2 the complete
  // operand spans four registers, but the base only needs two-register
  // alignment under Ztt v0.6 section 2.3.1/2.3.2.
  for (unsigned base : {0u, 2u, 4u}) {
    const auto formed = ztt_form_m_operand(
      f.proc, operands(base), {base, u64}, two_squares);
    assert(formed.groups == two_squares);
    assert(formed.group_registers == 2);
    assert(formed.register_span == 4);
  }

  bool trapped = false;
  try {
    (void)ztt_form_m_operand(f.proc, operands(3), {3, u64}, two_squares);
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);

  trapped = false;
  try {
    (void)ztt_form_m_operand(f.proc, operands(30), {30, u64}, two_squares);
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);

  // Exercise the same formation through a real instruction: the packed u16
  // source makes N_sq=2 while the u64 destination starts at m2.
  std::vector<elem_t> source(2 * ztt::kNumElements);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = elem_t(i + 1);
  set_m(f.proc, 0, u16, source);
  set_m(f.proc, 2, u64, filled(u64, 0));
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(2, 0));

  std::vector<elem_t> first;
  std::vector<elem_t> second;
  assert(f.proc.ZTU.read_m_as(2, u64, first));
  assert(f.proc.ZTU.read_m_as(4, u64, second));
  for (std::size_t i = 0; i < ztt::kNumElements; ++i) {
    assert(first[i] == source[i]);
    assert(second[i] == source[ztt::kNumElements + i]);
  }
}

void test_encoded_constraints_precede_data_access()
{
  fixture_t f;
  auto& proc = f.proc;
  bool trapped = false;
  try {
    execute_ztt(proc, ztt_opcode_t::mrowzip_ew, operands(0, 3, 3));
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
  assert(proc.ZTU.amestatus() == 0);

  trapped = false;
  try {
    execute_ztt(proc, ztt_opcode_t::ame_release, operands(1));
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
  assert(proc.ZTU.owned());
}

void test_settyp_rejects_unsupported_descriptor()
{
  fixture_t f;
  auto& proc = f.proc;
  constexpr uint32_t custom255 = 0x800000ffu;
  proc.ZTU.write_amexsat(1);
  proc.ZTU.write_amefflags(21);
  ztt_unit_t::m_register_t raw;
  raw.fill(~uint64_t(0));
  for (unsigned reg = 24; reg < 32; ++reg) {
    assert(proc.ZTU.set_m_datatype(reg, 32));
    assert(proc.ZTU.write_m_register(reg, raw));
  }

  const uint32_t old_m_dtype = proc.ZTU.m_datatype(24);
  const auto old_m_registers = proc.ZTU.m_register(24);
  proc.get_state()->XPR.write(4, custom255);
  execute_ztt(proc, ztt_opcode_t::msettyp, operands(24, 4));
  assert(proc.ZTU.m_datatype(24) == old_m_dtype);
  assert(proc.ZTU.m_register(24) == old_m_registers);
  assert(proc.ZTU.amestatus() == ztt::kAmestatusUn);
  for (unsigned reg = 24; reg < 32; ++reg)
    assert(proc.ZTU.m_register(reg) == raw);
  assert(proc.ZTU.amexsat() == 1);
  assert(proc.ZTU.amefflags() == 21);

  proc.ZTU.write_amestatus(0);
  assert(proc.ZTU.set_acc_datatype(0, 32));
  assert(proc.ZTU.write_acc(0, filled(32, 0x55)));
  const uint32_t old_acc_dtype = proc.ZTU.acc_datatype(0);
  const auto old_accumulator = proc.ZTU.accumulator(0);
  execute_ztt(proc, ztt_opcode_t::asettyp, operands(0, 4));
  assert(proc.ZTU.acc_datatype(0) == old_acc_dtype);
  assert(proc.ZTU.accumulator(0) == old_accumulator);
  assert(proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(proc.ZTU.amexsat() == 1);
  assert(proc.ZTU.amefflags() == 21);

  proc.ZTU.write_amestatus(0);
  const uint32_t old_dtype = proc.ZTU.m_datatype(31);
  const auto old_bits = proc.ZTU.m_register(31);
  proc.get_state()->XPR.write(4, custom255);
  bool trapped = false;
  try {
    execute_ztt(proc, ztt_opcode_t::msettyp, operands(31, 4));
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(!trapped);
  assert(proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert(proc.ZTU.m_datatype(31) == old_dtype);
  assert(proc.ZTU.m_register(31) == old_bits);
  assert(proc.ZTU.amexsat() == 1);
  assert(proc.ZTU.amefflags() == 21);

  proc.ZTU.write_amestatus(0);
  proc.get_state()->XPR.write(4, 64);
  trapped = false;
  try {
    execute_ztt(proc, ztt_opcode_t::msettyp, operands(31, 4));
  } catch (const trap_illegal_instruction&) {
    trapped = true;
  }
  assert(trapped);
  assert(proc.ZTU.amestatus() == 0);
  assert(proc.ZTU.m_datatype(31) == old_dtype);
  assert(proc.ZTU.m_register(31) == old_bits);
  assert(proc.ZTU.amexsat() == 1);
  assert(proc.ZTU.amefflags() == 21);

  proc.get_state()->XPR.write(4, 64);
  execute_ztt(proc, ztt_opcode_t::msettyp, operands(24, 4));
  assert(proc.ZTU.m_datatype(24) == 64);
  assert(proc.ZTU.m_datatype(25) == 32);
  for (unsigned reg = 24; reg < 26; ++reg)
    assert(proc.ZTU.m_register(reg) == ztt_unit_t::m_register_t{});
  for (unsigned reg = 26; reg < 32; ++reg)
    assert(proc.ZTU.m_register(reg) == raw);

  execute_ztt(proc, ztt_opcode_t::asettyp, operands(0, 4));
  assert(proc.ZTU.acc_datatype(0) == 64);
  assert(proc.ZTU.accumulator(0) == ztt_unit_t::accumulator_t{});
  assert(proc.ZTU.amestatus() == 0);
  assert(proc.ZTU.amexsat() == 1);
  assert(proc.ZTU.amefflags() == 21);
}

void test_settyp_state_and_priority_matrix()
{
  std::vector<uint32_t> supported_descriptors;
  for (unsigned bits : {4u, 8u, 16u, 32u, 64u, 128u})
    for (unsigned sign = 0; sign < 2; ++sign)
      for (unsigned saturating = 0; saturating < 2; ++saturating)
        for (unsigned rounding = 0; rounding < 4; ++rounding)
          supported_descriptors.push_back(bits | (sign << 30) |
            (saturating << 29) | (rounding << 27));
  const uint32_t float_formats[] = {
    (4u << 26) | (1u << 20) | (1u << 8) | 8,
    (5u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 8,
    (5u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 16,
    (8u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 16,
    (8u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 32,
    (11u << 26) | (1u << 21) | (1u << 20) | (1u << 8) | 64,
  };
  for (uint32_t format : float_formats)
    for (unsigned rounding = 0; rounding < 6; ++rounding)
      supported_descriptors.push_back(format | (rounding << 22));
  auto descriptors = supported_descriptors;
  descriptors.insert(descriptors.end(), {0, 0x800000ffu, 0x80000020u, 0x220u,
    float_formats[4] | (6u << 22), float_formats[4] & ~(1u << 20)});
  for (const char* isa : {"rv32i_ztt", "rv64i_ztt"})
    for (ztt_opcode_t opcode : {ztt_opcode_t::msettyp, ztt_opcode_t::asettyp})
      for (uint32_t dtype : descriptors)
        for (unsigned base = 0; base < 32; ++base)
          for (bool sticky_un : {false, true}) {
            fixture_t fixture(isa);
            auto& proc = fixture.proc;
            for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
              assert(proc.ZTU.set_m_datatype(reg, reg % 4 == 0 ? 128 : reg % 2 ? 16 : 64));
              ztt_unit_t::m_register_t raw;
              raw.fill(UINT64_C(0xfedcba9876543200) + reg);
              assert(proc.ZTU.write_m_register(reg, raw));
            }
            for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
              assert(proc.ZTU.set_acc_datatype(reg, 32));
              assert(proc.ZTU.write_acc_as(reg, 128,
                std::vector<elem_t>(ztt::kNumElements, ~elem_t(reg))));
            }
            proc.ZTU.write_amefflags(21);
            proc.ZTU.write_amexsat(1);
            proc.ZTU.write_amestype(32);
            if (sticky_un)
              proc.ZTU.set_unsupported();
            proc.get_state()->XPR.write(4, UINT64_C(0xdeadbeef00000000) | dtype);
            const ztt_unit_t before = proc.ZTU;
            const auto old_xpr = proc.get_state()->XPR;
            const bool matrix = opcode == ztt_opcode_t::msettyp;
            const bool known = std::find(supported_descriptors.begin(),
              supported_descriptors.end(), dtype) != supported_descriptors.end();
            const unsigned span = known && (dtype & 255) > 32 ? (dtype & 255) / 32 : 1;
            const bool invalid_encoded = !matrix && base >= ztt::kNumAccRegisters;
            const bool invalid_geometry = matrix && known &&
              (base % span != 0 || base + span > ztt::kNumMRegisters);
            const bool expected_trap = invalid_encoded || invalid_geometry;
            bool trapped = false;
            try {
              execute_ztt(proc, opcode, operands(base, 4));
            } catch (const trap_illegal_instruction&) {
              trapped = true;
            }
            assert(trapped == expected_trap);
            const bool committed = known && !expected_trap;
            for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
              assert(proc.ZTU.m_datatype(reg) ==
                (committed && matrix && reg == base ? dtype : before.m_datatype(reg)));
              assert(proc.ZTU.m_register(reg) ==
                (committed && matrix && reg >= base && reg < base + span ?
                 ztt_unit_t::m_register_t{} : before.m_register(reg)));
            }
            for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
              assert(proc.ZTU.acc_datatype(reg) ==
                (committed && !matrix && reg == base ? dtype : before.acc_datatype(reg)));
              assert(proc.ZTU.accumulator(reg) ==
                (committed && !matrix && reg == base ?
                 ztt_unit_t::accumulator_t{} : before.accumulator(reg)));
            }
            assert(proc.ZTU.amestatus() ==
              (sticky_un || (!known && !expected_trap) ? ztt::kAmestatusUn : 0));
            assert(proc.ZTU.amefflags() == before.amefflags());
            assert(proc.ZTU.amexsat() == before.amexsat());
            assert(proc.ZTU.amestype() == before.amestype());
            assert(proc.ZTU.owned() == before.owned());
            for (unsigned reg = 0; reg < 32; ++reg)
              assert(proc.get_state()->XPR[reg] == old_xpr[reg]);
          }
}

void test_settyp_state_access_and_x0()
{
  for (const char* isa : {"rv32i_ztt", "rv64i_ztt"})
    for (ztt_opcode_t opcode : {ztt_opcode_t::msettyp, ztt_opcode_t::asettyp})
      for (unsigned access = 0; access < 3; ++access) {
        fixture_t fixture(isa);
        auto& proc = fixture.proc;
        assert(proc.ZTU.set_m_datatype(0, 32));
        assert(proc.ZTU.set_acc_datatype(0, 32));
        assert(proc.ZTU.write_m(0, filled(32, 0x12345678)));
        assert(proc.ZTU.write_acc_as(0, 128,
          std::vector<elem_t>(ztt::kNumElements, ~elem_t(0))));
        proc.ZTU.write_amefflags(21);
        proc.ZTU.write_amexsat(1);
        if (access == 0)
          proc.get_state()->mstatus->write(0);
        if (access == 1)
          execute_ztt(proc, ztt_opcode_t::ame_release, operands(0));
        const reg_t status = proc.get_state()->mstatus->read();
        const ztt_unit_t before = proc.ZTU;
        bool trapped = false;
        try {
          execute_ztt(proc, opcode, operands(0, 0));
        } catch (const trap_illegal_instruction&) {
          trapped = true;
        }
        assert(trapped == (access != 2));
        assert(proc.ZTU.amestatus() == (access == 2 ? ztt::kAmestatusUn : 0));
        assert(proc.ZTU.m_datatype(0) == 32);
        assert(proc.ZTU.acc_datatype(0) == 32);
        for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
          assert(proc.ZTU.m_datatype(reg) == before.m_datatype(reg));
          assert(proc.ZTU.m_register(reg) == before.m_register(reg));
        }
        for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
          assert(proc.ZTU.acc_datatype(reg) == before.acc_datatype(reg));
          assert(proc.ZTU.accumulator(reg) == before.accumulator(reg));
        }
        assert(proc.ZTU.amefflags() == before.amefflags());
        assert(proc.ZTU.amexsat() == before.amexsat());
        assert(proc.ZTU.owned() == before.owned());
        if (trapped)
          assert(proc.get_state()->mstatus->read() == status);
      }
}

} // namespace

int main()
{
  test_all_opcodes_have_validation_descriptors();
  test_tuple_precedes_formation();
  test_supported_tuple_checks_complete_span();
  test_conversion_ignores_derived_md();
  test_conversion_between_packed_widths();
  test_wide_operand_base_alignment();
  test_encoded_constraints_precede_data_access();
  test_settyp_rejects_unsupported_descriptor();
  test_settyp_state_and_priority_matrix();
  test_settyp_state_access_and_x0();
  return 0;
}
