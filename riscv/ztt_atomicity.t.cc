// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "mmu.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "trap.h"
#include "ztt_execute.h"
#include "ztt_state.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

namespace {

using elem_t = ztt_unit_t::element_t;

constexpr uint32_t kU8 = 8;
constexpr uint32_t kU32 = 32;
constexpr uint32_t kU64 = 64;
constexpr uint32_t kU128 = 128;
constexpr reg_t kMmioBase = DRAM_BASE + 0x10000;
constexpr std::size_t kMmioSize = 4096;
constexpr std::size_t kMemorySize = 32768;

class test_sim_t final : public simif_t {
 public:
  test_sim_t() : memory(kMemorySize, 0), mmio_memory(kMmioSize, 0)
    { debug_mmu = nullptr; }

  char* addr_to_mem(reg_t address) override
  {
    if (address < DRAM_BASE || address - DRAM_BASE >= memory.size())
      return nullptr;
    return reinterpret_cast<char*>(memory.data() + address - DRAM_BASE);
  }
  bool mmio_load(reg_t, size_t, uint8_t*) override { return false; }
  bool mmio_store(reg_t address, size_t len, const uint8_t* bytes) override
  {
    if (!valid_mmio(address, len))
      return false;
    ++mmio_store_calls;
    std::copy(bytes, bytes + len,
              mmio_memory.begin() + std::size_t(address - kMmioBase));
    return true;
  }
  bool mmio_store_preflight(reg_t address, size_t len) override
  {
    if (!valid_mmio(address, len))
      return false;
    return reject_mmio_address == reg_t(-1) ||
           reject_mmio_address < address ||
           reject_mmio_address >= address + len;
  }
  void proc_reset(unsigned) override {}
  const cfg_t& get_cfg() const override { return cfg; }
  const std::map<size_t, processor_t*>& get_harts() const override
    { return harts; }
  const char* get_symbol(uint64_t) override { return nullptr; }

  cfg_t cfg;
  std::vector<uint8_t> memory;
  std::vector<uint8_t> mmio_memory;
  reg_t reject_mmio_address = reg_t(-1);
  std::size_t mmio_store_calls = 0;
  std::map<size_t, processor_t*> harts;

 private:
  static bool valid_mmio(reg_t address, size_t len)
  {
    return address >= kMmioBase && address + len >= address &&
           address + len <= kMmioBase + kMmioSize;
  }
};

insn_t operands(unsigned rd, unsigned rs1 = 0, unsigned rs2 = 0)
{
  return insn_t((insn_bits_t(rs2) << 20) |
                (insn_bits_t(rs1) << 15) |
                (insn_bits_t(rd) << 7));
}

class fixture_t {
 public:
  explicit fixture_t(const char* privilege_spec = "M")
    : proc("rv64i_ztt", privilege_spec, &sim.cfg, &sim, 0, false,
           nullptr, output)
  {
    sim.harts.emplace(0, &proc);
    proc.get_state()->mstatus->write(reg_t(1) << 25);
    execute_ztt(proc, ztt_opcode_t::ame_acquire, operands(1));
    assert(proc.ZTU.owned());
  }

  void set_clean()
  {
    proc.get_state()->mstatus->write(
      (proc.get_state()->mstatus->read() & ~MSTATUS_MS) | (reg_t(2) << 25));
  }

  test_sim_t sim;
  std::ostringstream output;
  processor_t proc;
};

std::vector<elem_t> sequence(uint32_t dtype, elem_t first)
{
  std::vector<elem_t> values(ztt_unit_t::element_count(dtype));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = first + i;
  return values;
}

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

std::vector<elem_t> read_m(processor_t& proc, unsigned reg)
{
  std::vector<elem_t> values;
  assert(proc.ZTU.read_m(reg, values));
  return values;
}

void test_complete_overlap_snapshot()
{
  const struct overlap_case_t {
    unsigned dest;
    unsigned source1;
    unsigned source2;
  } cases[] = {
    {0, 0, 1}, // Destination completely overlaps source 1.
    {0, 1, 0}, // Destination completely overlaps source 2.
    {0, 0, 0}, // Destination and both sources completely overlap.
  };
  for (const auto& test : cases) {
    fixture_t f;
    const auto value0 = sequence(kU32, 1);
    const auto value1 = sequence(kU32, 101);
    set_m(f.proc, 0, kU32, value0);
    set_m(f.proc, 1, kU32, value1);
    const auto source1 = test.source1 == 0 ? value0 : value1;
    const auto source2 = test.source2 == 0 ? value0 : value1;

    execute_ztt(f.proc, ztt_opcode_t::madd_ew,
                operands(test.dest, test.source1, test.source2));

    std::vector<elem_t> expected(source1.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = uint32_t(source1[i] + source2[i]);
    assert(read_m(f.proc, test.dest) == expected);
  }
}

void test_packed_partial_overlap_snapshot()
{
  fixture_t f;
  set_m(f.proc, 4, kU32, filled(kU32, 0));
  const auto source = sequence(kU8, 1);
  set_m(f.proc, 5, kU8, source);

  // The four-square destination spans m4-m7 while the packed source is m5.
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(4, 5));

  for (unsigned square = 0; square < 4; ++square) {
    std::vector<elem_t> actual;
    assert(f.proc.ZTU.read_m_as(4 + square, kU32, actual));
    for (std::size_t i = 0; i < actual.size(); ++i)
      assert(actual[i] == source[square * ztt::kNumElements + i]);
  }
}

void test_wide_partial_overlap_snapshot()
{
  fixture_t f;
  set_m(f.proc, 4, kU128, sequence(kU128, 0x100));
  set_m(f.proc, 5, kU32, filled(kU32, 0xa5a5a5a5));
  std::vector<elem_t> source_snapshot;
  assert(f.proc.ZTU.read_m_as(4, kU128, source_snapshot));

  // The wide source spans m4-m7 and the destination lies inside that span.
  execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(5, 4));

  const auto actual = read_m(f.proc, 5);
  for (std::size_t i = 0; i < actual.size(); ++i)
    assert(actual[i] == uint32_t(source_snapshot[i]));
}

void test_two_destination_snapshot()
{
  fixture_t f;
  const auto left = sequence(kU32, 0);
  const auto right = sequence(kU32, 100);
  set_m(f.proc, 1, kU32, left);
  set_m(f.proc, 2, kU32, right);

  execute_ztt(f.proc, ztt_opcode_t::mrowzip_ew, operands(0, 1, 2));

  std::vector<elem_t> expected0(ztt::kNumElements);
  std::vector<elem_t> expected1(ztt::kNumElements);
  for (std::size_t global = 0; global < 2 * ztt::kTileLength; ++global)
    for (std::size_t col = 0; col < ztt::kTileLength; ++col) {
      const auto& source = global & 1 ? right : left;
      auto& dest = global < ztt::kTileLength ? expected0 : expected1;
      dest[(global % ztt::kTileLength) * ztt::kTileLength + col] =
        source[(global / 2) * ztt::kTileLength + col];
    }
  assert(read_m(f.proc, 1) == expected0);
  assert(read_m(f.proc, 2) == expected1);
}

struct ame_snapshot_t {
  std::array<ztt_unit_t::m_register_t, ztt::kNumMRegisters> m;
  std::array<uint32_t, ztt::kNumMRegisters> md;
  std::array<ztt_unit_t::accumulator_t, ztt::kNumAccRegisters> acc;
  std::array<uint32_t, ztt::kNumAccRegisters> ad;
  uint32_t amestype;
  uint64_t amestatus;
  uint64_t amefflags;
  uint64_t amexsat;
  bool owned;
  reg_t ms;
};

ame_snapshot_t snapshot(processor_t& proc)
{
  ame_snapshot_t state{};
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg) {
    state.m[reg] = proc.ZTU.m_register(reg);
    state.md[reg] = proc.ZTU.m_datatype(reg);
  }
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    state.acc[reg] = proc.ZTU.accumulator(reg);
    state.ad[reg] = proc.ZTU.acc_datatype(reg);
  }
  state.amestype = proc.ZTU.amestype();
  state.amestatus = proc.ZTU.amestatus();
  state.amefflags = proc.ZTU.amefflags();
  state.amexsat = proc.ZTU.amexsat();
  state.owned = proc.ZTU.owned();
  state.ms = proc.get_state()->mstatus->read() & MSTATUS_MS;
  return state;
}

void assert_same(const ame_snapshot_t& before, const ame_snapshot_t& after)
{
  assert(before.m == after.m);
  assert(before.md == after.md);
  assert(before.acc == after.acc);
  assert(before.ad == after.ad);
  assert(before.amestype == after.amestype);
  assert(before.amestatus == after.amestatus);
  assert(before.amefflags == after.amefflags);
  assert(before.amexsat == after.amexsat);
  assert(before.owned == after.owned);
  assert(before.ms == after.ms);
}

void seed_architectural_state(fixture_t& f)
{
  for (unsigned reg = 0; reg < ztt::kNumMRegisters; ++reg)
    set_m(f.proc, reg, kU32, sequence(kU32, 1 + reg * 32));
  for (unsigned reg = 0; reg < ztt::kNumAccRegisters; ++reg) {
    assert(f.proc.ZTU.set_acc_datatype(reg, kU32));
    assert(f.proc.ZTU.write_acc(reg, sequence(kU32, 0x1000 + reg * 32)));
  }
  f.proc.ZTU.write_amestype(kU32);
  f.proc.ZTU.write_amestatus(0);
  f.proc.ZTU.write_amefflags(0x15);
  f.proc.ZTU.set_amexsat();
  f.set_clean();
  std::fill(f.sim.memory.begin(), f.sim.memory.end(), uint8_t(0x5a));
}

void expect_atomic_trap(fixture_t& f, reg_t cause,
                        const std::function<void()>& operation)
{
  const ame_snapshot_t before = snapshot(f.proc);
  const auto memory_before = f.sim.memory;
  bool trapped = false;
  try {
    operation();
  } catch (const trap_t& trap) {
    trapped = true;
    assert(trap.cause() == cause);
  }
  assert(trapped);
  assert_same(before, snapshot(f.proc));
  assert(memory_before == f.sim.memory);
}

void test_state_access_and_encoded_exceptions_are_atomic()
{
  {
    fixture_t f;
    seed_architectural_state(f);
    f.proc.get_state()->mstatus->write(
      f.proc.get_state()->mstatus->read() & ~MSTATUS_MS);
    expect_atomic_trap(f, CAUSE_ILLEGAL_INSTRUCTION, [&] {
      execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(0, 1, 2));
    });
  }
  {
    fixture_t f;
    seed_architectural_state(f);
    f.proc.ZTU.release();
    const ame_snapshot_t before = snapshot(f.proc);
    const auto memory_before = f.sim.memory;
    bool trapped = false;
    try {
      execute_ztt(f.proc, ztt_opcode_t::madd_ew, operands(0, 1, 2));
    } catch (const trap_t& trap) {
      trapped = true;
      assert(trap.cause() == CAUSE_ILLEGAL_INSTRUCTION);
    }
    assert(trapped);
    assert_same(before, snapshot(f.proc));
    assert(memory_before == f.sim.memory);
  }
  {
    fixture_t f;
    seed_architectural_state(f);
    expect_atomic_trap(f, CAUSE_ILLEGAL_INSTRUCTION, [&] {
      execute_ztt(f.proc, ztt_opcode_t::mrowzip_ew, operands(0, 3, 3));
    });
  }
}

void test_formation_and_post_read_exceptions_are_atomic()
{
  {
    fixture_t f;
    seed_architectural_state(f);
    set_m(f.proc, 0, kU8, sequence(kU8, 1));
    set_m(f.proc, 30, kU64, filled(kU64, 7));
    f.set_clean();
    expect_atomic_trap(f, CAUSE_ILLEGAL_INSTRUCTION, [&] {
      execute_ztt(f.proc, ztt_opcode_t::mconv_ew, operands(30, 0));
    });
  }
  {
    fixture_t f;
    seed_architectural_state(f);
    f.proc.get_state()->XPR.write(4, ztt::kTileLength);
    expect_atomic_trap(f, CAUSE_ILLEGAL_INSTRUCTION, [&] {
      execute_ztt(f.proc, ztt_opcode_t::mrowbcast_ew_x,
                  operands(0, 4, 1));
    });
  }
}

void test_memory_exceptions_are_atomic()
{
  const struct memory_case_t {
    ztt_opcode_t opcode;
    reg_t cause;
    reg_t address;
  } cases[] = {
    {ztt_opcode_t::mls_rm, CAUSE_MISALIGNED_LOAD, DRAM_BASE + 1},
    {ztt_opcode_t::mss_rm, CAUSE_MISALIGNED_STORE, DRAM_BASE + 1},
    {ztt_opcode_t::mls_rm, CAUSE_LOAD_ACCESS,
     DRAM_BASE + kMemorySize - 32},
    {ztt_opcode_t::mss_rm, CAUSE_STORE_ACCESS,
     DRAM_BASE + kMemorySize - 32},
  };
  for (const auto& test : cases) {
    fixture_t f;
    seed_architectural_state(f);
    f.proc.get_state()->XPR.write(4, test.address);
    expect_atomic_trap(f, test.cause, [&] {
      execute_ztt(f.proc, test.opcode, operands(0, 4));
    });
  }
}

void test_unsupported_store_marks_dirty()
{
  fixture_t f;
  seed_architectural_state(f);
  assert(f.proc.ZTU.set_m_datatype(0, 0));
  f.proc.ZTU.write_amestatus(0);
  f.set_clean();
  f.proc.get_state()->XPR.write(4, DRAM_BASE);

  execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));

  assert(f.proc.ZTU.amestatus() == ztt::kAmestatusUn);
  assert((f.proc.get_state()->mstatus->read() & MSTATUS_MS) ==
         (reg_t(3) << 25));
}

void test_opaque_memory_ignores_datatype()
{
  fixture_t f;
  constexpr std::size_t unit_bytes = ztt::kUnitDatatypeBits / 8;
  constexpr std::size_t transfer_bytes = ztt::kMRegisterBits / 8;
  const reg_t address = DRAM_BASE + 256;
  assert(f.proc.ZTU.set_m_datatype(3, 0));
  const auto values = sequence(kU32, 0x10203040);
  assert(f.proc.ZTU.write_m_as(3, kU32, values));
  f.proc.get_state()->XPR.write(4, address);

  execute_ztt(f.proc, ztt_opcode_t::mss_1r, operands(3, 4));
  assert(f.proc.ZTU.m_datatype(3) == 0);
  for (std::size_t i = 0; i < values.size(); ++i)
    for (std::size_t byte = 0; byte < unit_bytes; ++byte)
      assert(f.sim.memory[256 + i * unit_bytes + byte] ==
             uint8_t(values[i] >> (8 * byte)));

  assert(f.proc.ZTU.write_m_as(3, kU32, filled(kU32, 0)));
  execute_ztt(f.proc, ztt_opcode_t::mls_1r, operands(3, 4));
  std::vector<elem_t> actual;
  assert(f.proc.ZTU.read_m_as(3, kU32, actual));
  assert(actual == values);
  assert(f.proc.ZTU.m_datatype(3) == 0);
  static_assert(transfer_bytes == ztt::kNumElements * unit_bytes);
}

void test_strided_negative_and_overlapping_addresses()
{
  fixture_t f;
  const auto values = sequence(kU32, 0x100);
  set_m(f.proc, 0, kU32, values);
  constexpr std::size_t segment_bytes =
    ztt::kTileLength * sizeof(uint32_t);
  const std::size_t base_offset = 1024;
  f.proc.get_state()->XPR.write(4, DRAM_BASE + base_offset);
  f.proc.get_state()->XPR.write(5, reg_t(0) - segment_bytes);
  execute_ztt(f.proc, ztt_opcode_t::mss_st, operands(0, 4, 5));
  for (std::size_t row = 0; row < ztt::kTileLength; ++row)
    for (std::size_t col = 0; col < ztt::kTileLength; ++col)
      for (std::size_t byte = 0; byte < sizeof(uint32_t); ++byte)
        assert(f.sim.memory[base_offset - row * segment_bytes +
                            col * sizeof(uint32_t) + byte] ==
               uint8_t(values[row * ztt::kTileLength + col] >> (8 * byte)));

  std::fill(f.sim.memory.begin(), f.sim.memory.end(), uint8_t(0));
  f.proc.get_state()->XPR.write(5, 0);
  execute_ztt(f.proc, ztt_opcode_t::mss_st, operands(0, 4, 5));
  const std::size_t last_row = ztt::kTileLength - 1;
  for (std::size_t col = 0; col < ztt::kTileLength; ++col)
    for (std::size_t byte = 0; byte < sizeof(uint32_t); ++byte)
      assert(f.sim.memory[base_offset + col * sizeof(uint32_t) + byte] ==
             uint8_t(values[last_row * ztt::kTileLength + col] >>
                     (8 * byte)));

  std::fill(f.sim.memory.begin(), f.sim.memory.end(), uint8_t(0));
  constexpr std::size_t overlapping_stride = segment_bytes / 2;
  f.proc.get_state()->XPR.write(5, overlapping_stride);
  execute_ztt(f.proc, ztt_opcode_t::mss_st, operands(0, 4, 5));
  std::vector<uint8_t> expected(
    segment_bytes + (ztt::kTileLength - 1) * overlapping_stride, 0);
  for (std::size_t row = 0; row < ztt::kTileLength; ++row)
    for (std::size_t col = 0; col < ztt::kTileLength; ++col)
      for (std::size_t byte = 0; byte < sizeof(uint32_t); ++byte)
        expected[row * overlapping_stride + col * sizeof(uint32_t) + byte] =
          uint8_t(values[row * ztt::kTileLength + col] >> (8 * byte));
  assert(std::equal(expected.begin(), expected.end(),
                    f.sim.memory.begin() + base_offset));
}

void test_preflight_capable_mmio_store()
{
  fixture_t f;
  const auto values = sequence(kU32, 0x11223300);
  set_m(f.proc, 0, kU32, values);
  f.proc.get_state()->XPR.write(4, kMmioBase);

  execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));
  assert(f.sim.mmio_store_calls == ztt::kNumElements * sizeof(uint32_t));
  for (std::size_t i = 0; i < values.size(); ++i)
    for (std::size_t byte = 0; byte < sizeof(uint32_t); ++byte)
      assert(f.sim.mmio_memory[i * sizeof(uint32_t) + byte] ==
             uint8_t(values[i] >> (8 * byte)));

  std::fill(f.sim.mmio_memory.begin(), f.sim.mmio_memory.end(), uint8_t(0x5a));
  f.sim.mmio_store_calls = 0;
  f.sim.reject_mmio_address = kMmioBase + 3 * sizeof(uint32_t);
  bool trapped = false;
  try {
    execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));
  } catch (const trap_store_access_fault&) {
    trapped = true;
  }
  assert(trapped);
  assert(f.sim.mmio_store_calls == 0);
  assert(std::all_of(f.sim.mmio_memory.begin(), f.sim.mmio_memory.end(),
                     [](uint8_t byte) { return byte == 0x5a; }));
}

void write_u64(std::vector<uint8_t>& memory, std::size_t offset,
               uint64_t value)
{
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    memory[offset + byte] = uint8_t(value >> (8 * byte));
}

void test_virtual_cross_page_fault_atomicity()
{
  fixture_t f("MSU");
  constexpr reg_t virtual_base = 0x400000;
  constexpr std::size_t root_offset = 0;
  constexpr std::size_t level1_offset = 0x1000;
  constexpr std::size_t level0_offset = 0x2000;
  constexpr std::size_t data0_offset = 0x4000;
  constexpr std::size_t data1_offset = 0x5000;
  const auto table_pte = [](reg_t address) {
    return ((address >> PGSHIFT) << PTE_PPN_SHIFT) | PTE_V;
  };
  const auto leaf_pte = [](reg_t address, bool writable) {
    return ((address >> PGSHIFT) << PTE_PPN_SHIFT) |
           PTE_V | PTE_R | (writable ? PTE_W : 0) | PTE_A | PTE_D;
  };
  write_u64(f.sim.memory, root_offset,
            table_pte(DRAM_BASE + level1_offset));
  write_u64(f.sim.memory, level1_offset + 2 * sizeof(uint64_t),
            table_pte(DRAM_BASE + level0_offset));
  write_u64(f.sim.memory, level0_offset,
            leaf_pte(DRAM_BASE + data0_offset, true));
  write_u64(f.sim.memory, level0_offset + sizeof(uint64_t),
            leaf_pte(DRAM_BASE + data1_offset, true));
  const reg_t satp = (reg_t(SATP_MODE_SV39) << 60) |
                     ((DRAM_BASE + root_offset) >> PGSHIFT);
  f.proc.set_max_vaddr_bits(39);
  f.proc.get_state()->satp->write(satp);
  f.proc.set_privilege(PRV_S, false);
  f.proc.get_mmu()->flush_tlb();

  const auto values = sequence(kU8, 0x20);
  set_m(f.proc, 0, kU8, values);
  constexpr std::size_t first_page_bytes = 17;
  const reg_t address = virtual_base + PGSIZE - first_page_bytes;
  f.proc.get_state()->XPR.write(4, address);
  execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t physical = i < first_page_bytes
      ? data0_offset + PGSIZE - first_page_bytes + i
      : data1_offset + i - first_page_bytes;
    assert(f.sim.memory[physical] == uint8_t(values[i]));
  }

  auto assert_fault_without_writes = [&](uint64_t second_leaf) {
    std::fill(f.sim.memory.begin() + data0_offset + PGSIZE - first_page_bytes,
              f.sim.memory.begin() + data0_offset + PGSIZE, uint8_t(0x5a));
    std::fill(f.sim.memory.begin() + data1_offset,
              f.sim.memory.begin() + data1_offset + values.size() -
                first_page_bytes, uint8_t(0x5a));
    write_u64(f.sim.memory, level0_offset + sizeof(uint64_t), second_leaf);
    f.proc.get_mmu()->flush_tlb();
    bool trapped = false;
    try {
      execute_ztt(f.proc, ztt_opcode_t::mss_rm, operands(0, 4));
    } catch (const trap_store_page_fault&) {
      trapped = true;
    }
    assert(trapped);
    assert(std::all_of(
      f.sim.memory.begin() + data0_offset + PGSIZE - first_page_bytes,
      f.sim.memory.begin() + data0_offset + PGSIZE,
      [](uint8_t byte) { return byte == 0x5a; }));
    assert(std::all_of(
      f.sim.memory.begin() + data1_offset,
      f.sim.memory.begin() + data1_offset + values.size() - first_page_bytes,
      [](uint8_t byte) { return byte == 0x5a; }));
  };
  assert_fault_without_writes(0);
  assert_fault_without_writes(leaf_pte(DRAM_BASE + data1_offset, false));
}

} // namespace

int main()
{
  test_complete_overlap_snapshot();
  test_packed_partial_overlap_snapshot();
  test_wide_partial_overlap_snapshot();
  test_two_destination_snapshot();
  test_state_access_and_encoded_exceptions_are_atomic();
  test_formation_and_post_read_exceptions_are_atomic();
  test_memory_exceptions_are_atomic();
  test_unsupported_store_marks_dirty();
  test_opaque_memory_ignores_datatype();
  test_strided_negative_and_overlapping_addresses();
  test_preflight_capable_mmio_store();
  test_virtual_cross_page_fault_atomicity();
  return 0;
}
