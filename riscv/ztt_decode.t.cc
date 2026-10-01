// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "cfg.h"
#include "mmu.h"
#include "platform.h"
#include "processor.h"
#include "simif.h"
#include "ztt_encoding.h"
#include "encoding.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {
struct encoding_t {
  insn_bits_t match;
  insn_bits_t mask;
  const char* name;
};

// encoding.h is the single source of truth for instruction encodings.  The
// macro turns every DECLARE_INSN line into a table entry; the test filters the
// Ztt entries below instead of maintaining a second 138-entry list.
#define DECLARE_INSN(name, match, mask) {match, mask, #name},
const encoding_t all_encodings[] = {
#include "encoding.h"
};
#undef DECLARE_INSN

class test_sim_t final : public simif_t {
 public:
  test_sim_t() : memory(4096, 0) { debug_mmu = nullptr; }

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
} // namespace

int main()
{
  test_sim_t sim;
  processor_t proc("rv64i_ztt", "M", &sim.cfg, &sim, 0, false,
                   nullptr, std::cerr);
  sim.harts.emplace(0, &proc);

  std::size_t ztt_count = 0;
  for (const auto& encoding : all_encodings) {
    const std::string name(encoding.name);
    if (name.rfind("ztt_", 0) != 0)
      continue;
    ++ztt_count;

    // Put each canonical encoding at a distinct address.  load_insn() then
    // exercises the same fetch and decoder path used by processor_t::step().
    const reg_t address = DRAM_BASE + (ztt_count - 1) * sizeof(uint32_t);
    const uint32_t bits = uint32_t(encoding.match);
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      sim.memory[(ztt_count - 1) * sizeof(bits) + byte] = bits >> (byte * 8);
    const insn_fetch_t fetch = proc.get_mmu()->load_insn(address);
    insn_t fetched_insn = fetch.insn;
    const insn_bits_t fetched_bits = fetched_insn.bits();
    assert(fetched_bits == encoding.match);
    assert(fetch.func != &illegal_instruction);
  }

  // Keep this count tied to the extension registry and catch accidental
  // omissions when a new instruction is added to the encoding table.
  assert(ztt_count == 138);

  // Exercise the generated instruction wrapper, not only its registration in
  // the decoder.  These two instructions are safe to execute in isolation
  // and provide a smoke check for the complete fetch/decode/execute_ztt path.
  const std::size_t execute_offset = ztt_count * sizeof(uint32_t);
  const reg_t acquire_address = DRAM_BASE + execute_offset;
  const uint32_t acquire_bits = MATCH_ZTT_AME_ACQUIRE | (1u << 7);
  for (unsigned byte = 0; byte < sizeof(acquire_bits); ++byte)
    sim.memory[execute_offset + byte] =
        acquire_bits >> (byte * 8);

  proc.get_state()->mstatus->write(reg_t(1) << 25);
  const insn_fetch_t acquire = proc.get_mmu()->load_insn(acquire_address);
  assert(acquire.func != &illegal_instruction);
  assert(acquire.func(&proc, acquire.insn, acquire_address) ==
         acquire_address + sizeof(uint32_t));
  assert(proc.ZTU.owned());
  assert(proc.get_state()->XPR[1] == 1);

  const reg_t release_address = acquire_address + sizeof(uint32_t);
  const uint32_t release_bits = MATCH_ZTT_AME_RELEASE;
  for (unsigned byte = 0; byte < sizeof(release_bits); ++byte)
    sim.memory[execute_offset + sizeof(release_bits) + byte] =
        release_bits >> (byte * 8);

  const insn_fetch_t release = proc.get_mmu()->load_insn(release_address);
  assert(release.func != &illegal_instruction);
  assert(release.func(&proc, release.insn, release_address) ==
         release_address + sizeof(uint32_t));
  assert(!proc.ZTU.owned());
  return 0;
}
