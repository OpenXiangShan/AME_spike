// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_csr.h"
#include "processor.h"
#include "trap.h"
#include "ztt_config.h"
#include "ztt_state.h"

namespace {

constexpr reg_t kMsInitial = reg_t(1) << 25;
constexpr reg_t kMsDirty = reg_t(3) << 25;

void set_ms(processor_t& proc, reg_t value) noexcept
{
  state_t* const state = proc.get_state();
  state->mstatus->write((state->mstatus->read() & ~MSTATUS_MS) | value);
  if (state->v && state->vsstatus)
    state->vsstatus->write((state->vsstatus->read() & ~SSTATUS_MS) | value);
}

} // namespace

bool ztt_ame_state_enabled(processor_t& proc) noexcept
{
  state_t* const state = proc.get_state();
  if (!state->mstatus || (state->mstatus->read() & MSTATUS_MS) == 0)
    return false;
  return !state->v || !state->vsstatus ||
         (state->vsstatus->read() & SSTATUS_MS) != 0;
}

void ztt_set_state_initial(processor_t& proc) noexcept
{
  set_ms(proc, kMsInitial);
}

void ztt_mark_state_dirty(processor_t& proc) noexcept
{
  set_ms(proc, kMsDirty);
}

ztt_csr_t::ztt_csr_t(processor_t* proc, ztt_unit_t* unit, reg_t address,
                     ztt_csr_kind_t kind)
  : csr_t(proc, address), unit(unit), kind(kind)
{
}

bool ztt_csr_t::writable() const noexcept
{
  switch (kind) {
    case ztt_csr_kind_t::amestype:
    case ztt_csr_kind_t::amefflags:
    case ztt_csr_kind_t::amexsat:
    case ztt_csr_kind_t::amestatus:
      return true;
    default:
      return false;
  }
}

bool ztt_csr_t::requires_owner() const noexcept
{
  return kind != ztt_csr_kind_t::amenlen &&
         kind != ztt_csr_kind_t::ameudsz &&
         kind != ztt_csr_kind_t::ameown;
}

void ztt_csr_t::verify_permissions(insn_t insn, bool write) const
{
  // CC2, CC4, and CC5 are provisional writable CSRs even though the
  // placeholder addresses currently lie in a read-only custom range.
  csr_t::verify_permissions(insn, write && !writable());
  if ((write && !writable()) || !ztt_ame_state_enabled(*proc) ||
      (requires_owner() && !unit->owned()))
    throw trap_illegal_instruction(insn.bits());
}

reg_t ztt_csr_t::read() const noexcept
{
  switch (kind) {
    case ztt_csr_kind_t::amenlen: return ztt::kTileLength;
    case ztt_csr_kind_t::ameudsz: return ztt::kUnitDatatypeBits;
    case ztt_csr_kind_t::amestype: return unit->amestype();
    case ztt_csr_kind_t::ameown: return unit->owned() ? 1 : 0;
    case ztt_csr_kind_t::amefflags: return unit->amefflags();
    case ztt_csr_kind_t::amexsat: return unit->amexsat();
    case ztt_csr_kind_t::amestatus: return unit->amestatus();
  }
  return 0;
}

bool ztt_csr_t::unlogged_write(reg_t value) noexcept
{
  switch (kind) {
    case ztt_csr_kind_t::amestype: unit->write_amestype(value); break;
    case ztt_csr_kind_t::amefflags: unit->write_amefflags(value); break;
    case ztt_csr_kind_t::amexsat: unit->write_amexsat(value); break;
    case ztt_csr_kind_t::amestatus: unit->write_amestatus(value); break;
    default: return false;
  }
  ztt_mark_state_dirty(*proc);
  return true;
}
