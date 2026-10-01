// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_CSR_H
#define _RISCV_ZTT_CSR_H

#include "csrs.h"

class ztt_unit_t;

enum class ztt_csr_kind_t {
  amenlen,
  ameudsz,
  amestype,
  ameown,
  amefflags,
  amexsat,
  amestatus,
};

class ztt_csr_t final : public csr_t {
 public:
  ztt_csr_t(processor_t* proc, ztt_unit_t* unit, reg_t address,
            ztt_csr_kind_t kind);
  void verify_permissions(insn_t insn, bool write) const override;
  reg_t read() const noexcept override;

 protected:
  bool unlogged_write(reg_t value) noexcept override;

 private:
  bool writable() const noexcept;
  bool requires_owner() const noexcept;

  ztt_unit_t* unit;
  ztt_csr_kind_t kind;
};

bool ztt_ame_state_enabled(processor_t& proc) noexcept;
void ztt_set_state_initial(processor_t& proc) noexcept;
void ztt_mark_state_dirty(processor_t& proc) noexcept;

#endif
