// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_DECODE_H
#define _RISCV_ZTT_DECODE_H

#include "decode.h"

inline unsigned ztt_acc_98(insn_t insn) { return insn.rd(); }
inline unsigned ztt_acc_1516(insn_t insn) { return insn.rs1(); }
inline unsigned ztt_acc_2021(insn_t insn) { return insn.rs2(); }

#endif
