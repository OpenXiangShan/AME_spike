// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_VALIDATION_H
#define _RISCV_ZTT_VALIDATION_H

#include "decode.h"
#include "ztt_execute.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

class processor_t;

struct ztt_m_meta_t {
  unsigned reg = 0;
  uint32_t dtype = 0;
};

struct ztt_acc_meta_t {
  unsigned reg = 0;
  uint32_t dtype = 0;
};

struct ztt_m_operand_t {
  ztt_m_meta_t meta;
  std::size_t squares = 0;
  std::size_t groups = 0;
  std::size_t group_registers = 0;
  std::size_t register_span = 0;
};

struct ztt_dtype_caps_t {
  bool valid = false;
  bool integer = false;
  bool floating = false;
  bool signed_type = false;
  bool saturating = false;
  bool ordered = false;
  bool raw_bits = false;
  bool semantic_zero = false;
  bool integral_exponent = false;
  bool packed = false;
  unsigned bits = 0;
};

enum class ztt_tuple_rule_t {
  none,
  one_numeric,
  same_numeric_unary,
  same_float_unary,
  reciprocal,
  prefix_add,
  same_numeric_binary,
  same_float_binary,
  log_bias_binary,
  same_integer_binary,
  scalar_numeric,
  scalar_float,
  log_bias_scalar,
  scalar_integer_raw,
  compare,
  compare_scalar,
  predicate_selected,
  data_integer_exponent,
  scalar_exponent,
  same_nonpacked,
  indexed_nonpacked,
  scatter_nonpacked,
  axis_id,
  shift_vector,
  shift_scalar,
  arithmetic_shift_vector,
  arithmetic_shift_scalar,
  zip_pair,
  conversion,
  packed_from_nonpacked,
  nonpacked_from_packed,
  matrix_accumulate,
  acc_move,
  broadcast_conversion,
};

struct ztt_dtype_tuple_t {
  bool has_dest = false;
  bool has_source1 = false;
  bool has_source2 = false;
  bool has_scalar = false;
  uint32_t dest = 0;
  uint32_t source1 = 0;
  uint32_t source2 = 0;
  uint32_t scalar = 0;
};

// Stage 1: checks constraints determined only by the instruction encoding.
void ztt_validate_encoded_fields(processor_t& proc, ztt_opcode_t opcode,
                                 insn_t insn);

// Stage 2: reads datatype metadata only. The encoded register fields must have
// already passed ztt_validate_encoded_fields().
ztt_m_meta_t ztt_read_m_meta(const processor_t& proc, unsigned reg);
ztt_acc_meta_t ztt_read_acc_meta(const processor_t& proc, unsigned reg);

ztt_dtype_caps_t ztt_dtype_caps(uint32_t dtype);
ztt_tuple_rule_t ztt_tuple_rule(ztt_opcode_t opcode);

bool ztt_supports_tuple(ztt_opcode_t opcode, const ztt_dtype_tuple_t& tuple);

// Stage 3: validates the exact operation/datatype tuple. Failure sets UN and
// returns false without interpreting register-group geometry.
bool ztt_validate_tuple(processor_t& proc, ztt_opcode_t opcode,
                        const ztt_dtype_tuple_t& tuple);

// Stage 4: validates base alignment and the complete physical-register span.
// This must only be called after ztt_validate_tuple() succeeds.
std::size_t ztt_instruction_square_count(
  std::initializer_list<ztt_m_meta_t> operands);
ztt_m_operand_t ztt_form_m_operand(const processor_t& proc, insn_t insn,
                                   const ztt_m_meta_t& meta,
                                   std::size_t instruction_squares);
ztt_m_operand_t ztt_form_m_operand(const processor_t& proc, insn_t insn,
                                   const ztt_m_meta_t& meta);
void ztt_form_acc_span(insn_t insn, unsigned base, unsigned count);

#endif
