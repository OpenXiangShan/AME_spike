// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_validation.h"
#include "processor.h"
#include "trap.h"
#include "ztt_config.h"
#include "ztt_fp_transcendental.h"
#include "ztt_state.h"
#include <algorithm>
#include <array>

namespace {

constexpr unsigned field_rd = 1;
constexpr unsigned field_rs1 = 2;
constexpr unsigned field_rs2 = 4;

struct instruction_desc_t {
  unsigned m_fields = 0;
  unsigned acc_fields = 0;
  ztt_tuple_rule_t tuple_rule = ztt_tuple_rule_t::none;
  bool distinct_rs1_rs2 = false;
  bool requires_rv64 = false;
};

[[noreturn]] void illegal(insn_t insn)
{
  throw trap_illegal_instruction(insn.bits());
}

instruction_desc_t instruction_desc(ztt_opcode_t opcode)
{
  using rule = ztt_tuple_rule_t;
  switch (opcode) {
    case ztt_opcode_t::ame_acquire:
    case ztt_opcode_t::ame_release:
      return {};
    case ztt_opcode_t::agettyp:
      return {0, field_rs1, rule::none};
    case ztt_opcode_t::asettyp:
      return {0, field_rd, rule::one_numeric};
    case ztt_opcode_t::mgettyp:
      return {field_rs1, 0, rule::none};
    case ztt_opcode_t::msettyp:
      return {field_rd, 0, rule::one_numeric};

    case ztt_opcode_t::mabs_ew:
      return {field_rd | field_rs1, 0, rule::same_numeric_unary};
    case ztt_opcode_t::mcos_ew:
    case ztt_opcode_t::mexp2_ew:
    case ztt_opcode_t::mfrintm_ew:
    case ztt_opcode_t::mfrintn_ew:
    case ztt_opcode_t::mfrintp_ew:
    case ztt_opcode_t::mfrintz_ew:
    case ztt_opcode_t::mlog2_ew:
    case ztt_opcode_t::mrsqrt_ew:
    case ztt_opcode_t::msin_ew:
    case ztt_opcode_t::msqrt_ew:
    case ztt_opcode_t::mtanh_ew:
      return {field_rd | field_rs1, 0, rule::same_float_unary};
    case ztt_opcode_t::mrec_ew:
      return {field_rd | field_rs1, 0, rule::reciprocal};

    case ztt_opcode_t::mand_ew:
    case ztt_opcode_t::mandnot_ew:
    case ztt_opcode_t::mor_ew:
    case ztt_opcode_t::mornot_ew:
    case ztt_opcode_t::mxor_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::same_integer_binary};
    case ztt_opcode_t::mand_ew_x:
    case ztt_opcode_t::mandnot_ew_x:
    case ztt_opcode_t::mor_ew_x:
    case ztt_opcode_t::mornot_ew_x:
    case ztt_opcode_t::mxor_ew_x:
      return {field_rd | field_rs2, 0, rule::scalar_integer_raw};

    case ztt_opcode_t::mcmpge_ew:
    case ztt_opcode_t::mcmplt_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::compare};
    case ztt_opcode_t::mcmpge_ew_x:
    case ztt_opcode_t::mcmplt_ew_x:
      return {field_rd | field_rs2, 0, rule::compare_scalar};
    case ztt_opcode_t::mcmovge_ew:
    case ztt_opcode_t::mcmovlt_ew:
    case ztt_opcode_t::mselge_ew:
    case ztt_opcode_t::msellt_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::predicate_selected};

    case ztt_opcode_t::mldexp_ew:
    case ztt_opcode_t::mldexpacc_ew:
    case ztt_opcode_t::mrdexp_ew:
    case ztt_opcode_t::mrdexpacc_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::data_integer_exponent};
    case ztt_opcode_t::mldexp_ew_x:
    case ztt_opcode_t::mldexpacc_ew_x:
      return {field_rd | field_rs2, 0, rule::scalar_exponent};

    case ztt_opcode_t::mcolbcast_ew_x:
    case ztt_opcode_t::mcolshift_ew_x:
    case ztt_opcode_t::mrowbcast_ew_x:
    case ztt_opcode_t::mrowshift_ew_x:
      return {field_rd | field_rs2, 0, rule::same_nonpacked};
    case ztt_opcode_t::mcolgather_ew:
    case ztt_opcode_t::mrowgather_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::indexed_nonpacked};
    case ztt_opcode_t::mcolscatadd_ew:
    case ztt_opcode_t::mrowscatadd_ew:
    case ztt_opcode_t::mcolscatmax_ew:
    case ztt_opcode_t::mrowscatmax_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::scatter_nonpacked};
    case ztt_opcode_t::mcolid_ew:
    case ztt_opcode_t::mrowid_ew:
      return {field_rd, 0, rule::axis_id};
    case ztt_opcode_t::mcolunzip_ew:
    case ztt_opcode_t::mcolzip_ew:
    case ztt_opcode_t::mrowunzip_ew:
    case ztt_opcode_t::mrowzip_ew:
      return {field_rs1 | field_rs2, 0, rule::zip_pair, true};

    case ztt_opcode_t::msll_ew:
    case ztt_opcode_t::msrl_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::shift_vector};
    case ztt_opcode_t::msll_ew_x:
    case ztt_opcode_t::msrl_ew_x:
      return {field_rd | field_rs2, 0, rule::shift_scalar};
    case ztt_opcode_t::msra_ew:
      return {field_rd | field_rs1 | field_rs2, 0,
              rule::arithmetic_shift_vector};
    case ztt_opcode_t::msra_ew_x:
      return {field_rd | field_rs2, 0, rule::arithmetic_shift_scalar};

    case ztt_opcode_t::mconv_ew:
      return {field_rd | field_rs1, 0, rule::conversion};
    case ztt_opcode_t::mpack_ew_x:
      return {field_rd | field_rs2, 0, rule::packed_from_nonpacked};
    case ztt_opcode_t::munpack_ew_x:
      return {field_rd | field_rs2, 0, rule::nonpacked_from_packed};

    case ztt_opcode_t::mmov_m_m:
      return {field_rd | field_rs1, 0, rule::same_numeric_unary};
    case ztt_opcode_t::mmov_m_a:
      return {field_rd, field_rs1, rule::acc_move};
    case ztt_opcode_t::mmov_a_m:
      return {field_rs1, field_rd, rule::acc_move};
    case ztt_opcode_t::mbcast_m_x:
      return {field_rd, 0, rule::broadcast_conversion};

    case ztt_opcode_t::mmulacc_2d:
    case ztt_opcode_t::mmulaccneg_2d:
    case ztt_opcode_t::mmulatacc_2d:
    case ztt_opcode_t::mmulataccneg_2d:
    case ztt_opcode_t::mmulbtacc_2d:
    case ztt_opcode_t::mmulbtaccneg_2d:
      return {field_rs1 | field_rs2, field_rd, rule::matrix_accumulate};

    case ztt_opcode_t::mprefixadd_col:
    case ztt_opcode_t::mprefixadd_row:
      return {field_rd | field_rs1, 0, rule::prefix_add};
    case ztt_opcode_t::mprefixmax_col:
    case ztt_opcode_t::mprefixmax_row:
    case ztt_opcode_t::mreduceadd_col:
    case ztt_opcode_t::mreduceadd_row:
    case ztt_opcode_t::mreducemax_col:
    case ztt_opcode_t::mreducemax_row:
    case ztt_opcode_t::mreducemin_col:
    case ztt_opcode_t::mreducemin_row:
      return {field_rd | field_rs1, 0, rule::same_numeric_unary};

    case ztt_opcode_t::mls_cm:
    case ztt_opcode_t::mls_rm:
    case ztt_opcode_t::mls_st:
    case ztt_opcode_t::mls_tst:
    case ztt_opcode_t::mss_cm:
    case ztt_opcode_t::mss_rm:
    case ztt_opcode_t::mss_st:
    case ztt_opcode_t::mss_tst:
    case ztt_opcode_t::mzero_2d_m:
      return {field_rd, 0, rule::one_numeric};
    case ztt_opcode_t::mzero_2d_acc:
      return {0, field_rd, rule::one_numeric};

    case ztt_opcode_t::mls_1r:
    case ztt_opcode_t::mss_1r:
    case ztt_opcode_t::mmove8_m_x:
    case ztt_opcode_t::mmove16_m_x:
    case ztt_opcode_t::mmove32_m_x:
      return {field_rd, 0, rule::none};
    case ztt_opcode_t::mmove64_m_x:
      return {field_rd, 0, rule::none, false, true};
    case ztt_opcode_t::mmove8_x_m:
    case ztt_opcode_t::mmove16_x_m:
    case ztt_opcode_t::mmove32_x_m:
      return {field_rs1, 0, rule::none};
    case ztt_opcode_t::mmove64_x_m:
      return {field_rs1, 0, rule::none, false, true};

    case ztt_opcode_t::mabsdiff_ew:
    case ztt_opcode_t::madd_ew:
    case ztt_opcode_t::mhdiff_ew:
    case ztt_opcode_t::mmax_ew:
    case ztt_opcode_t::mmean_ew:
    case ztt_opcode_t::mmin_ew:
    case ztt_opcode_t::mmul_ew:
    case ztt_opcode_t::mmulacc_ew:
    case ztt_opcode_t::mmulaccneg_ew:
    case ztt_opcode_t::mmuladd_ew:
    case ztt_opcode_t::mmulneg_ew:
    case ztt_opcode_t::mmulsub_ew:
    case ztt_opcode_t::msub_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::same_numeric_binary};

    case ztt_opcode_t::mlog2sub_ew:
    case ztt_opcode_t::msublog2_ew:
      return {field_rd | field_rs1 | field_rs2, 0, rule::log_bias_binary};

    case ztt_opcode_t::mabsdiff_ew_x:
    case ztt_opcode_t::madd_ew_x:
    case ztt_opcode_t::mhdiff_ew_x:
    case ztt_opcode_t::mmax_ew_x:
    case ztt_opcode_t::mmean_ew_x:
    case ztt_opcode_t::mmin_ew_x:
    case ztt_opcode_t::mmul_ew_x:
    case ztt_opcode_t::mmulacc_ew_x:
    case ztt_opcode_t::mmulaccneg_ew_x:
    case ztt_opcode_t::mmuladd_ew_x:
    case ztt_opcode_t::mmulneg_ew_x:
    case ztt_opcode_t::mmulsub_ew_x:
    case ztt_opcode_t::msub_ew_x:
      return {field_rd | field_rs2, 0, rule::scalar_numeric};

    case ztt_opcode_t::mlog2sub_ew_x:
    case ztt_opcode_t::msublog2_ew_x:
      return {field_rd | field_rs2, 0, rule::log_bias_scalar};
  }
  illegal(insn_t(0));
}

bool same_storage_shape(uint32_t a, uint32_t b)
{
  return ztt_unit_t::datatype_bits(a) == ztt_unit_t::datatype_bits(b);
}

// RNO is a property of a result-producing floating-point operation.  It must
// not make operations that only copy, compare, index, or access raw storage
// unsupported, nor should an RNO source be rejected merely because its value
// is converted into a destination with a different rounding mode.
bool destination_uses_float_rounding(ztt_opcode_t opcode)
{
  switch (opcode) {
    case ztt_opcode_t::asettyp:
    case ztt_opcode_t::msettyp:
    case ztt_opcode_t::mabs_ew:
    case ztt_opcode_t::mcolbcast_ew_x:
    case ztt_opcode_t::mcolgather_ew:
    case ztt_opcode_t::mcolscatmax_ew:
    case ztt_opcode_t::mcolshift_ew_x:
    case ztt_opcode_t::mcolunzip_ew:
    case ztt_opcode_t::mcolzip_ew:
    case ztt_opcode_t::mcmovge_ew:
    case ztt_opcode_t::mcmovlt_ew:
    case ztt_opcode_t::mmax_ew:
    case ztt_opcode_t::mmax_ew_x:
    case ztt_opcode_t::mmin_ew:
    case ztt_opcode_t::mmin_ew_x:
    case ztt_opcode_t::mmov_m_a:
    case ztt_opcode_t::mmov_a_m:
    case ztt_opcode_t::mmov_m_m:
    case ztt_opcode_t::mls_cm:
    case ztt_opcode_t::mls_rm:
    case ztt_opcode_t::mls_st:
    case ztt_opcode_t::mls_tst:
    case ztt_opcode_t::mrowbcast_ew_x:
    case ztt_opcode_t::mrowgather_ew:
    case ztt_opcode_t::mrowscatmax_ew:
    case ztt_opcode_t::mrowshift_ew_x:
    case ztt_opcode_t::mrowunzip_ew:
    case ztt_opcode_t::mrowzip_ew:
    case ztt_opcode_t::mss_cm:
    case ztt_opcode_t::mss_rm:
    case ztt_opcode_t::mss_st:
    case ztt_opcode_t::mss_tst:
    case ztt_opcode_t::mprefixmax_col:
    case ztt_opcode_t::mprefixmax_row:
    case ztt_opcode_t::mreducemax_col:
    case ztt_opcode_t::mreducemax_row:
    case ztt_opcode_t::mreducemin_col:
    case ztt_opcode_t::mreducemin_row:
    case ztt_opcode_t::mselge_ew:
    case ztt_opcode_t::msellt_ew:
    case ztt_opcode_t::mzero_2d_acc:
    case ztt_opcode_t::mzero_2d_m:
      return false;
    default:
      break;
  }
  return true;
}

} // namespace

void ztt_validate_encoded_fields(processor_t& proc, ztt_opcode_t opcode,
                                 insn_t insn)
{
  const instruction_desc_t desc = instruction_desc(opcode);
  const std::array<unsigned, 3> regs{{unsigned(insn.rd()), unsigned(insn.rs1()),
                                      unsigned(insn.rs2())}};
  const std::array<unsigned, 3> fields{{field_rd, field_rs1, field_rs2}};
  for (std::size_t i = 0; i < regs.size(); ++i) {
    if ((desc.m_fields & fields[i]) && regs[i] >= ztt::kNumMRegisters)
      illegal(insn);
    if ((desc.acc_fields & fields[i]) && regs[i] >= ztt::kNumAccRegisters)
      illegal(insn);
  }
  if (desc.distinct_rs1_rs2 && insn.rs1() == insn.rs2())
    illegal(insn);
  if (desc.requires_rv64 && proc.get_xlen() < 64)
    illegal(insn);
  if (opcode == ztt_opcode_t::ame_release &&
      (insn.rd() != 0 || insn.rs1() != 0))
    illegal(insn);
}

ztt_m_meta_t ztt_read_m_meta(const processor_t& proc, unsigned reg)
{
  return {reg, proc.ZTU.m_datatype(reg)};
}

ztt_acc_meta_t ztt_read_acc_meta(const processor_t& proc, unsigned reg)
{
  return {reg, proc.ZTU.acc_datatype(reg)};
}

ztt_dtype_caps_t ztt_dtype_caps(uint32_t dtype)
{
  ztt_dtype_caps_t caps;
  caps.valid = ztt_unit_t::datatype_supported(dtype);
  if (!caps.valid)
    return caps;
  caps.integer = ztt_unit_t::datatype_integer(dtype);
  caps.floating = ztt_unit_t::datatype_floating(dtype);
  caps.signed_type = ztt_unit_t::datatype_signed(dtype);
  caps.saturating = ztt_unit_t::datatype_saturating(dtype);
  caps.ordered = caps.integer || caps.floating;
  caps.raw_bits = true;
  caps.semantic_zero = true;
  caps.integral_exponent = caps.integer;
  caps.bits = ztt_unit_t::datatype_bits(dtype);
  caps.packed = caps.bits < ztt::kUnitDatatypeBits;
  return caps;
}

ztt_tuple_rule_t ztt_tuple_rule(ztt_opcode_t opcode)
{
  return instruction_desc(opcode).tuple_rule;
}

bool ztt_supports_tuple(ztt_opcode_t opcode, const ztt_dtype_tuple_t& tuple)
{
  if (unsigned(opcode) > unsigned(ztt_opcode_t::mzero_2d_m))
    return false;

  const auto has_operands = [&](bool dest, bool source1 = false,
                                bool source2 = false, bool scalar = false) {
    return tuple.has_dest == dest && tuple.has_source1 == source1 &&
           tuple.has_source2 == source2 && tuple.has_scalar == scalar;
  };
  const auto d = ztt_dtype_caps(tuple.dest);
  const auto a = ztt_dtype_caps(tuple.source1);
  const auto b = ztt_dtype_caps(tuple.source2);
  const auto s = ztt_dtype_caps(tuple.scalar);
  const auto numeric = [](const ztt_dtype_caps_t& caps) {
    return caps.integer || caps.floating;
  };
  const auto present_valid = [&](bool present, const ztt_dtype_caps_t& caps) {
    return !present || caps.valid;
  };

  bool valid = present_valid(tuple.has_dest, d) &&
               present_valid(tuple.has_source1, a) &&
               present_valid(tuple.has_source2, b) &&
               present_valid(tuple.has_scalar, s);
  const bool destination_rno = tuple.has_dest && d.floating &&
    ztt_unit_t::datatype_rounding_mode(tuple.dest) == 5;
  if (destination_rno && destination_uses_float_rounding(opcode))
    valid = false;
  if (valid) {
    switch (ztt_tuple_rule(opcode)) {
      case ztt_tuple_rule_t::none:
        valid = has_operands(false);
        break;
      case ztt_tuple_rule_t::one_numeric:
        valid = has_operands(true) && numeric(d);
        break;
      case ztt_tuple_rule_t::same_numeric_unary:
        valid = has_operands(true, true) && numeric(d) && d.valid &&
                tuple.dest == tuple.source1;
        break;
      case ztt_tuple_rule_t::same_float_unary:
        valid = has_operands(true, true) && d.floating &&
                tuple.dest == tuple.source1;
        break;
      case ztt_tuple_rule_t::reciprocal:
        valid = has_operands(true, true) && d.floating &&
          (tuple.dest == tuple.source1 ||
           (ztt_unit_t::datatype_float_kind(tuple.dest) == ztt_float_kind_t::f16 &&
            ((tuple.dest ^ tuple.source1) & ~(0xfu << 22)) == 0 &&
            ztt_unit_t::datatype_rounding_mode(tuple.source1) < 5));
        break;
      case ztt_tuple_rule_t::prefix_add:
        valid = has_operands(true, true) && numeric(d) &&
          (tuple.dest == tuple.source1 ||
           (d.integer && d.bits == 4 && d.saturating && !a.saturating &&
            (tuple.dest ^ tuple.source1) == (1u << 29)));
        break;
      case ztt_tuple_rule_t::same_numeric_binary:
        valid = has_operands(true, true, true) &&
                numeric(d) && tuple.dest == tuple.source1 &&
                tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::same_float_binary:
        valid = has_operands(true, true, true) &&
                d.floating && a.floating && b.floating &&
                tuple.dest == tuple.source1 && tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::log_bias_binary:
        valid = has_operands(true, true, true) &&
                d.floating && a.floating && tuple.dest == tuple.source1 &&
                ((b.floating && tuple.dest == tuple.source2) || b.integer);
        break;
      case ztt_tuple_rule_t::same_integer_binary:
        valid = has_operands(true, true, true) &&
                d.integer && tuple.dest == tuple.source1 &&
                tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::scalar_numeric:
        valid = has_operands(true, false, true, true) &&
                numeric(d) && numeric(s) && tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::scalar_float:
        valid = has_operands(true, false, true, true) &&
                d.floating && b.floating && s.floating &&
                tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::log_bias_scalar:
        valid = has_operands(true, false, true, true) &&
                d.floating && b.floating && tuple.dest == tuple.source2 &&
                ((s.floating && tuple.dest == tuple.scalar) || s.integer);
        break;
      case ztt_tuple_rule_t::scalar_integer_raw:
        valid = has_operands(true, false, true, true) &&
                d.integer && numeric(s) && tuple.dest == tuple.source2;
        break;
      case ztt_tuple_rule_t::compare:
        valid = has_operands(true, true, true) &&
                d.integer && numeric(a) && tuple.source1 == tuple.source2 &&
                same_storage_shape(tuple.dest, tuple.source1);
        break;
      case ztt_tuple_rule_t::compare_scalar:
        valid = has_operands(true, false, true, true) &&
                d.integer && numeric(b) && numeric(s) &&
                same_storage_shape(tuple.dest, tuple.source2);
        break;
      case ztt_tuple_rule_t::predicate_selected:
        valid = has_operands(true, true, true) &&
                a.ordered && numeric(d) && tuple.dest == tuple.source2 &&
                same_storage_shape(tuple.dest, tuple.source1);
        break;
      case ztt_tuple_rule_t::data_integer_exponent:
        valid = has_operands(true, true, true) &&
                numeric(d) && b.integral_exponent &&
                tuple.dest == tuple.source1 &&
                same_storage_shape(tuple.dest, tuple.source2);
        break;
      case ztt_tuple_rule_t::scalar_exponent:
        valid = has_operands(true, true) && numeric(d) &&
                tuple.dest == tuple.source1;
        break;
      case ztt_tuple_rule_t::same_nonpacked:
        valid = has_operands(true, true) && numeric(d) &&
                tuple.dest == tuple.source1 && !d.packed;
        break;
      case ztt_tuple_rule_t::indexed_nonpacked:
        valid = has_operands(true, true, true) &&
                numeric(d) && b.integer && tuple.dest == tuple.source1 &&
                !d.packed && !b.packed;
        break;
      case ztt_tuple_rule_t::scatter_nonpacked:
        valid = has_operands(true, true, true) &&
                numeric(d) && numeric(a) && b.integer &&
                (d.integer == a.integer) && !d.packed && !a.packed &&
                !b.packed;
        break;
      case ztt_tuple_rule_t::axis_id:
        valid = has_operands(true) && d.integer &&
                ((d.signed_type && d.bits > 2) || (!d.signed_type && d.bits >= 2));
        break;
      case ztt_tuple_rule_t::shift_vector:
        valid = has_operands(true, true, true) &&
                d.integer && a.integer && b.integer &&
                same_storage_shape(tuple.source1, tuple.source2);
        break;
      case ztt_tuple_rule_t::shift_scalar:
        valid = has_operands(true, true) && d.integer && a.integer;
        break;
      case ztt_tuple_rule_t::arithmetic_shift_vector:
        valid = has_operands(true, true, true) &&
                d.integer && a.integer && a.signed_type && b.integer &&
                same_storage_shape(tuple.source1, tuple.source2);
        break;
      case ztt_tuple_rule_t::arithmetic_shift_scalar:
        valid = has_operands(true, true) && d.integer && a.integer &&
                a.signed_type;
        break;
      case ztt_tuple_rule_t::zip_pair:
        valid = has_operands(false, true, true) && numeric(a) &&
                tuple.source1 == tuple.source2 && !a.packed;
        break;
      case ztt_tuple_rule_t::conversion:
        valid = has_operands(true, true) && numeric(d) && numeric(a);
        break;
      case ztt_tuple_rule_t::packed_from_nonpacked:
        valid = has_operands(true, false, true) && numeric(d) && numeric(b) &&
                d.packed && !b.packed;
        break;
      case ztt_tuple_rule_t::nonpacked_from_packed:
        valid = has_operands(true, false, true) && numeric(d) && numeric(b) &&
                !d.packed && b.packed;
        break;
      case ztt_tuple_rule_t::matrix_accumulate:
        valid = has_operands(true, true, true) &&
                numeric(d) && numeric(a) && tuple.source1 == tuple.source2 &&
                (d.integer == a.integer);
        break;
      case ztt_tuple_rule_t::acc_move:
        valid = has_operands(true, true) && numeric(d) &&
                tuple.dest == tuple.source1 &&
                ztt_unit_t::square_count(tuple.dest) <= ztt::kNumAccRegisters;
        break;
      case ztt_tuple_rule_t::broadcast_conversion:
        valid = has_operands(true, false, false, true) && numeric(d) && numeric(s);
        break;
    }
  }
  ztt_fp_trans_op trans_op;
  if (valid && d.floating && ztt_fp_transcendental_operation(opcode, trans_op)) {
    const uint32_t source_dtype = tuple.has_source1 ? tuple.source1 : tuple.source2;
    valid = ztt_fp_transcendental_supported(trans_op, source_dtype, tuple.dest);
  }
  return valid;
}

bool ztt_validate_tuple(processor_t& proc, ztt_opcode_t opcode,
                        const ztt_dtype_tuple_t& tuple)
{
  const bool supported = ztt_supports_tuple(opcode, tuple);
  if (!supported)
    proc.ZTU.set_unsupported();
  return supported;
}

std::size_t ztt_instruction_square_count(
  std::initializer_list<ztt_m_meta_t> operands)
{
  std::size_t squares = 1;
  for (const auto& operand : operands)
    squares = std::max(squares, ztt_unit_t::square_count(operand.dtype));
  return squares;
}

ztt_m_operand_t ztt_form_m_operand(const processor_t& proc, insn_t insn,
                                   const ztt_m_meta_t& meta,
                                   std::size_t instruction_squares)
{
  const std::size_t native_squares = ztt_unit_t::square_count(meta.dtype);
  const std::size_t group_registers = ztt_unit_t::group_registers(meta.dtype);
  if (!proc.ZTU.valid_m_group(meta.reg, meta.dtype) ||
      instruction_squares == 0 || instruction_squares % native_squares != 0)
    illegal(insn);
  const std::size_t groups = instruction_squares / native_squares;
  const std::size_t register_span = groups * group_registers;
  // Each wide square is an independently aligned register group.  The
  // operand base therefore follows the per-square group size; register_span
  // only controls the complete physical span and its bounds.
  if (meta.reg % group_registers != 0 ||
      meta.reg + register_span > ztt::kNumMRegisters)
    illegal(insn);
  return {meta, instruction_squares, groups, group_registers, register_span};
}

ztt_m_operand_t ztt_form_m_operand(const processor_t& proc, insn_t insn,
                                   const ztt_m_meta_t& meta)
{
  return ztt_form_m_operand(proc, insn, meta,
                            ztt_unit_t::square_count(meta.dtype));
}

void ztt_form_acc_span(insn_t insn, unsigned base, unsigned count)
{
  if (count == 0 || base >= ztt::kNumAccRegisters ||
      base + count > ztt::kNumAccRegisters)
    illegal(insn);
}
