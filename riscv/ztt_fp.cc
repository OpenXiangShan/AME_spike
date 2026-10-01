// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_fp.h"
#include "softfloat.h"
#include "specialize.h"
#include "ztt_state.h"
#include <algorithm>
#include <cstdint>

namespace {

using boost::multiprecision::cpp_int;

struct fp_format_t {
  unsigned exponent_bits;
  unsigned fraction_bits;
  int exponent_bias;
  bool has_infinity;
  bool finite_max_exponent;
  uint64_t canonical_nan;
  uint64_t quiet_bit;
};

fp_format_t fp_format(ztt_float_kind_t kind)
{
  switch (kind) {
    case ztt_float_kind_t::e4m3:
      return {4, 3, 7, false, true, defaultNaNE4M3, 0};
    case ztt_float_kind_t::e5m2:
      return {5, 2, 15, true, false, defaultNaNE5M2, 0x2};
    case ztt_float_kind_t::f16:
      return {5, 10, 15, true, false, defaultNaNF16UI, 0x0200};
    case ztt_float_kind_t::bf16:
      return {8, 7, 127, true, false, defaultNaNBF16UI, 0x0040};
    case ztt_float_kind_t::f32:
      return {8, 23, 127, true, false, defaultNaNF32UI, 0x00400000};
    case ztt_float_kind_t::f64:
      return {11, 52, 1023, true, false,
              defaultNaNF64UI,
              UINT64_C(0x0008000000000000)};
    case ztt_float_kind_t::none:
      return {};
  }
  return {};
}

uint64_t value_mask(const fp_format_t& format)
{
  const unsigned width = format.exponent_bits + format.fraction_bits + 1;
  return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
}

cpp_int round_positive_shift(const cpp_int& value, unsigned shift, bool sign,
                             uint32_t dtype, bool& inexact)
{
  if (shift == 0)
    return value;
  const cpp_int quotient = value >> shift;
  const cpp_int remainder = value - (quotient << shift);
  if (remainder == 0)
    return quotient;
  inexact = true;
  const cpp_int half = cpp_int(1) << (shift - 1);
  bool increment = false;
  switch (ztt_unit_t::datatype_rounding_mode(dtype)) {
    case 0: // RNE
      increment = remainder > half ||
                  (remainder == half && bool(quotient & 1));
      break;
    case 1: // RTZ
      break;
    case 2: // RDN
      increment = sign;
      break;
    case 3: // RUP
      increment = !sign;
      break;
    case 4: // RMM
      increment = remainder >= half;
      break;
    default:
      // RNO is rejected by tuple validation. Do not infer round-to-odd here.
      break;
  }
  return quotient + (increment ? 1 : 0);
}

} // namespace

uint_fast8_t ztt_to_softfloat_round(uint32_t dtype)
{
  switch (ztt_unit_t::datatype_rounding_mode(dtype)) {
    case 0: return softfloat_round_near_even;
    case 1: return softfloat_round_minMag;
    case 2: return softfloat_round_min;
    case 3: return softfloat_round_max;
    case 4: return softfloat_round_near_maxMag;
    default: return softfloat_round_near_even;
  }
}

uint64_t ztt_fp_canonical_nan(uint32_t dtype)
{
  return fp_format(ztt_unit_t::datatype_float_kind(dtype)).canonical_nan;
}

ztt_exact_fp_t ztt_decode_exact_fp(uint64_t bits, uint32_t dtype)
{
  const fp_format_t format =
    fp_format(ztt_unit_t::datatype_float_kind(dtype));
  if (format.exponent_bits == 0)
    return {};

  bits &= value_mask(format);
  const uint64_t fraction_mask =
    (UINT64_C(1) << format.fraction_bits) - 1;
  const uint64_t exponent_mask =
    (UINT64_C(1) << format.exponent_bits) - 1;
  const uint64_t fraction = bits & fraction_mask;
  const uint64_t exponent =
    (bits >> format.fraction_bits) & exponent_mask;

  ztt_exact_fp_t result;
  result.sign = (bits >> (format.fraction_bits + format.exponent_bits)) & 1;
  if (exponent == exponent_mask &&
      (!format.finite_max_exponent || fraction == fraction_mask)) {
    result.nan = fraction != 0 || format.finite_max_exponent;
    result.signaling_nan = result.nan && format.quiet_bit != 0 &&
                           (fraction & format.quiet_bit) == 0;
    result.infinity = !result.nan;
    return result;
  }
  if (exponent == 0) {
    result.significand = fraction;
    result.exponent = 1 - format.exponent_bias - int(format.fraction_bits);
  } else {
    result.significand = (cpp_int(1) << format.fraction_bits) + fraction;
    result.exponent = int(exponent) - format.exponent_bias -
                      int(format.fraction_bits);
  }
  return result;
}

ztt_fp_result_t ztt_fp_finalize(uint64_t bits, uint32_t dst_dtype,
                                uint8_t flags, bool inexact)
{
  const fp_format_t format =
    fp_format(ztt_unit_t::datatype_float_kind(dst_dtype));
  if (format.exponent_bits == 0)
    return {};

  bits &= value_mask(format);
  if (inexact) {
    flags |= softfloat_flag_inexact;
    const uint64_t exponent_mask =
      ((UINT64_C(1) << format.exponent_bits) - 1) << format.fraction_bits;
    if ((bits & exponent_mask) == 0)
      flags |= softfloat_flag_underflow;
  }
  if (flags & (softfloat_flag_overflow | softfloat_flag_underflow))
    flags |= softfloat_flag_inexact;
  return {bits, uint8_t(flags & ztt::kAmeFlagMask)};
}

ztt_fp_result_t ztt_round_pack(const ztt_exact_fp_t& exact,
                               uint32_t dst_dtype)
{
  const fp_format_t format =
    fp_format(ztt_unit_t::datatype_float_kind(dst_dtype));
  if (format.exponent_bits == 0)
    return {};

  const unsigned width = format.exponent_bits + format.fraction_bits + 1;
  const uint64_t sign_bit = exact.sign ? UINT64_C(1) << (width - 1) : 0;
  const uint64_t exponent_mask =
    (UINT64_C(1) << format.exponent_bits) - 1;
  const uint64_t fraction_mask =
    (UINT64_C(1) << format.fraction_bits) - 1;
  const auto canonical_nan = [&]() {
    return ztt_fp_finalize(ztt_fp_canonical_nan(dst_dtype), dst_dtype,
      exact.signaling_nan || exact.invalid ? softfloat_flag_invalid : 0);
  };
  const auto infinity = [&]() {
    return ztt_fp_finalize(sign_bit |
      (exponent_mask << format.fraction_bits), dst_dtype);
  };
  const auto max_finite = [&]() {
    const uint64_t exponent_field = format.finite_max_exponent
                                  ? exponent_mask : exponent_mask - 1;
    const uint64_t fraction_field = format.finite_max_exponent
                                  ? fraction_mask - 1 : fraction_mask;
    return sign_bit | (exponent_field << format.fraction_bits) |
           fraction_field;
  };
  const auto overflow = [&]() {
    uint64_t bits = max_finite();
    if (format.has_infinity) {
      const unsigned mode = ztt_unit_t::datatype_rounding_mode(dst_dtype);
      const bool to_infinity = mode == 0 || mode == 4 ||
        (mode == 3 && !exact.sign) || (mode == 2 && exact.sign);
      if (to_infinity)
        bits = sign_bit | (exponent_mask << format.fraction_bits);
    }
    return ztt_fp_finalize(bits, dst_dtype, softfloat_flag_overflow);
  };

  if (exact.nan)
    return canonical_nan();
  if (exact.infinity)
    // A required infinity is not a finite overflow.  Formats without an
    // infinity encoding saturate it as required by Ztt 6.2.3, retaining only
    // flags supplied by the operation that produced the infinity (for
    // example DZ for finite/zero division).
    return format.has_infinity ? infinity()
                               : ztt_fp_finalize(max_finite(), dst_dtype);
  if (exact.significand == 0)
    return ztt_fp_finalize(sign_bit, dst_dtype);

  const unsigned precision = format.fraction_bits + 1;
  const int minimum_exponent = 1 - format.exponent_bias;
  const int maximum_exponent = int(exponent_mask) - format.exponent_bias -
                               (format.finite_max_exponent ? 0 : 1);
  const unsigned significand_bits =
    unsigned(boost::multiprecision::msb(exact.significand)) + 1;
  int unbiased_exponent = exact.exponent + int(significand_bits) - 1;
  if (unbiased_exponent > maximum_exponent)
    return overflow();

  bool inexact = false;
  if (unbiased_exponent >= minimum_exponent) {
    const int shift = int(significand_bits) - int(precision);
    cpp_int rounded = shift > 0
      ? round_positive_shift(exact.significand, unsigned(shift), exact.sign,
                             dst_dtype, inexact)
      : exact.significand << unsigned(-shift);
    if (unsigned(boost::multiprecision::msb(rounded)) + 1 > precision) {
      rounded >>= 1;
      ++unbiased_exponent;
      if (unbiased_exponent > maximum_exponent)
        return overflow();
    }

    const cpp_int maximum_significand =
      (cpp_int(1) << format.fraction_bits) +
      (format.finite_max_exponent ? fraction_mask - 1 : fraction_mask);
    if (unbiased_exponent == maximum_exponent &&
        rounded > maximum_significand)
      return overflow();

    const uint64_t exponent_field =
      uint64_t(unbiased_exponent + format.exponent_bias);
    const uint64_t fraction_field =
      (rounded - (cpp_int(1) << format.fraction_bits)).convert_to<uint64_t>();
    const uint64_t bits = sign_bit |
      (exponent_field << format.fraction_bits) |
      (fraction_field & fraction_mask);
    return ztt_fp_finalize(bits, dst_dtype, 0, inexact);
  }

  // Tininess is detected after rounding, as required by Ztt v0.6 section 6.2.3.
  const int quantum_exponent = minimum_exponent - int(format.fraction_bits);
  const int shift = quantum_exponent - exact.exponent;
  cpp_int rounded = shift > 0
    ? round_positive_shift(exact.significand, unsigned(shift), exact.sign,
                           dst_dtype, inexact)
    : exact.significand << unsigned(-shift);
  const cpp_int normal_threshold = cpp_int(1) << format.fraction_bits;
  if (rounded >= normal_threshold) {
    const uint64_t bits = sign_bit | (UINT64_C(1) << format.fraction_bits);
    return ztt_fp_finalize(bits, dst_dtype, 0, inexact);
  }
  return ztt_fp_finalize(sign_bit | rounded.convert_to<uint64_t>(),
                         dst_dtype, 0, inexact);
}

ztt_fp_result_t ztt_fp_scale_exact(uint64_t bits, uint32_t dtype,
                                   int64_t amount)
{
  ztt_exact_fp_t exact = ztt_decode_exact_fp(bits, dtype);
  if (!exact.nan && !exact.infinity)
    exact.exponent += amount;
  return ztt_round_pack(exact, dtype);
}

ztt_fp_result_t ztt_fp_scale_acc_exact(uint64_t accumulator_bits,
                                       uint64_t value_bits,
                                       uint32_t dtype, int64_t amount)
{
  ztt_exact_fp_t lhs = ztt_decode_exact_fp(accumulator_bits, dtype);
  ztt_exact_fp_t rhs = ztt_decode_exact_fp(value_bits, dtype);
  if (lhs.nan || rhs.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = lhs.signaling_nan || rhs.signaling_nan;
    return ztt_round_pack(nan, dtype);
  }
  if (lhs.infinity || rhs.infinity) {
    if (lhs.infinity && rhs.infinity && lhs.sign != rhs.sign) {
      ztt_exact_fp_t nan;
      nan.nan = true;
      nan.invalid = true;
      return ztt_round_pack(nan, dtype);
    }
    return ztt_round_pack(lhs.infinity ? lhs : rhs, dtype);
  }
  rhs.exponent += amount;
  if (lhs.significand == 0 && rhs.significand == 0) {
    ztt_exact_fp_t zero;
    zero.sign = lhs.sign == rhs.sign ? lhs.sign
      : ztt_unit_t::datatype_rounding_mode(dtype) == 2;
    return ztt_round_pack(zero, dtype);
  }
  if (lhs.significand == 0)
    return ztt_round_pack(rhs, dtype);
  if (rhs.significand == 0)
    return ztt_round_pack(lhs, dtype);

  const int64_t common_exponent = std::min(lhs.exponent, rhs.exponent);
  cpp_int left = lhs.significand << unsigned(lhs.exponent - common_exponent);
  cpp_int right = rhs.significand << unsigned(rhs.exponent - common_exponent);
  if (lhs.sign)
    left = -left;
  if (rhs.sign)
    right = -right;
  const cpp_int sum = left + right;
  ztt_exact_fp_t result;
  if (sum == 0) {
    result.sign = ztt_unit_t::datatype_rounding_mode(dtype) == 2;
  } else {
    result.sign = sum < 0;
    result.significand = result.sign ? -sum : sum;
    result.exponent = common_exponent;
  }
  return ztt_round_pack(result, dtype);
}

namespace {

ztt_exact_fp_t exact_integer(const cpp_int& value)
{
  ztt_exact_fp_t exact;
  if (value < 0) {
    exact.sign = true;
    exact.significand = -value;
  } else {
    exact.significand = value;
  }
  return exact;
}

ztt_fp_result_t add_exact_values(ztt_exact_fp_t lhs, ztt_exact_fp_t rhs,
                                 uint32_t dst_dtype, int64_t scale_amount,
                                 bool absolute)
{
  if (lhs.nan || rhs.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = lhs.signaling_nan || rhs.signaling_nan;
    return ztt_round_pack(nan, dst_dtype);
  }
  if (lhs.infinity || rhs.infinity) {
    if (lhs.infinity && rhs.infinity && lhs.sign != rhs.sign) {
      ztt_exact_fp_t nan;
      nan.nan = true;
      nan.invalid = true;
      return ztt_round_pack(nan, dst_dtype);
    }
    ztt_exact_fp_t result = lhs.infinity ? lhs : rhs;
    if (absolute)
      result.sign = false;
    return ztt_round_pack(result, dst_dtype);
  }

  if (lhs.significand == 0 && rhs.significand == 0) {
    ztt_exact_fp_t zero;
    zero.sign = lhs.sign == rhs.sign
      ? lhs.sign
      : ztt_unit_t::datatype_rounding_mode(dst_dtype) == 2;
    if (absolute)
      zero.sign = false;
    return ztt_round_pack(zero, dst_dtype);
  }
  if (lhs.significand == 0) {
    rhs.exponent += scale_amount;
    if (absolute)
      rhs.sign = false;
    return ztt_round_pack(rhs, dst_dtype);
  }
  if (rhs.significand == 0) {
    lhs.exponent += scale_amount;
    if (absolute)
      lhs.sign = false;
    return ztt_round_pack(lhs, dst_dtype);
  }

  const int64_t common_exponent = std::min(lhs.exponent, rhs.exponent);
  cpp_int left = lhs.significand << unsigned(lhs.exponent - common_exponent);
  cpp_int right = rhs.significand << unsigned(rhs.exponent - common_exponent);
  if (lhs.sign)
    left = -left;
  if (rhs.sign)
    right = -right;
  const cpp_int sum = left + right;
  ztt_exact_fp_t result;
  if (sum == 0) {
    result.sign = ztt_unit_t::datatype_rounding_mode(dst_dtype) == 2;
  } else {
    result.sign = sum < 0;
    result.significand = result.sign ? -sum : sum;
    result.exponent = common_exponent + scale_amount;
    if (absolute)
      result.sign = false;
  }
  return ztt_round_pack(result, dst_dtype);
}

} // namespace

ztt_fp_result_t ztt_fp_from_integer(const cpp_int& value, uint32_t dst_dtype)
{
  return ztt_round_pack(exact_integer(value), dst_dtype);
}

ztt_fp_result_t ztt_fp_addsub_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                    uint32_t dst_dtype, bool subtract_rhs,
                                    int64_t scale_amount, bool absolute)
{
  ztt_exact_fp_t lhs = ztt_decode_exact_fp(lhs_bits, dst_dtype);
  ztt_exact_fp_t rhs = ztt_decode_exact_fp(rhs_bits, dst_dtype);
  if (subtract_rhs)
    rhs.sign = !rhs.sign;
  return add_exact_values(lhs, rhs, dst_dtype, scale_amount, absolute);
}

ztt_fp_result_t ztt_fp_addsub_mixed_exact(
    uint64_t lhs_bits, uint32_t lhs_dtype,
    uint64_t rhs_bits, uint32_t rhs_dtype,
    uint32_t dst_dtype, bool subtract_rhs)
{
  ztt_exact_fp_t lhs = ztt_decode_exact_fp(lhs_bits, lhs_dtype);
  ztt_exact_fp_t rhs = ztt_decode_exact_fp(rhs_bits, rhs_dtype);
  if (subtract_rhs)
    rhs.sign = !rhs.sign;
  return add_exact_values(lhs, rhs, dst_dtype, 0, false);
}

ztt_fp_result_t ztt_fp_sub_integer_exact(uint64_t fp_bits,
                                         const cpp_int& integer,
                                         uint32_t dst_dtype,
                                         bool integer_minus_fp)
{
  ztt_exact_fp_t fp = ztt_decode_exact_fp(fp_bits, dst_dtype);
  ztt_exact_fp_t integral = exact_integer(integer);
  if (integer_minus_fp) {
    fp.sign = !fp.sign;
    return add_exact_values(integral, fp, dst_dtype, 0, false);
  }
  integral.sign = !integral.sign;
  return add_exact_values(fp, integral, dst_dtype, 0, false);
}

ztt_fp_result_t ztt_fp_mul_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint32_t dst_dtype, bool negate)
{
  const ztt_exact_fp_t lhs = ztt_decode_exact_fp(lhs_bits, dst_dtype);
  const ztt_exact_fp_t rhs = ztt_decode_exact_fp(rhs_bits, dst_dtype);
  if (lhs.nan || rhs.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = lhs.signaling_nan || rhs.signaling_nan;
    return ztt_round_pack(nan, dst_dtype);
  }
  if ((lhs.infinity && rhs.significand == 0) ||
      (rhs.infinity && lhs.significand == 0)) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.invalid = true;
    return ztt_round_pack(nan, dst_dtype);
  }
  if (lhs.infinity || rhs.infinity) {
    ztt_exact_fp_t infinity;
    infinity.infinity = true;
    infinity.sign = lhs.sign ^ rhs.sign ^ negate;
    return ztt_round_pack(infinity, dst_dtype);
  }
  ztt_exact_fp_t result;
  result.sign = lhs.sign ^ rhs.sign ^ negate;
  result.significand = lhs.significand * rhs.significand;
  result.exponent = lhs.exponent + rhs.exponent;
  return ztt_round_pack(result, dst_dtype);
}

ztt_fp_result_t ztt_fp_fma_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint64_t addend_bits, uint32_t dst_dtype)
{
  const ztt_exact_fp_t lhs = ztt_decode_exact_fp(lhs_bits, dst_dtype);
  const ztt_exact_fp_t rhs = ztt_decode_exact_fp(rhs_bits, dst_dtype);
  const ztt_exact_fp_t addend = ztt_decode_exact_fp(addend_bits, dst_dtype);
  if (lhs.nan || rhs.nan || addend.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = lhs.signaling_nan || rhs.signaling_nan ||
                        addend.signaling_nan;
    return ztt_round_pack(nan, dst_dtype);
  }
  if ((lhs.infinity && rhs.significand == 0) ||
      (rhs.infinity && lhs.significand == 0)) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.invalid = true;
    return ztt_round_pack(nan, dst_dtype);
  }

  const bool product_sign = lhs.sign ^ rhs.sign;
  if (lhs.infinity || rhs.infinity) {
    if (addend.infinity && addend.sign != product_sign) {
      ztt_exact_fp_t nan;
      nan.nan = true;
      nan.invalid = true;
      return ztt_round_pack(nan, dst_dtype);
    }
    ztt_exact_fp_t infinity;
    infinity.infinity = true;
    infinity.sign = product_sign;
    return ztt_round_pack(infinity, dst_dtype);
  }
  if (addend.infinity)
    return ztt_round_pack(addend, dst_dtype);

  const cpp_int product = lhs.significand * rhs.significand;
  const int64_t product_exponent = lhs.exponent + rhs.exponent;
  if (product == 0 && addend.significand == 0) {
    ztt_exact_fp_t zero;
    zero.sign = product_sign == addend.sign ? product_sign
      : ztt_unit_t::datatype_rounding_mode(dst_dtype) == 2;
    return ztt_round_pack(zero, dst_dtype);
  }

  const int64_t common_exponent = std::min(product_exponent,
                                            addend.exponent);
  cpp_int product_term = product << unsigned(product_exponent - common_exponent);
  cpp_int addend_term = addend.significand <<
                        unsigned(addend.exponent - common_exponent);
  if (product_sign)
    product_term = -product_term;
  if (addend.sign)
    addend_term = -addend_term;
  const cpp_int sum = product_term + addend_term;
  ztt_exact_fp_t result;
  if (sum == 0) {
    result.sign = ztt_unit_t::datatype_rounding_mode(dst_dtype) == 2;
  } else {
    result.sign = sum < 0;
    result.significand = result.sign ? -sum : sum;
    result.exponent = common_exponent;
  }
  return ztt_round_pack(result, dst_dtype);
}

ztt_fp_result_t ztt_fp_div_exact(uint64_t lhs_bits, uint64_t rhs_bits,
                                 uint32_t dst_dtype)
{
  const ztt_exact_fp_t lhs = ztt_decode_exact_fp(lhs_bits, dst_dtype);
  const ztt_exact_fp_t rhs = ztt_decode_exact_fp(rhs_bits, dst_dtype);
  if (lhs.nan || rhs.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = lhs.signaling_nan || rhs.signaling_nan;
    return ztt_round_pack(nan, dst_dtype);
  }
  const bool sign = lhs.sign ^ rhs.sign;
  if ((lhs.infinity && rhs.infinity) ||
      (lhs.significand == 0 && rhs.significand == 0)) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.invalid = true;
    return ztt_round_pack(nan, dst_dtype);
  }
  if (lhs.infinity || rhs.significand == 0) {
    ztt_exact_fp_t infinity;
    infinity.infinity = true;
    infinity.sign = sign;
    const uint8_t flags = rhs.significand == 0 && !lhs.infinity
      ? softfloat_flag_infinite : 0;
    ztt_fp_result_t result = ztt_round_pack(infinity, dst_dtype);
    result.flags |= flags;
    return result;
  }
  if (rhs.infinity || lhs.significand == 0) {
    ztt_exact_fp_t zero;
    zero.sign = sign;
    return ztt_round_pack(zero, dst_dtype);
  }

  const fp_format_t format = fp_format(
    ztt_unit_t::datatype_float_kind(dst_dtype));
  const unsigned precision = format.fraction_bits + 1;
  const int bit_delta = int(boost::multiprecision::msb(lhs.significand)) -
                        int(boost::multiprecision::msb(rhs.significand));
  const unsigned guard_bits = precision + 5;
  const unsigned shift = unsigned(std::max(0, int(guard_bits) - bit_delta));
  const cpp_int numerator = lhs.significand << shift;
  cpp_int quotient = numerator / rhs.significand;
  const cpp_int remainder = numerator % rhs.significand;
  if (remainder != 0)
    quotient |= 1; // sticky information, below every destination rounding bit
  ztt_exact_fp_t exact;
  exact.sign = sign;
  exact.significand = quotient;
  exact.exponent = lhs.exponent - rhs.exponent - int64_t(shift);
  return ztt_round_pack(exact, dst_dtype);
}

ztt_fp_result_t ztt_fp_sqrt_exact(uint64_t bits, uint32_t dst_dtype)
{
  ztt_exact_fp_t source = ztt_decode_exact_fp(bits, dst_dtype);
  if (source.nan) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.signaling_nan = source.signaling_nan;
    return ztt_round_pack(nan, dst_dtype);
  }
  if (source.sign && (source.infinity || source.significand != 0)) {
    ztt_exact_fp_t nan;
    nan.nan = true;
    nan.invalid = true;
    return ztt_round_pack(nan, dst_dtype);
  }
  if (source.infinity || source.significand == 0)
    return ztt_round_pack(source, dst_dtype);

  if (source.exponent & 1) {
    source.significand <<= 1;
    --source.exponent;
  }
  const fp_format_t format = fp_format(
    ztt_unit_t::datatype_float_kind(dst_dtype));
  const unsigned precision = format.fraction_bits + 1;
  const unsigned root_bits =
    (unsigned(boost::multiprecision::msb(source.significand)) + 2) / 2;
  const unsigned wanted_bits = precision + 5;
  const unsigned shift = wanted_bits > root_bits ? wanted_bits - root_bits : 0;
  const cpp_int radicand = source.significand << (2 * shift);
  cpp_int root = sqrt(radicand);
  if (root * root != radicand)
    root |= 1; // sticky information below the destination rounding position
  ztt_exact_fp_t exact;
  exact.significand = root;
  exact.exponent = source.exponent / 2 - int64_t(shift);
  return ztt_round_pack(exact, dst_dtype);
}
