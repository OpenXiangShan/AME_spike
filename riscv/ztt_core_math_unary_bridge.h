// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
// Internal template for bit-preserving C ABI wrappers around CORE-MATH.

#include <stdint.h>
#include <string.h>

#if !defined(ZTT_CORE_MATH_SOURCE) || !defined(ZTT_CORE_MATH_TYPE) || \
    !defined(ZTT_CORE_MATH_BITS_TYPE) || \
    !defined(ZTT_CORE_MATH_FUNCTION) || !defined(ZTT_CORE_MATH_BRIDGE)
#error "CORE-MATH unary bridge configuration is incomplete"
#endif

#include ZTT_CORE_MATH_SOURCE

ZTT_CORE_MATH_BITS_TYPE ZTT_CORE_MATH_BRIDGE(ZTT_CORE_MATH_BITS_TYPE bits)
{
  ZTT_CORE_MATH_TYPE input;
  memcpy(&input, &bits, sizeof(input));
  const ZTT_CORE_MATH_TYPE output = ZTT_CORE_MATH_FUNCTION(input);
  ZTT_CORE_MATH_BITS_TYPE output_bits;
  memcpy(&output_bits, &output, sizeof(output_bits));
  return output_bits;
}
