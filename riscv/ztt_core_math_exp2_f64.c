// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
#define ZTT_CORE_MATH_SOURCE \
  "../core-math/src/binary64/exp2/exp2.c"
#define ZTT_CORE_MATH_TYPE double
#define ZTT_CORE_MATH_BITS_TYPE uint64_t
#define ZTT_CORE_MATH_FUNCTION cr_exp2
#define ZTT_CORE_MATH_BRIDGE ztt_core_math_exp2_f64
#include "ztt_core_math_unary_bridge.h"
