// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#define ZTT_CORE_MATH_SOURCE \
  "../core-math/src/binary32/cos/cosf.c"
#define ZTT_CORE_MATH_TYPE float
#define ZTT_CORE_MATH_BITS_TYPE uint32_t
#define ZTT_CORE_MATH_FUNCTION cr_cosf
#define ZTT_CORE_MATH_BRIDGE ztt_core_math_cos_f32
#include "ztt_core_math_unary_bridge.h"
