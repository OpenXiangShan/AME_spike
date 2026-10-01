// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
#define ZTT_CORE_MATH_SOURCE \
  "../core-math/src/binaryb16/sin/sin_bf16.c"
#define ZTT_CORE_MATH_TYPE __bf16
#define ZTT_CORE_MATH_BITS_TYPE uint16_t
#define ZTT_CORE_MATH_FUNCTION cr_sin_bf16
#define ZTT_CORE_MATH_BRIDGE ztt_core_math_sin_bf16
#include "ztt_core_math_unary_bridge.h"
