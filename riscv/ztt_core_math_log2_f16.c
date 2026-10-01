// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
#define ZTT_CORE_MATH_SOURCE \
  "../core-math/src/binary16/log2/log2f16.c"
#define ZTT_CORE_MATH_TYPE _Float16
#define ZTT_CORE_MATH_BITS_TYPE uint16_t
#define ZTT_CORE_MATH_FUNCTION cr_log2f16
#define ZTT_CORE_MATH_BRIDGE ztt_core_math_log2_f16
#include "ztt_core_math_unary_bridge.h"
