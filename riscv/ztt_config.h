// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_CONFIG_H
#define _RISCV_ZTT_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace ztt {

constexpr std::size_t kNumElements = 16;
constexpr std::size_t kTileLength = 4;
constexpr std::size_t kNumMRegisters = 32;
constexpr std::size_t kNumAccRegisters = 4;
constexpr std::size_t kUnitDatatypeBits = 32;
constexpr std::size_t kMRegisterBits = kNumElements * kUnitDatatypeBits;
constexpr std::size_t kMaxDatatypeBits = 128;
constexpr std::size_t kAccumulatorBits = kNumElements * kMaxDatatypeBits;
// Greatest supported integer width; signed 128-bit is the fixed exponent type.
constexpr uint32_t kMaxIntDatatype = (1u << 30) | 128u;

// Ztt v0.6 provisional CSR allocation. CC0-CC5 are the user custom
// capability/context slots used by the draft; amestatus uses a custom slot.
constexpr uint32_t kCsrAmenlen = 0xcc0;
constexpr uint32_t kCsrAmeudsz = 0xcc1;
constexpr uint32_t kCsrAmestype = 0xcc2;
constexpr uint32_t kCsrAmeown = 0xcc3;
constexpr uint32_t kCsrAmefflags = 0xcc4;
constexpr uint32_t kCsrAmexsat = 0xcc5;
constexpr uint32_t kCsrAmestatus = 0x800;
constexpr uint64_t kAmestatusUn = 1;
constexpr uint64_t kAmeFlagMask = 0x1f;
constexpr uint64_t kAmeSatMask = 0x1;

} // namespace ztt

#endif
