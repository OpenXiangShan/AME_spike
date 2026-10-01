// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#ifndef _RISCV_ZTT_STATE_H
#define _RISCV_ZTT_STATE_H

#include "ztt_config.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class ztt_float_kind_t {
  none,
  e4m3,
  e5m2,
  f16,
  bf16,
  f32,
  f64,
};

class ztt_unit_t {
 public:
  using element_t = unsigned __int128;
  using m_register_t = std::array<uint64_t, ztt::kMRegisterBits / 64>;
  using accumulator_t = std::array<uint64_t, ztt::kAccumulatorBits / 64>;
  struct m_write_t {
    std::size_t reg;
    uint32_t dtype;
    std::vector<element_t> elements;
  };
  struct acc_write_t {
    std::size_t reg;
    uint32_t dtype;
    std::vector<element_t> elements;
  };

  void reset();

  uint64_t amestatus() const { return status; }
  void write_amestatus(uint64_t value) { status &= value & ztt::kAmestatusUn; }
  void set_unsupported() { status |= ztt::kAmestatusUn; }

  uint32_t amestype() const { return scalar_type; }
  void write_amestype(uint64_t value) { scalar_type = uint32_t(value); }
  uint64_t amefflags() const { return fp_flags; }
  void write_amefflags(uint64_t value) { fp_flags = value & ztt::kAmeFlagMask; }
  void set_amefflags(uint8_t value) { fp_flags |= value & ztt::kAmeFlagMask; }
  uint64_t amexsat() const { return sat_flag ? 1 : 0; }
  void write_amexsat(uint64_t value) { sat_flag = value & ztt::kAmeSatMask; }
  void set_amexsat() { sat_flag = true; }
  bool owned() const { return backend_owned; }
  void acquire() { backend_owned = true; }
  void release() { backend_owned = false; }

  static bool datatype_supported(uint32_t dtype);
  static std::size_t datatype_bits(uint32_t dtype) { return dtype & 0xff; }
  static std::size_t group_registers(uint32_t dtype);

  bool valid_m_group_geometry(std::size_t reg, uint32_t dtype) const;
  bool valid_m_group(std::size_t reg, uint32_t dtype) const;
  bool set_m_datatype(std::size_t reg, uint32_t dtype);
  bool set_acc_datatype(std::size_t reg, uint32_t dtype);
  uint32_t m_datatype(std::size_t reg) const { return md.at(reg); }
  uint32_t acc_datatype(std::size_t reg) const { return ad.at(reg); }

  bool broadcast_x(std::size_t dest, uint64_t scalar);
  bool move_m_to_m(std::size_t dest, std::size_t source);
  bool move_acc_to_m(std::size_t dest, std::size_t source);
  bool zero_acc(std::size_t dest);

  bool read_m(std::size_t reg, std::vector<element_t>& elements) const;
  bool write_m(std::size_t reg, const std::vector<element_t>& elements);
  bool read_m_as(std::size_t reg, uint32_t dtype,
                 std::vector<element_t>& elements) const;
  bool write_m_as(std::size_t reg, uint32_t dtype,
                  const std::vector<element_t>& elements);
  bool write_m_atomic(const std::vector<m_write_t>& writes);
  bool read_acc(std::size_t reg, std::vector<element_t>& elements) const;
  bool write_acc(std::size_t reg, const std::vector<element_t>& elements);
  bool read_acc_as(std::size_t reg, uint32_t dtype,
                   std::vector<element_t>& elements) const;
  bool write_acc_as(std::size_t reg, uint32_t dtype,
                    const std::vector<element_t>& elements);
  bool write_acc_atomic(const std::vector<acc_write_t>& writes);

  static std::size_t square_count(uint32_t dtype);
  static std::size_t element_count(uint32_t dtype);
  static element_t element_mask(std::size_t width);
  static bool datatype_floating(uint32_t dtype)
    { return datatype_float_kind(dtype) != ztt_float_kind_t::none; }
  static bool datatype_integer(uint32_t dtype)
    { return datatype_supported(dtype) && !datatype_floating(dtype); }
  static ztt_float_kind_t datatype_float_kind(uint32_t dtype);
  static unsigned datatype_rounding_mode(uint32_t dtype)
    { return (dtype & (1u << 8)) == 0 ? ((dtype >> 27) & 0x3) : ((dtype >> 22) & 0xf); }
  static bool datatype_signed(uint32_t dtype)
    { return datatype_integer(dtype) && ((dtype >> 30) & 1); }
  static bool datatype_saturating(uint32_t dtype)
    { return datatype_integer(dtype) && ((dtype >> 29) & 1); }

  const m_register_t& m_register(std::size_t reg) const { return m.at(reg); }
  bool write_m_register(std::size_t reg, const m_register_t& value);
  const accumulator_t& accumulator(std::size_t reg) const { return acc.at(reg); }

 private:
  void clear_m_group(std::size_t reg, std::size_t nregs);
  void write_m_bits(std::size_t base, std::size_t bit_offset,
                    std::size_t width, element_t value);
  element_t read_m_bits(std::size_t base, std::size_t bit_offset,
                        std::size_t width) const;
  static element_t read_bits(const uint64_t* words, std::size_t bit_offset,
                             std::size_t width);
  static void write_bits(uint64_t* words, std::size_t bit_offset,
                         std::size_t width, element_t value);

  uint64_t status = 0;
  uint32_t scalar_type = 0;
  uint8_t fp_flags = 0;
  bool sat_flag = false;
  bool backend_owned = false;
  std::array<m_register_t, ztt::kNumMRegisters> m{};
  std::array<uint32_t, ztt::kNumMRegisters> md{};
  std::array<accumulator_t, ztt::kNumAccRegisters> acc{};
  std::array<uint32_t, ztt::kNumAccRegisters> ad{};
};

#endif
