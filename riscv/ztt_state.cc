// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_state.h"
#include <algorithm>

void ztt_unit_t::reset()
{
  status = 0;
  scalar_type = 0;
  fp_flags = 0;
  sat_flag = false;
  backend_owned = false;
  m = {};
  md = {};
  acc = {};
  ad = {};
}

bool ztt_unit_t::datatype_supported(uint32_t dtype)
{
  if (dtype == 0 || (dtype >> 31) != 0)
    return false;

  if ((dtype & (1u << 8)) != 0)
    return datatype_float_kind(dtype) != ztt_float_kind_t::none;
  // Integer rounding occupies bits 28:27 in v0.6.  Bits 26:9 remain
  // reserved and must be zero; sign (30) and saturation (29) are legal.
  if ((dtype & 0x07fffe00u) != 0)
    return false;

  switch (datatype_bits(dtype)) {
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      return true;
    default:
      return false;
  }
}

ztt_float_kind_t ztt_unit_t::datatype_float_kind(uint32_t dtype)
{
  if (dtype == 0 || (dtype >> 31) != 0 || (dtype & (1u << 8)) == 0 ||
      (dtype & 0x000ffe00u) != 0 || datatype_rounding_mode(dtype) > 5)
    return ztt_float_kind_t::none;

  const unsigned exponent = (dtype >> 26) & 0x1f;
  const bool infinity = (dtype >> 21) & 1;
  const bool denormal = (dtype >> 20) & 1;
  const unsigned bits = datatype_bits(dtype);
  if (!denormal)
    return ztt_float_kind_t::none;
  if (bits == 8 && exponent == 4 && !infinity)
    return ztt_float_kind_t::e4m3;
  if (bits == 8 && exponent == 5 && infinity)
    return ztt_float_kind_t::e5m2;
  if (bits == 16 && exponent == 5 && infinity)
    return ztt_float_kind_t::f16;
  if (bits == 16 && exponent == 8 && infinity)
    return ztt_float_kind_t::bf16;
  if (bits == 32 && exponent == 8 && infinity)
    return ztt_float_kind_t::f32;
  if (bits == 64 && exponent == 11 && infinity)
    return ztt_float_kind_t::f64;
  return ztt_float_kind_t::none;
}

std::size_t ztt_unit_t::group_registers(uint32_t dtype)
{
  const std::size_t bits = datatype_bits(dtype);
  return std::max<std::size_t>(1, (bits + ztt::kUnitDatatypeBits - 1) /
                                  ztt::kUnitDatatypeBits);
}

bool ztt_unit_t::valid_m_group(std::size_t reg, uint32_t dtype) const
{
  return datatype_supported(dtype) && valid_m_group_geometry(reg, dtype);
}

bool ztt_unit_t::valid_m_group_geometry(std::size_t reg,
                                        uint32_t dtype) const
{
  if (reg >= ztt::kNumMRegisters)
    return false;
  const std::size_t nregs = group_registers(dtype);
  return reg % nregs == 0 && reg + nregs <= ztt::kNumMRegisters;
}

void ztt_unit_t::clear_m_group(std::size_t reg, std::size_t nregs)
{
  for (std::size_t i = 0; i < nregs; ++i)
    m[reg + i].fill(0);
}

bool ztt_unit_t::set_m_datatype(std::size_t reg, uint32_t dtype)
{
  if (!valid_m_group_geometry(reg, dtype))
    return false;
  md[reg] = dtype;
  clear_m_group(reg, group_registers(dtype));
  return true;
}

bool ztt_unit_t::set_acc_datatype(std::size_t reg, uint32_t dtype)
{
  if (reg >= ztt::kNumAccRegisters)
    return false;
  ad[reg] = dtype;
  acc[reg].fill(0);
  return true;
}

ztt_unit_t::element_t ztt_unit_t::element_mask(std::size_t width)
{
  return width == 128 ? ~element_t(0) : (element_t(1) << width) - 1;
}

std::size_t ztt_unit_t::square_count(uint32_t dtype)
{
  const std::size_t width = datatype_bits(dtype);
  return width < ztt::kUnitDatatypeBits ? ztt::kUnitDatatypeBits / width : 1;
}

std::size_t ztt_unit_t::element_count(uint32_t dtype)
{
  return ztt::kNumElements * square_count(dtype);
}

ztt_unit_t::element_t ztt_unit_t::read_bits(const uint64_t* words,
                                             std::size_t bit_offset,
                                             std::size_t width)
{
  element_t value = 0;
  for (std::size_t bit = 0; bit < width; ++bit) {
    const std::size_t absolute = bit_offset + bit;
    value |= element_t((words[absolute / 64] >> (absolute % 64)) & 1) << bit;
  }
  return value;
}

void ztt_unit_t::write_bits(uint64_t* words, std::size_t bit_offset,
                            std::size_t width, element_t value)
{
  for (std::size_t bit = 0; bit < width; ++bit) {
    const std::size_t absolute = bit_offset + bit;
    const uint64_t mask = uint64_t(1) << (absolute % 64);
    uint64_t& word = words[absolute / 64];
    word = (word & ~mask) | (uint64_t((value >> bit) & 1) * mask);
  }
}

ztt_unit_t::element_t ztt_unit_t::read_m_bits(std::size_t base,
                                               std::size_t bit_offset,
                                               std::size_t width) const
{
  element_t value = 0;
  for (std::size_t bit = 0; bit < width; ++bit) {
    const std::size_t absolute = bit_offset + bit;
    const std::size_t reg = base + absolute / ztt::kMRegisterBits;
    const std::size_t in_reg = absolute % ztt::kMRegisterBits;
    value |= element_t((m[reg][in_reg / 64] >> (in_reg % 64)) & 1) << bit;
  }
  return value;
}

void ztt_unit_t::write_m_bits(std::size_t base, std::size_t bit_offset,
                              std::size_t width, element_t value)
{
  for (std::size_t bit = 0; bit < width; ++bit) {
    const std::size_t absolute = bit_offset + bit;
    const std::size_t reg = base + absolute / ztt::kMRegisterBits;
    const std::size_t in_reg = absolute % ztt::kMRegisterBits;
    const uint64_t mask = uint64_t(1) << (in_reg % 64);
    uint64_t& word = m[reg][in_reg / 64];
    word = (word & ~mask) | (uint64_t((value >> bit) & 1) * mask);
  }
}

bool ztt_unit_t::read_m(std::size_t reg,
                        std::vector<element_t>& elements) const
{
  if (reg >= ztt::kNumMRegisters)
    return false;
  return read_m_as(reg, md[reg], elements);
}

bool ztt_unit_t::read_m_as(std::size_t reg, uint32_t dtype,
                           std::vector<element_t>& elements) const
{
  if (!valid_m_group(reg, dtype))
    return false;
  const std::size_t width = datatype_bits(dtype);
  elements.resize(element_count(dtype));
  for (std::size_t i = 0; i < elements.size(); ++i)
    elements[i] = read_m_bits(reg, i * width, width);
  return true;
}

bool ztt_unit_t::write_m(std::size_t reg,
                         const std::vector<element_t>& elements)
{
  if (reg >= ztt::kNumMRegisters)
    return false;
  return write_m_as(reg, md[reg], elements);
}

bool ztt_unit_t::write_m_as(std::size_t reg, uint32_t dtype,
                            const std::vector<element_t>& elements)
{
  if (!valid_m_group(reg, dtype) || elements.size() != element_count(dtype))
    return false;
  const std::size_t width = datatype_bits(dtype);
  for (std::size_t i = 0; i < elements.size(); ++i)
    write_m_bits(reg, i * width, width, elements[i] & element_mask(width));
  return true;
}

bool ztt_unit_t::write_m_atomic(const std::vector<m_write_t>& writes)
{
  for (const auto& write : writes)
    if (!valid_m_group(write.reg, write.dtype) ||
        write.elements.size() != element_count(write.dtype))
      return false;

  const auto saved = m;
  for (const auto& write : writes) {
    if (!write_m_as(write.reg, write.dtype, write.elements)) {
      m = saved;
      return false;
    }
  }
  return true;
}

bool ztt_unit_t::write_m_register(std::size_t reg, const m_register_t& value)
{
  if (reg >= ztt::kNumMRegisters)
    return false;
  m[reg] = value;
  return true;
}

bool ztt_unit_t::read_acc(std::size_t reg,
                          std::vector<element_t>& elements) const
{
  if (reg >= ztt::kNumAccRegisters || !datatype_supported(ad[reg]))
    return false;
  return read_acc_as(reg, ad[reg], elements);
}

bool ztt_unit_t::read_acc_as(std::size_t reg, uint32_t dtype,
                             std::vector<element_t>& elements) const
{
  if (reg >= ztt::kNumAccRegisters || !datatype_supported(dtype))
    return false;
  const std::size_t width = datatype_bits(dtype);
  elements.resize(ztt::kNumElements);
  for (std::size_t i = 0; i < elements.size(); ++i)
    elements[i] = read_bits(acc[reg].data(), i * width, width);
  return true;
}

bool ztt_unit_t::write_acc(std::size_t reg,
                           const std::vector<element_t>& elements)
{
  if (reg >= ztt::kNumAccRegisters || !datatype_supported(ad[reg]) ||
      elements.size() != ztt::kNumElements)
    return false;
  return write_acc_as(reg, ad[reg], elements);
}

bool ztt_unit_t::write_acc_as(std::size_t reg, uint32_t dtype,
                              const std::vector<element_t>& elements)
{
  if (reg >= ztt::kNumAccRegisters || !datatype_supported(dtype) ||
      elements.size() != ztt::kNumElements)
    return false;
  accumulator_t tmp = acc[reg];
  const std::size_t width = datatype_bits(dtype);
  for (std::size_t i = 0; i < elements.size(); ++i)
    write_bits(tmp.data(), i * width, width, elements[i] & element_mask(width));
  acc[reg] = tmp;
  return true;
}

bool ztt_unit_t::write_acc_atomic(const std::vector<acc_write_t>& writes)
{
  for (const auto& write : writes)
    if (write.reg >= ztt::kNumAccRegisters ||
        !datatype_supported(write.dtype) ||
        write.elements.size() != ztt::kNumElements)
      return false;

  const auto saved = acc;
  for (const auto& write : writes) {
    if (!write_acc_as(write.reg, write.dtype, write.elements)) {
      acc = saved;
      return false;
    }
  }
  return true;
}

bool ztt_unit_t::broadcast_x(std::size_t dest, uint64_t scalar)
{
  if (dest >= ztt::kNumMRegisters)
    return false;
  const uint32_t dtype = md[dest];
  if (!datatype_supported(dtype)) {
    set_unsupported();
    return false;
  }
  if (!valid_m_group(dest, dtype))
    return false;

  const std::size_t width = datatype_bits(dtype);
  const std::size_t pack = width < ztt::kUnitDatatypeBits
                         ? ztt::kUnitDatatypeBits / width : 1;
  const std::size_t elements = ztt::kNumElements * pack;
  clear_m_group(dest, group_registers(dtype));
  for (std::size_t e = 0; e < elements; ++e)
    write_m_bits(dest, e * width, width, scalar);
  return true;
}

bool ztt_unit_t::move_m_to_m(std::size_t dest, std::size_t source)
{
  if (dest >= ztt::kNumMRegisters || source >= ztt::kNumMRegisters)
    return false;
  const uint32_t dtype = md[dest];
  if (dtype == 0 || dtype != md[source]) {
    set_unsupported();
    return false;
  }
  if (!valid_m_group(dest, dtype) || !valid_m_group(source, dtype))
    return false;

  const std::size_t nregs = group_registers(dtype);
  std::array<m_register_t, ztt::kMaxDatatypeBits / ztt::kUnitDatatypeBits> tmp{};
  for (std::size_t i = 0; i < nregs; ++i)
    tmp[i] = m[source + i];
  for (std::size_t i = 0; i < nregs; ++i)
    m[dest + i] = tmp[i];
  return true;
}

bool ztt_unit_t::move_acc_to_m(std::size_t dest, std::size_t source)
{
  if (dest >= ztt::kNumMRegisters || source >= ztt::kNumAccRegisters)
    return false;
  const uint32_t dtype = md[dest];
  if (dtype == 0 || dtype != ad[source]) {
    set_unsupported();
    return false;
  }
  if (!valid_m_group(dest, dtype))
    return false;

  const std::size_t nregs = group_registers(dtype);
  clear_m_group(dest, nregs);
  const std::size_t words_per_m = ztt::kMRegisterBits / 64;
  for (std::size_t r = 0; r < nregs; ++r)
    for (std::size_t w = 0; w < words_per_m; ++w)
      m[dest + r][w] = acc[source][r * words_per_m + w];
  return true;
}

bool ztt_unit_t::zero_acc(std::size_t dest)
{
  if (dest >= ztt::kNumAccRegisters)
    return false;
  if (!datatype_supported(ad[dest])) {
    set_unsupported();
    return false;
  }
  acc[dest].fill(0);
  return true;
}
