// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.

#include "ztt_state.h"
#include <cassert>
#include <vector>

constexpr uint32_t fp_dtype(unsigned exponent, unsigned rounding,
                            bool infinity, unsigned bits)
{
  return (exponent << 26) | (rounding << 22) |
         (uint32_t(infinity) << 21) | (1u << 20) | (1u << 8) | bits;
}

int main()
{
  ztt_unit_t unit;
  const uint32_t int32 = 32;
  const uint32_t int8 = 8;
  const uint32_t int128 = 128;

  assert(ztt_unit_t::datatype_supported(int32));
  for (uint32_t dtype : {
         fp_dtype(4, 0, false, 8), fp_dtype(5, 0, true, 8),
         fp_dtype(5, 0, true, 16), fp_dtype(8, 0, true, 16),
         fp_dtype(8, 0, true, 32), fp_dtype(11, 0, true, 64)})
    assert(ztt_unit_t::datatype_supported(dtype));
  assert(ztt_unit_t::datatype_float_kind(fp_dtype(4, 0, false, 8)) ==
         ztt_float_kind_t::e4m3);
  assert(ztt_unit_t::datatype_float_kind(fp_dtype(8, 0, true, 16)) ==
         ztt_float_kind_t::bf16);
  assert(!ztt_unit_t::datatype_saturating(fp_dtype(8, 0, true, 32)));
  assert(!ztt_unit_t::datatype_supported(fp_dtype(8, 6, true, 32)));
  assert(!ztt_unit_t::datatype_supported(fp_dtype(8, 0, false, 32)));
  assert(!ztt_unit_t::datatype_supported(0));
  assert(ztt_unit_t::group_registers(int128) == 4);
  assert(unit.set_m_datatype(0, int32));
  assert(unit.m_datatype(0) == int32);
  assert(unit.broadcast_x(0, 0x12345678));
  assert(unit.m_register(0)[0] == 0x1234567812345678ULL);
  assert(unit.m_register(0)[7] == 0x1234567812345678ULL);

  unit.write_amestatus(0);
  assert(unit.set_m_datatype(1, 0));
  assert(unit.m_datatype(1) == 0);
  assert(unit.amestatus() == 0);

  unit.write_amestatus(0);
  assert(unit.set_m_datatype(4, int128));
  assert(unit.set_m_datatype(8, int128));
  assert(unit.move_m_to_m(8, 4));
  assert(unit.set_m_datatype(16, int8));
  assert(unit.broadcast_x(16, 0xab));
  assert(unit.m_register(16)[0] == 0xababababababababULL);

  std::vector<ztt_unit_t::element_t> packed(64);
  for (std::size_t i = 0; i < packed.size(); ++i)
    packed[i] = i;
  assert(unit.write_m(16, packed));
  std::vector<ztt_unit_t::element_t> packed_result;
  assert(unit.read_m(16, packed_result));
  assert(packed_result == packed);

  std::vector<ztt_unit_t::element_t> wide(16);
  for (std::size_t i = 0; i < wide.size(); ++i)
    wide[i] = (ztt_unit_t::element_t(i) << 100) | i;
  assert(unit.write_m(4, wide));
  std::vector<ztt_unit_t::element_t> wide_result;
  assert(unit.read_m(4, wide_result));
  assert(wide_result == wide);

  std::vector<ztt_unit_t::element_t> atomic_a(16, 0x11);
  std::vector<ztt_unit_t::element_t> atomic_b(16, 0x22);
  const auto before_atomic = unit.m_register(0);
  assert(!unit.write_m_atomic({
    {0, int32, atomic_a},
    {31, int128, atomic_b}, // invalid span: m31 cannot hold four registers
  }));
  assert(unit.m_register(0) == before_atomic);

  std::vector<ztt_unit_t::element_t> acc_a(16, 0x33);
  std::vector<ztt_unit_t::element_t> acc_b(16, 0x44);
  const auto before_acc = unit.accumulator(0);
  assert(!unit.write_acc_atomic({
    {0, int32, acc_a},
    {ztt::kNumAccRegisters, int32, acc_b},
  }));
  assert(unit.accumulator(0) == before_acc);

  assert(unit.set_acc_datatype(0, int32));
  assert(unit.zero_acc(0));
  assert(unit.move_acc_to_m(0, 0));

  constexpr uint32_t custom255 = 0x800000ffu;
  ztt_unit_t::m_register_t raw_m;
  raw_m.fill(~uint64_t(0));
  for (unsigned reg = 24; reg < 32; ++reg)
    assert(unit.write_m_register(reg, raw_m));
  unit.write_amestatus(0);
  assert(unit.set_m_datatype(24, custom255));
  assert(unit.m_datatype(24) == custom255);
  for (unsigned reg = 24; reg < 32; ++reg)
    for (uint64_t word : unit.m_register(reg))
      assert(word == 0);
  assert(unit.amestatus() == 0);
  assert(!unit.set_m_datatype(31, custom255));

  assert(unit.set_acc_datatype(0, int32));
  assert(unit.write_acc(0, acc_a));
  assert(unit.set_acc_datatype(0, custom255));
  assert(unit.acc_datatype(0) == custom255);
  for (uint64_t word : unit.accumulator(0))
    assert(word == 0);
  assert(unit.amestatus() == 0);
  return 0;
}
