/* Ztt v0.6 comprehensive self-checking smoke test for Spike. */
.option norvc

.macro ZTT_R3 funct7, dest, src1, src2
  .word (((\funct7) & 0x7f) << 25) | (((\src2) & 0x1f) << 20) | \
        (((\src1) & 0x1f) << 15) | (((\dest) & 0x1f) << 7) | 0x2b
.endm
.macro ZTT_R2 funct7, dest, src1, xfunct5
  ZTT_R3 \funct7, \dest, \src1, \xfunct5
.endm
.macro ZTT_R1 funct7, dest, xfunct10
  .word (((\funct7) & 0x7f) << 25) | \
        (((\xfunct10) & 0x3ff) << 15) | (((\dest) & 0x1f) << 7) | 0x2b
.endm
.macro MSETTYP md, xreg
  ZTT_R2 0x51, \md, \xreg, 0x03
.endm
.macro ASETTYP ad, xreg
  ZTT_R2 0x51, \ad, \xreg, 0x01
.endm
.macro MGETTYP xreg, ms
  ZTT_R2 0x51, \xreg, \ms, 0x02
.endm
.macro MLS_RM md, xreg
  ZTT_R2 0x51, \md, \xreg, 0x18
.endm
.macro MSS_RM ms, xreg
  ZTT_R2 0x51, \ms, \xreg, 0x1b
.endm
.macro FAIL code
  /* Spike decodes a test result as (tohost >> 1); bit 0 marks a result. */
  li t6, ((\code) * 2 + 1)
  j fail
.endm

.section .text.init
.globl _start
_start:
  /* Enable AME and acquire the unit. */
  li t0, 0x02000000
  csrs mstatus, t0
  ZTT_R2 0x52, 1, 0, 0x09       /* ame.acquire x1, x0 */
  li t0, 1
  bne x1, t0, 1f
  j 2f
1: FAIL 1
2:

  /* Packed Int8 elementwise add. */
  li t0, 0x40000008
  MSETTYP 0, 5
  MSETTYP 1, 5
  MSETTYP 2, 5
  la s0, int8_ones
  MLS_RM 0, 8
  la s0, int8_twos
  MLS_RM 1, 8
  ZTT_R3 0x02, 2, 0, 1
  la s0, int8_out
  MSS_RM 2, 8
  la a0, int8_out
  la a1, int8_expected
  li a2, 64
  call compare_bytes
  bnez a0, 3f
  j 4f
3: FAIL 2
4:

  /* Int64 operands span two physical M registers. */
  li t0, 0x40000040
  MSETTYP 4, 5
  MSETTYP 6, 5
  MSETTYP 8, 5
  la s0, int64_ones
  MLS_RM 4, 8
  la s0, int64_twos
  MLS_RM 6, 8
  ZTT_R3 0x02, 8, 4, 6
  la s0, int64_out
  MSS_RM 8, 8
  la a0, int64_out
  la a1, int64_expected
  li a2, 128
  call compare_bytes
  bnez a0, 5f
  j 6f
5: FAIL 3
6:

  /* Int128 addition must preserve both 64-bit halves. */
  li t0, 0x40000080
  MSETTYP 0, 5
  MSETTYP 4, 5
  MSETTYP 8, 5
  la s0, int128_a
  MLS_RM 0, 8
  la s0, int128_b
  MLS_RM 4, 8
  ZTT_R3 0x02, 8, 0, 4
  la s0, int128_out
  MSS_RM 8, 8
  la a0, int128_out
  la a1, int128_expected
  li a2, 256
  call compare_bytes
  bnez a0, 7f
  j 8f
7: FAIL 4
8:

  /* mconv expands four packed Int8 squares into Int32 registers. */
  li t0, 0x40000008
  MSETTYP 0, 5
  la s0, int8_ones
  MLS_RM 0, 8
  li t0, 0x40000020
  MSETTYP 12, 5
  MSETTYP 13, 5
  MSETTYP 14, 5
  MSETTYP 15, 5
  ZTT_R2 0x51, 12, 0, 0x0d
  la s0, conv_out
  MSS_RM 12, 8
  addi s0, s0, 64
  MSS_RM 13, 8
  addi s0, s0, 64
  MSS_RM 14, 8
  addi s0, s0, 64
  MSS_RM 15, 8
  la a0, conv_out
  la a1, conv_expected
  li a2, 256
  call compare_bytes
  bnez a0, 9f
  j 10f
9: FAIL 5
10:

  /* Row-strided load followed by contiguous store. */
  li t0, 0x40000020
  MSETTYP 16, 5
  la s0, strided_input
  li s1, 32
  ZTT_R3 0x54, 16, 8, 9
  la s0, strided_out
  MSS_RM 16, 8
  la a0, strided_out
  la a1, matrix_1_to_16
  li a2, 64
  call compare_bytes
  bnez a0, 11f
  j 12f
11: FAIL 6
12:

  /* Identity multiply checks accumulator zero, 2-D MAC and moves. */
  li t0, 0x40000020
  MSETTYP 16, 5
  MSETTYP 17, 5
  MSETTYP 24, 5
  ASETTYP 0, 5
  la s0, identity_matrix
  MLS_RM 16, 8
  la s0, matrix_1_to_16
  MLS_RM 17, 8
  ZTT_R1 0x53, 0, 0x000
  ZTT_R3 0x4a, 0, 16, 17
  ZTT_R2 0x51, 24, 0, 0x1c
  la s0, multiply_out
  MSS_RM 24, 8
  la a0, multiply_out
  la a1, matrix_1_to_16
  li a2, 64
  call compare_bytes
  bnez a0, 13f
  j 14f
13: FAIL 7
14:

  /* Saturating Int8 add and SAT flag. */
  li t0, 0x40000008
  MSETTYP 20, 5
  MSETTYP 21, 5
  li t0, 0x60000008
  MSETTYP 20, 5
  MSETTYP 21, 5
  MSETTYP 22, 5
  la s0, sat_input
  MLS_RM 20, 8
  MLS_RM 21, 8
  csrw 0xcc5, zero
  ZTT_R3 0x02, 22, 20, 21
  la s0, sat_out
  MSS_RM 22, 8
  la a0, sat_out
  la a1, sat_expected
  li a2, 64
  call compare_bytes
  bnez a0, 15f
  csrr t0, 0xcc5
  li t1, 1
  bne t0, t1, 15f
  j 16f
15: FAIL 8
16:

  /* FP32 reciprocal of zero: infinity and divide-by-zero flag. */
  li t0, 0x20300120
  MSETTYP 26, 5
  MSETTYP 27, 5
  la s0, fp_zero
  MLS_RM 26, 8
  csrw 0xcc4, zero
  ZTT_R2 0x51, 27, 26, 0x11
  la s0, fp_out
  MSS_RM 27, 8
  la a0, fp_out
  la a1, fp_expected
  li a2, 64
  call compare_bytes
  bnez a0, 17f
  csrr t0, 0xcc4
  li t1, 8
  bne t0, t1, 17f
  j 18f
17: FAIL 9
18:

  /* FP32 max/min NaN and signed-zero behavior. */
  li t0, 0x20300120
  MSETTYP 24, 5
  MSETTYP 25, 5
  MSETTYP 26, 5
  MSETTYP 27, 5
  la s0, fp_special_a
  MLS_RM 24, 8
  la s0, fp_special_b
  MLS_RM 25, 8
  ZTT_R3 0x06, 26, 24, 25
  la s0, fp_max_out
  MSS_RM 26, 8
  la a0, fp_max_out
  la a1, fp_max_expected
  li a2, 64
  call compare_bytes
  bnez a0, 19f
  ZTT_R3 0x0a, 27, 24, 25
  la s0, fp_min_out
  MSS_RM 27, 8
  la a0, fp_min_out
  la a1, fp_min_expected
  li a2, 64
  call compare_bytes
  bnez a0, 19f
  j 20f
19: FAIL 10
20:

  /* 64-bit debug move through a 32-bit unit datatype. */
  li t0, 0x40000020
  MSETTYP 29, 5
  li t0, 0x0123456789abcdef
  li t1, 7
  ZTT_R3 0x5c, 29, 5, 6
  ZTT_R3 0x60, 7, 29, 6
  bne t0, t2, 21f
  j 22f
21: FAIL 11
22:

  /* FP32 transcendental operations use the CORE-MATH-backed instruction path. */
  li t0, 0x20300120
  MSETTYP 20, 5
  MSETTYP 21, 5
  la s0, fp_ones
  MLS_RM 20, 8
  csrw 0xcc4, zero
  ZTT_R2 0x51, 21, 20, 0x0e    /* mcos.ew */
  la s0, fp_trans_out
  MSS_RM 21, 8
  la a0, fp_trans_out
  la a1, fp_cos_one_expected
  li a2, 64
  call compare_bytes
  bnez a0, 23f
  ZTT_R2 0x51, 21, 20, 0x13    /* msin.ew */
  la s0, fp_trans_out
  MSS_RM 21, 8
  la a0, fp_trans_out
  la a1, fp_sin_one_expected
  li a2, 64
  call compare_bytes
  bnez a0, 23f
  ZTT_R2 0x51, 21, 20, 0x15    /* mtanh.ew */
  la s0, fp_trans_out
  MSS_RM 21, 8
  la a0, fp_trans_out
  la a1, fp_tanh_one_expected
  li a2, 64
  call compare_bytes
  bnez a0, 23f
  ZTT_R2 0x51, 21, 20, 0x10    /* mlog2.ew */
  la s0, fp_trans_out
  MSS_RM 21, 8
  la a0, fp_trans_out
  la a1, fp_log2_one_expected
  li a2, 64
  call compare_bytes
  bnez a0, 23f
  ZTT_R2 0x51, 21, 20, 0x0f    /* mexp2.ew */
  la s0, fp_trans_out
  MSS_RM 21, 8
  la a0, fp_trans_out
  la a1, fp_exp2_one_expected
  li a2, 64
  call compare_bytes
  bnez a0, 23f
  csrr t0, 0xcc4
  li t1, 1
  bne t0, t1, 23f
  j 24f
23: FAIL 12
24:

  /* Datatype masking, invalid MSETTYP atomicity, unsupported probe, and release. */
  li t0, 0x10040000020
  MSETTYP 28, 5
  MGETTYP 6, 28
  li t0, 0x40000020
  bne t1, t0, dtype_mask_fail

  /* A rejected MSETTYP must preserve both Md and the M register group. */
  la s0, matrix_1_to_16
  MLS_RM 28, 8
  csrw 0x800, zero
  li t0, 0
  MSETTYP 28, 5
  csrr t0, 0x800
  li t1, 1
  bne t0, t1, dtype_zero_status_fail
  MGETTYP 6, 28
  li t0, 0x40000020
  bne t1, t0, dtype_zero_md_fail
  la s0, dtype_zero_out
  MSS_RM 28, 8
  la a0, dtype_zero_out
  la a1, matrix_1_to_16
  li a2, 64
  call compare_bytes
  bnez a0, dtype_zero_data_fail

  /*
   * M30 has never received a datatype.  After clearing the previous UN,
   * using that uninitialized datatype must set UN independently.
   */
  csrw 0x800, zero
  ZTT_R3 0x02, 30, 30, 30
  csrr t0, 0x800
  li t1, 1
  bne t0, t1, dtype_un_status_fail
  csrw 0x800, zero
  csrr t0, 0x800
  bnez t0, dtype_clear_status_fail
  csrr t0, mstatus
  srli t0, t0, 25
  andi t0, t0, 3
  li t1, 3
  bne t0, t1, dtype_mstatus_fail
  ZTT_R2 0x52, 0, 0, 0x0a
  csrr t0, 0xcc3
  bnez t0, dtype_release_status_fail
  li t6, 1
  j finish
dtype_mask_fail: FAIL 13
dtype_zero_status_fail: FAIL 14
dtype_zero_md_fail: FAIL 15
dtype_zero_data_fail: FAIL 16
dtype_un_status_fail: FAIL 17
dtype_clear_status_fail: FAIL 18
dtype_mstatus_fail: FAIL 19
dtype_release_status_fail: FAIL 20

/* a0=lhs, a1=rhs, a2=size; return a0=0 when equal. */
compare_bytes:
  beqz a2, 2f
1:
  lbu t0, 0(a0)
  lbu t1, 0(a1)
  bne t0, t1, 3f
  addi a0, a0, 1
  addi a1, a1, 1
  addi a2, a2, -1
  bnez a2, 1b
2:
  li a0, 0
  ret
3:
  li a0, 1
  ret

fail:
  la t0, tohost
  sd t6, 0(t0)
  fence
1: j 1b

finish:
  la t0, tohost
  sd t6, 0(t0)
  fence
2: j 2b

.section .rodata
.balign 16
int8_ones:
  .rept 64
  .byte 1
  .endr
int8_twos:
  .rept 64
  .byte 2
  .endr
int8_expected:
  .rept 64
  .byte 3
  .endr
sat_input:
  .rept 64
  .byte 120
  .endr
sat_expected:
  .rept 64
  .byte 127
  .endr
int64_ones:
  .rept 16
  .dword 1
  .endr
int64_twos:
  .rept 16
  .dword 2
  .endr
int64_expected:
  .rept 16
  .dword 3
  .endr
int128_a:
  .rept 16
  .dword 0xffffffffffffffff, 0x7fffffffffffffff
  .endr
int128_b:
  .rept 16
  .dword 2, 0
  .endr
int128_expected:
  .rept 16
  .dword 1, 0x8000000000000000
  .endr
conv_expected:
  .rept 64
  .word 1
  .endr
strided_input:
  .word 1,2,3,4,0,0,0,0
  .word 5,6,7,8,0,0,0,0
  .word 9,10,11,12,0,0,0,0
  .word 13,14,15,16,0,0,0,0
matrix_1_to_16:
  .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
identity_matrix:
  .word 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
fp_zero:
  .zero 64
fp_expected:
  .rept 16
  .word 0x7f800000
  .endr
fp_special_a:
  .word 0x7fc12345,0x80000000
  .rept 14
  .word 0x3f800000
  .endr
fp_special_b:
  .word 0x3f800000,0
  .rept 14
  .word 0x40000000
  .endr
fp_max_expected:
  .word 0x7fc00000,0
  .rept 14
  .word 0x40000000
  .endr
fp_min_expected:
  .word 0x7fc00000,0x80000000
  .rept 14
  .word 0x3f800000
  .endr
fp_ones:
  .rept 16
  .word 0x3f800000
  .endr
fp_cos_one_expected:
  .rept 16
  .word 0x3f0a5140
  .endr
fp_sin_one_expected:
  .rept 16
  .word 0x3f576aa4
  .endr
fp_tanh_one_expected:
  .rept 16
  .word 0x3f42f7d6
  .endr
fp_log2_one_expected:
  .zero 64
fp_exp2_one_expected:
  .rept 16
  .word 0x40000000
  .endr

.section .bss
.balign 64
int8_out: .zero 64
int64_out: .zero 128
int128_out: .zero 256
conv_out: .zero 256
strided_out: .zero 64
multiply_out: .zero 64
sat_out: .zero 64
fp_out: .zero 64
fp_max_out: .zero 64
fp_min_out: .zero 64
fp_trans_out: .zero 64
dtype_zero_out: .zero 64

.section .tohost,"aw",@progbits
.balign 8
.globl tohost
tohost: .dword 0
.globl fromhost
fromhost: .dword 0
