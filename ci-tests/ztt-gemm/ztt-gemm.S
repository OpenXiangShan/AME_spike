/* Ztt v0.6 GEMM operator test for AMESpike. */
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
.macro MLS_RM md, xreg
  ZTT_R2 0x51, \md, \xreg, 0x18
.endm
.macro MSS_RM ms, xreg
  ZTT_R2 0x51, \ms, \xreg, 0x1b
.endm
.macro MMOV_A_M acc, ms
  ZTT_R2 0x51, \acc, \ms, 0x1d
.endm
.macro MMOV_M_A md, acc
  ZTT_R2 0x51, \md, \acc, 0x1c
.endm

.section .text.init
.globl _start
_start:
  li      t0, 0x02000000
  csrs    mstatus, t0

  /* ame.acquire a0, x0 (dedicated backend, non-waiting). */
  ZTT_R2  0x52, 10, 0, 0x09
  li      t1, 1
  bne     a0, t1, fail

  li      t0, 0x40000020              /* Int32 */
  MSETTYP 0, 5
  MSETTYP 1, 5
  MSETTYP 2, 5
  ZTT_R2  0x51, 0, 5, 0x01           /* asettyp acc0, t0 */

  /* C + A x B. */
  la      s0, matrix_a
  MLS_RM  0, 8
  la      s0, matrix_b
  MLS_RM  1, 8
  la      s0, matrix_c
  MLS_RM  2, 8
  MMOV_A_M 0, 2
  ZTT_R3  0x4a, 0, 0, 1              /* mmulacc.2d acc0, m0, m1 */
  MMOV_M_A 2, 0
  la      s0, result
  MSS_RM  2, 8
  la      a0, result
  la      a1, expected_add
  call    compare_words
  bnez    a0, fail

  /* C - A x B. */
  la      s0, matrix_c
  MLS_RM  2, 8
  MMOV_A_M 0, 2
  ZTT_R3  0x4b, 0, 0, 1              /* mmulaccneg.2d */
  MMOV_M_A 2, 0
  la      s0, result
  MSS_RM  2, 8
  la      a0, result
  la      a1, expected_sub
  call    compare_words
  bnez    a0, fail

  /* C + A^T x B. */
  la      s0, matrix_c
  MLS_RM  2, 8
  MMOV_A_M 0, 2
  ZTT_R3  0x4d, 0, 0, 1              /* mmulatacc.2d */
  MMOV_M_A 2, 0
  la      s0, result
  MSS_RM  2, 8
  la      a0, result
  la      a1, expected_at
  call    compare_words
  bnez    a0, fail

  /* C + A x B^T. */
  la      s0, matrix_c
  MLS_RM  2, 8
  MMOV_A_M 0, 2
  ZTT_R3  0x4f, 0, 0, 1              /* mmulbtacc.2d */
  MMOV_M_A 2, 0
  la      s0, result
  MSS_RM  2, 8
  la      a0, result
  la      a1, expected_bt
  call    compare_words
  bnez    a0, fail

  li      t1, 1
  j       finish
fail:
  li      t1, 3
finish:
  la      t0, tohost
  sd      t1, 0(t0)
  fence
1: j       1b

compare_words:
  li      t2, 16
1:
  lw      t3, 0(a0)
  lw      t4, 0(a1)
  bne     t3, t4, 2f
  addi    a0, a0, 4
  addi    a1, a1, 4
  addi    t2, t2, -1
  bnez    t2, 1b
  li      a0, 0
  ret
2:
  li      a0, 1
  ret

.section .rodata
.balign 16
matrix_a:
  .word 1, 2, 3, 4
  .word 0, 1, 2, 3
  .word 2, 0, 1, 2
  .word 3, 2, 0, 1
matrix_b:
  .word 2, 1, 0, 1
  .word 0, 2, 1, 0
  .word 3, 1, 2, 1
  .word 1, 0, 1, 2
matrix_c:
  .word 1, 2, 3, 4
  .word 4, 3, 2, 1
  .word 2, 2, 2, 2
  .word 7, 7, 7, 7
expected_add:
  .word 16, 10, 15, 16
  .word 13, 7, 10, 9
  .word 11, 5, 6, 9
  .word 14, 14, 10, 12
expected_sub:
  .word -14, -6, -9, -8
  .word -5, -1, -6, -7
  .word -7, -1, -2, -5
  .word 0, 0, 4, 2
expected_at:
  .word 12, 5, 10, 13
  .word 10, 7, 5, 7
  .word 11, 10, 6, 6
  .word 22, 19, 15, 15
expected_bt:
  .word 9, 9, 18, 16
  .word 8, 7, 10, 9
  .word 8, 3, 12, 9
  .word 16, 11, 19, 12

.section .bss
.balign 16
result:
  .space 64

.section .tohost,"aw",@progbits
.balign 8
.globl tohost
tohost:
  .dword 0
.globl fromhost
fromhost:
  .dword 0
