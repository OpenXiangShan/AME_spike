/* Ztt v0.6 provisional-encoding fetch/decode smoke test. */
.option norvc
.text
.globl _start
_start:
  /* AME instructions require MS != Off before ownership is acquired. */
  li      t0, 0x02000000
  csrs    mstatus, t0

  /* ame.acquire a0, x0: dedicated backend, non-waiting descriptor. */
  .word   0xa490052b
  li      t1, 1
  bne     a0, t1, fail

  /* Configure m0, m1, and m2 as Uint32. */
  li      t0, 32
  .word   0xa232802b                    /* msettyp m0, t0 */
  .word   0xa23280ab                    /* msettyp m1, t0 */
  .word   0xa232812b                    /* msettyp m2, t0 */

  /* mbcast.m.x carries an explicit source descriptor in rs2. */
  li      t2, 32
  li      t1, 2
  .word   0xb07300ab                    /* mbcast.m.x m1, t1, t2 */
  li      t1, 3
  .word   0xb073012b                    /* mbcast.m.x m2, t1, t2 */

  .word   0x0420802b                    /* madd.ew m0, m1, m2 */
  la      t0, result
  .word   0xa3b2802b                    /* mss.rm m0, t0 */
  lw      t2, 0(t0)
  li      t3, 5
  bne     t2, t3, fail

  .word   0xa22003ab                    /* mgettyp t2, m0 */
  li      t3, 32
  bne     t2, t3, fail

  csrr    t2, 0xcc0                    /* amenlen */
  li      t3, 4
  bne     t2, t3, fail
  csrr    t2, 0xcc1                    /* ameudsz */
  li      t3, 32
  bne     t2, t3, fail
  csrr    t2, 0xcc3                    /* ameown */
  li      t3, 1
  bne     t2, t3, fail

  .word   0xa4a0002b                    /* ame.release */
  csrr    t2, 0xcc3
  bnez    t2, fail

  li      t1, 1
  j       finish

fail:
  li      t1, 3

finish:
  la      t0, tohost
  sd      t1, 0(t0)
  fence
1:
  j       1b

.section .bss
.balign 64
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
