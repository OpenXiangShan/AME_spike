# AMEspike C examples

The GEMM testcases, matrix inputs, Host/reference calculations, and instruction
audit come from AME_gem5. The build and bare-metal runtime target AMEspike.

The suite matches the three cases in `AME_gem5/example`:

| Case | Computation | Default size |
| --- | --- | --- |
| `gemm_fp32` | FP32 GEMM, `C += A x B` | N=128 |
| `gemm_int8` | INT8 GEMM with INT32 accumulator and INT32 → INT64 → INT32 conversion | N=128 |
| `gemm_mixed` | INT8 × INT8 → INT32 widening GEMM | N=128 |

Run from the repository root:

```sh
make -C example build
make -C example/gemm_fp32 run SPIKE="$PWD/build/spike"
make -C example gemm_int8 SPIKE="$PWD/build/spike"
make -C example gemm_mixed SPIKE="$PWD/build/spike"
make -C example test SPIKE="$PWD/build/spike"
make -C example/gemm_fp32 test-host-golden SPIKE="$PWD/build/spike"
make -C example/gemm_fp32 test-reference SPIKE="$PWD/build/spike"
```

All three default testcases use the original generated N=128 matrices.
For `gemm_fp32`, `test-host-golden` and `test-reference` use the unchanged N=8
JSON configuration. Its `MATRIX_INPUT`, `MATRIX_REFERENCE`, and
`GOLDEN=host|reference` options remain available. The two integer cases use
their generated matrices and Host C golden.

Terminal output follows gem5's `[TEST]`, `[PASS]`/`[FAIL]`, and `[RESULT]`
format. The backend is correctly labeled `Spike` rather than `MinorCPU`.
A successful `make -C example test` ends with:

```text
[RESULT] example suite: PASS (3/3 cases passed)
         PASS: gemm_fp32
         PASS: gemm_int8
         PASS: gemm_mixed
[PASS] suite report: .../example/build/reports/EXAMPLE_REPORT.md
```

## Toolchain

`common/toolchain.mk` defaults to `../toolchain/install`. Override it with
`AME_TOOLCHAIN_ROOT=/path/to/toolchain`. It uses Clang's integrated assembler,
packaged `ld.lld`, recursively discovered `libgcc.a`, and `llvm-objdump` with
`--mattr=+xxiangshanztt`. No GNU assembler, linker, or compiler is required.

## Spike runtime

`crt0.S` initializes scalar FP, vector, and Spike's AME state (mstatus bit 25).
The linker retains the `tohost`/`fromhost` symbols. The runtime emits text using
Spike's HTIF console and exits through the same `tohost` pass/fail convention
as the old assembly examples. No proxy kernel or gem5 semihosting is needed.

The minimal target `fopen(..., "w")` maps report output to the HTIF console,
not to a host filesystem path. Startup supplies a synthetic program name and
report argument to the unchanged `main`. This is a testcase runtime, not a
general filesystem or command-line implementation. Target `stderr` also uses
the console; on failure its diagnostic text is retained in the temporary output.

The Makefile captures stdout into a temporary report and promotes it only after
a successful exit and a nonempty-output check. Simulator stderr goes to
`<case>/build/logs/spike-run.log`. A timeout fails the run (default 120 seconds,
overridable with `SPIKE_TIMEOUT`). It is never treated as a successful result.

Reports are compared byte-for-byte with `cmp`:

- `<case>/build/reports/test_llm_<case>.spike.txt`
- `<case>/build/reports/test_llm_<case>.host.txt` (or `.reference.txt` for FP32)
- `<case>/build/reports/REPORT.md`
- `build/reports/EXAMPLE_REPORT.md` after `make test`

Build errors are compile/link failures; missing audited mnemonics are
objdump/audit failures. Simulator errors/timeouts are startup/runtime failures;
unequal reports are result-comparison failures. None require changing Ztt
instruction semantics merely to make the example pass.
