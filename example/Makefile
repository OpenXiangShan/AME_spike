#
# Copyright (c) 2026 BOSC & ICT, CAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

CASES := \
	gemm_fp32 \
	gemm_int8 \
	gemm_mixed

BUILD_DIR := $(CURDIR)/build
SUITE_REPORT := $(BUILD_DIR)/reports/EXAMPLE_REPORT.md

.DEFAULT_GOAL := help
.NOTPARALLEL: test $(CASES)
.PHONY: help build test report stats clean $(CASES) \
	$(addprefix build-,$(CASES)) $(addprefix test-,$(CASES))

help:
	@echo "AMEspike Ztt v0.6 examples"
	@echo
	@echo "  make gemm_fp32   Build and test one case on Host and Spike"
	@echo "  make gemm_int8   Test INT8 GEMM with INT32/INT64 conversion"
	@echo "  make gemm_mixed  Test INT8 x INT8 to INT32 widening GEMM"
	@echo "  make -C gemm_fp32 test-host-golden  Use configured matrices + Host C golden"
	@echo "  make -C gemm_fp32 test-reference    Use configured matrices + reference"
	@echo "  make test        Build and test all cases"
	@echo "  make build       Build all Host/RISC-V binaries and disassemblies"
	@echo "  make clean       Remove all example build output"
	@echo
	@echo "Each case keeps a uniform main.c entry point and local operators.c source."
	@echo "Custom selection: make gemm_fp32 GOLDEN=host|reference MATRIX_INPUT=... [MATRIX_REFERENCE=...]"
	@echo
	@echo "Default toolchain: ../toolchain/install (run ../scripts/build_toolchain.sh first)"
	@echo
	@echo "Override the packaged toolchain or simulator:"
	@echo "  AME_TOOLCHAIN_ROOT=/path/to/toolchain"
	@echo "  SPIKE=/path/to/spike"

build: $(addprefix build-,$(CASES))

test: $(addprefix test-,$(CASES)) report

$(CASES):
	+$(MAKE) --no-print-directory -C $@ test

$(addprefix build-,$(CASES)): build-%:
	+$(MAKE) --no-print-directory -C $* all

$(addprefix test-,$(CASES)): test-%:
	+$(MAKE) --no-print-directory -C $* test

report:
	@mkdir -p $(dir $(SUITE_REPORT))
	python3 common/write_report.py suite --output $(SUITE_REPORT) \
		$(foreach case,$(CASES),$(case)/build/reports/REPORT.md)
	@echo "[PASS] suite report: $(SUITE_REPORT)"

stats:
	python3 common/code_stats.py $(CURDIR)

clean:
	@for case in $(CASES); do \
		$(MAKE) --no-print-directory -C $$case clean; \
	done
	rm -rf $(BUILD_DIR)
