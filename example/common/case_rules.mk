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
.DELETE_ON_ERROR:

ifndef CASE_BINARY
$(error CASE_BINARY must be set before including common/case_rules.mk)
endif
ifndef CASE_DESCRIPTION
$(error CASE_DESCRIPTION must be set before including common/case_rules.mk)
endif
ifndef REQUIRED_MNEMONICS
$(error REQUIRED_MNEMONICS must be set before including common/case_rules.mk)
endif

CASE_DIR := $(CURDIR)
EXAMPLE_ROOT := $(abspath $(CASE_DIR)/..)
AMESPIKE_ROOT := $(abspath $(EXAMPLE_ROOT)/..)
COMMON_DIR := $(EXAMPLE_ROOT)/common
RUNTIME_DIR := $(EXAMPLE_ROOT)/common

BUILD_DIR ?= $(CASE_DIR)/build
HOST_DIR := $(BUILD_DIR)/host
TARGET_DIR := $(BUILD_DIR)/target
REPORT_DIR := $(BUILD_DIR)/reports
DISASM_DIR := $(BUILD_DIR)/disasm
LOG_DIR := $(BUILD_DIR)/logs
GENERATED_DIR := $(BUILD_DIR)/generated
BUILD_CONFIG := $(GENERATED_DIR)/build-config.txt

SOURCE := $(CASE_DIR)/main.c
OPERATOR_SOURCE := $(CASE_DIR)/operators.c
SOURCES := $(SOURCE) $(OPERATOR_SOURCE)
HOST_BINARY := $(HOST_DIR)/$(CASE_BINARY)
TARGET_BINARY := $(TARGET_DIR)/$(CASE_BINARY)
HOST_REPORT := $(REPORT_DIR)/$(CASE_BINARY).host.txt
SPIKE_REPORT := $(REPORT_DIR)/$(CASE_BINARY).spike.txt
CASE_REPORT := $(REPORT_DIR)/REPORT.md
DISASSEMBLY := $(DISASM_DIR)/$(CASE_BINARY).dis
CASE_DISPLAY := $(patsubst test_llm_%,%,$(CASE_BINARY))

# Validation modes:
#   GOLDEN=host       compare Spike with the host C implementation
#   GOLDEN=reference  compare Spike with MATRIX_REFERENCE
GOLDEN ?= host
MATRIX_INPUT ?=
MATRIX_REFERENCE ?=

ifneq ($(filter $(GOLDEN),host reference),$(GOLDEN))
$(error GOLDEN must be either host or reference)
endif

ifneq ($(strip $(MATRIX_INPUT)),)
override MATRIX_INPUT := $(abspath $(MATRIX_INPUT))
MATRIX_HEADER := $(GENERATED_DIR)/ztt_matrix_input.h
MATRIX_BUILD_FLAGS := -DZTT_EXTERNAL_MATRIX_INPUT -include $(MATRIX_HEADER)
GENERATED_HEADERS := $(MATRIX_HEADER)
endif

ifeq ($(GOLDEN),reference)
ifeq ($(strip $(MATRIX_INPUT)),)
$(error GOLDEN=reference requires MATRIX_INPUT=/path/to/input.json)
endif
override MATRIX_REFERENCE := $(if $(strip $(MATRIX_REFERENCE)),$(abspath $(MATRIX_REFERENCE)),)
REFERENCE_REPORT := $(REPORT_DIR)/$(CASE_BINARY).reference.txt
GOLDEN_REPORT := $(REFERENCE_REPORT)
GOLDEN_LABEL := External reference
RUN_DEPENDENCIES := $(REFERENCE_REPORT) $(SPIKE_REPORT)
else
GOLDEN_REPORT := $(HOST_REPORT)
GOLDEN_LABEL := Host C golden
RUN_DEPENDENCIES := $(HOST_REPORT) $(SPIKE_REPORT)
endif

HOST_CC ?= $(CC)
include $(COMMON_DIR)/toolchain.mk
SPIKE ?= $(AMESPIKE_ROOT)/build/spike
SPIKE_TIMEOUT ?= 120

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/*.h)
RUNTIME_SOURCES := $(RUNTIME_DIR)/crt0.S $(RUNTIME_DIR)/runtime.c
RUNTIME_HEADERS := $(wildcard $(RUNTIME_DIR)/include/*.h)

HOST_CFLAGS := -O2 -g -Wall -Wextra -Werror -I$(COMMON_DIR) $(MATRIX_BUILD_FLAGS)
TARGET_CFLAGS := \
	$(TARGET_FLAGS) \
	-O2 -g -Wall -Wextra -Werror \
	-ffreestanding -fno-builtin \
	-ffunction-sections -fdata-sections \
	-fno-vectorize -fno-slp-vectorize -ffp-contract=off \
	-I$(COMMON_DIR) -I$(RUNTIME_DIR)/include $(MATRIX_BUILD_FLAGS)
TARGET_LDFLAGS += \
	-Wl,-T,$(RUNTIME_DIR)/link.ld \
	-Wl,--gc-sections

.DEFAULT_GOAL := all
.PHONY: all test run audit report clean force check-host-tools check-target-tools check-spike

force:

all: $(HOST_BINARY) $(TARGET_BINARY) audit

test:
	@echo "[TEST] $(CASE_DISPLAY): $(CASE_DESCRIPTION)"
	+$(MAKE) --no-print-directory report

run: $(RUN_DEPENDENCIES)
	@mkdir -p $(LOG_DIR)
	@if cmp -s $(GOLDEN_REPORT) $(SPIKE_REPORT); then \
		rm -f $(LOG_DIR)/golden-spike.diff; \
		echo "[PASS] $(CASE_DISPLAY): Spike == $(GOLDEN_LABEL)"; \
	else \
		echo "[FAIL] result comparison: $(CASE_DISPLAY)"; \
		diff -u $(GOLDEN_REPORT) $(SPIKE_REPORT) | tee $(LOG_DIR)/golden-spike.diff; \
		exit 1; \
	fi

audit: $(DISASSEMBLY)
	@for mnemonic in $(REQUIRED_MNEMONICS); do \
		if ! grep -Fq "$$mnemonic" $(DISASSEMBLY); then \
			echo "[FAIL] $(CASE_DISPLAY): missing instruction $$mnemonic"; \
			exit 1; \
		fi; \
	done
	@echo "[PASS] $(CASE_DISPLAY): instruction audit"

report: run audit
	python3 $(COMMON_DIR)/write_report.py case \
		--name $(CASE_BINARY) \
		--description "$(CASE_DESCRIPTION)" \
		--source "$(notdir $(SOURCE)) + $(notdir $(OPERATOR_SOURCE))" \
		--golden $(GOLDEN_REPORT) \
		--golden-label "$(GOLDEN_LABEL)" \
		--spike $(SPIKE_REPORT) \
		--output $(CASE_REPORT)
	@echo "[PASS] $(CASE_DISPLAY): report $(CASE_REPORT)"

check-host-tools:
	@command -v "$(HOST_CC)" >/dev/null 2>&1 || test -x "$(HOST_CC)" || { \
		echo "[FAIL] host compiler not found: $(HOST_CC)"; exit 1; }

check-target-tools: check-toolchain

check-spike:
	@test -x "$(SPIKE)" || { echo "[FAIL] Spike not found: $(SPIKE)"; exit 1; }

ifeq ($(NO_BUILD),1)
$(HOST_BINARY):
	@test -x $@ || { echo "[FAIL] --no-build binary is missing: $@"; exit 1; }

$(TARGET_BINARY):
	@test -f $@ || { echo "[FAIL] --no-build binary is missing: $@"; exit 1; }
else
$(HOST_BINARY): $(SOURCES) $(COMMON_HEADERS) $(GENERATED_HEADERS) $(BUILD_CONFIG) | check-host-tools
	@mkdir -p $(HOST_DIR) $(LOG_DIR)
	$(HOST_CC) $(HOST_CFLAGS) $(SOURCES) -o $@ 2>&1 | tee $(LOG_DIR)/host-build.log

$(TARGET_BINARY): $(SOURCES) $(COMMON_HEADERS) $(GENERATED_HEADERS) $(BUILD_CONFIG) $(RUNTIME_SOURCES) $(RUNTIME_HEADERS) $(RUNTIME_DIR)/link.ld $(COMMON_DIR)/toolchain.mk $(COMMON_DIR)/case_rules.mk | check-target-tools
	@mkdir -p $(TARGET_DIR) $(LOG_DIR)
	$(CLANG) $(TARGET_CFLAGS) $(RUNTIME_SOURCES) $(SOURCES) \
		$(TARGET_LDFLAGS) $(TARGET_LIBGCC) -o $@ 2>&1 | tee $(LOG_DIR)/target-build.log
endif

$(BUILD_CONFIG): force
	@mkdir -p $(GENERATED_DIR)
	@printf '%s\n' \
		'MATRIX_INPUT=$(MATRIX_INPUT)' \
		'MATRIX_BUILD_FLAGS=$(MATRIX_BUILD_FLAGS)' > $@.tmp
	@if ! cmp -s $@.tmp $@; then cp $@.tmp $@; fi
	@rm -f $@.tmp

ifneq ($(strip $(MATRIX_INPUT)),)
$(MATRIX_HEADER): force $(MATRIX_INPUT) $(COMMON_DIR)/matrix_case.py
	@mkdir -p $(GENERATED_DIR)
	python3 $(COMMON_DIR)/matrix_case.py header \
		--input $(MATRIX_INPUT) --output $@
endif

ifeq ($(GOLDEN),reference)
$(REFERENCE_REPORT): force $(MATRIX_INPUT) $(MATRIX_REFERENCE) $(COMMON_DIR)/matrix_case.py
	@mkdir -p $(REPORT_DIR) $(LOG_DIR)
	python3 $(COMMON_DIR)/matrix_case.py report \
		--input $(MATRIX_INPUT) \
		$(if $(MATRIX_REFERENCE),--reference $(MATRIX_REFERENCE),) \
		--output $@
	@echo "[PASS] $(CASE_DISPLAY): external reference prepared"
endif

$(DISASSEMBLY): $(TARGET_BINARY) | check-target-tools
	@mkdir -p $(DISASM_DIR) $(LOG_DIR)
	$(LLVM_OBJDUMP) -d --mattr=+xxiangshanztt $< > $@

$(HOST_REPORT): force $(HOST_BINARY)
	@mkdir -p $(REPORT_DIR) $(LOG_DIR)
	$(HOST_BINARY) $@ 2>&1 | tee $(LOG_DIR)/host-run.log
	@test -f $@ || { echo "[FAIL] Host report was not generated: $@"; exit 1; }
	@echo "[PASS] $(CASE_DISPLAY): Host"

$(SPIKE_REPORT): force $(TARGET_BINARY) | check-spike
	@mkdir -p $(REPORT_DIR) $(LOG_DIR)
	timeout $(SPIKE_TIMEOUT) $(SPIKE) --isa=rv64gcv_ztt $(TARGET_BINARY) \
		> $@.tmp 2> $(LOG_DIR)/spike-run.log || { \
		status=$$?; cat $(LOG_DIR)/spike-run.log; \
		echo "[FAIL] Spike startup/runtime: exit $$status (stdout: $@.tmp)"; exit $$status; }
	@test -s $@.tmp || { echo "[FAIL] runtime: empty Spike report: $@.tmp"; exit 1; }
	mv $@.tmp $@
	@echo "[PASS] $(CASE_DISPLAY): Spike"

clean:
	rm -rf $(BUILD_DIR)
