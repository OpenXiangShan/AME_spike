# Copyright (c) 2026 BOSC & ICT, CAS
# All rights reserved.
# See LICENSE for license details.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
TOOLCHAIN_ROOT ?= $(if $(AME_TOOLCHAIN_ROOT),$(AME_TOOLCHAIN_ROOT),$(if $(AMESPIKE_TOOLCHAIN),$(AMESPIKE_TOOLCHAIN),$(REPO_ROOT)/toolchain/install))
CLANG ?= $(if $(AMESPIKE_CLANG),$(AMESPIKE_CLANG),$(TOOLCHAIN_ROOT)/bin/clang)
LLVM_OBJDUMP ?= $(if $(AMESPIKE_OBJDUMP),$(AMESPIKE_OBJDUMP),$(TOOLCHAIN_ROOT)/bin/llvm-objdump)
LD_LLD ?= $(if $(AMESPIKE_LD_LLD),$(AMESPIKE_LD_LLD),$(TOOLCHAIN_ROOT)/bin/ld.lld)
CLANG_RESOURCE_DIR ?= $(if $(AMESPIKE_CLANG_RESOURCE_DIR),$(AMESPIKE_CLANG_RESOURCE_DIR),$(lastword $(sort $(wildcard $(TOOLCHAIN_ROOT)/lib/clang/*))))
TARGET_LIBGCC ?= $(if $(AMESPIKE_LIBGCC),$(AMESPIKE_LIBGCC),$(firstword $(shell find "$(TOOLCHAIN_ROOT)" -type f -name libgcc.a -print 2>/dev/null | sort)))

TARGET_FLAGS := \
	-target riscv64-unknown-elf \
	-resource-dir $(CLANG_RESOURCE_DIR) \
	-march=rv64gcv_xxiangshanztt \
	-mabi=lp64d \
	-mcmodel=medany \
	-B$(TOOLCHAIN_ROOT)/bin

TARGET_LDFLAGS := \
	-nostdlib -nostartfiles -static \
	--ld-path=$(LD_LLD)

.PHONY: check-toolchain
check-toolchain:
	@test -x "$(CLANG)" || { echo "[FAIL] Run ./scripts/build_toolchain.sh first"; exit 1; }
	@test -x "$(LLVM_OBJDUMP)" || { echo "[FAIL] Missing llvm-objdump: $(LLVM_OBJDUMP)"; exit 1; }
	@test -x "$(LD_LLD)" || { echo "[FAIL] Missing ld.lld: $(LD_LLD)"; exit 1; }
	@test -d "$(CLANG_RESOURCE_DIR)" || { echo "[FAIL] Missing Clang resource directory"; exit 1; }
	@test -f "$(TARGET_LIBGCC)" || { echo "[FAIL] Missing libgcc.a under $(TOOLCHAIN_ROOT)"; exit 1; }
