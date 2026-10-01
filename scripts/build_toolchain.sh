#!/usr/bin/env bash

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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LLVM_SUBMODULE="$REPO_ROOT/toolchain/AME_llvm"
INSTALL_DIR="$REPO_ROOT/toolchain/install"

LLVM_ARCHIVE="$LLVM_SUBMODULE/ame_toolchain.tar.gz"
LLVM_ARCHIVE_PART_PREFIX="$LLVM_SUBMODULE/ame_toolchain.tar.gz.part"
LLVM_ARCHIVE_CHECKSUM="$LLVM_SUBMODULE/ame_toolchain.sha256"

STAGING_DIR=""
BACKUP_DIR=""
MERGED_ARCHIVE=""
ARCHIVE_TO_EXTRACT=""

cleanup() {
    if [[ -n "$STAGING_DIR" && -d "$STAGING_DIR" ]]; then
        rm -rf -- "$STAGING_DIR"
    fi

    if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
        if [[ ! -e "$INSTALL_DIR" ]]; then
            mv -- "$BACKUP_DIR" "$INSTALL_DIR"
        else
            rm -rf -- "$BACKUP_DIR"
        fi
    fi
}

trap cleanup EXIT

require_dir() {
    [[ -d "$1" ]] || {
        echo "[FAIL] Required directory not found: $1" >&2
        exit 1
    }
}

require_file() {
    [[ -f "$1" ]] || {
        echo "[FAIL] Required file not found: $1" >&2
        exit 1
    }
}

verify_checksum() {
    local archive=$1
    local expected_checksum
    local actual_checksum

    require_file "$LLVM_ARCHIVE_CHECKSUM"

    expected_checksum=$(awk 'NF { print $1; exit }' "$LLVM_ARCHIVE_CHECKSUM")
    if [[ ! "$expected_checksum" =~ ^[[:xdigit:]]{64}$ ]]; then
        echo "[FAIL] Invalid SHA256 checksum in:" >&2
        echo "       $LLVM_ARCHIVE_CHECKSUM" >&2
        exit 1
    fi

    actual_checksum=$(sha256sum "$archive" | awk '{ print $1 }')

    if [[ "$actual_checksum" != "$expected_checksum" ]]; then
        echo "[FAIL] Toolchain archive checksum mismatch" >&2
        echo "       expected: $expected_checksum" >&2
        echo "       actual:   $actual_checksum" >&2
        exit 1
    fi

    echo "[PASS] Toolchain archive checksum verified"
}

# Initialize the pinned submodule revisions before selecting the toolchain
# archive so core-math is available even when a complete archive exists.
echo "[INFO] Initializing all submodules"
git -C "$REPO_ROOT" submodule sync --recursive
git -C "$REPO_ROOT" -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
    submodule update --init --recursive --progress

require_dir "$REPO_ROOT/core-math"
require_dir "$LLVM_SUBMODULE"

#
# ----------------------------------------------------------------------
# 1. Select toolchain archive source
# ----------------------------------------------------------------------
#

if [[ -f "$LLVM_ARCHIVE" ]]; then
    echo "[INFO] Using local toolchain archive:"
    echo "       $LLVM_ARCHIVE"

    echo "[INFO] Skipping SHA256 verification for local toolchain archive"
    ARCHIVE_TO_EXTRACT="$LLVM_ARCHIVE"

else
    echo "[INFO] Complete toolchain archive not found; looking for split parts"

    mapfile -t LLVM_ARCHIVE_PARTS < <(
        find "$LLVM_SUBMODULE" \
            -maxdepth 1 \
            -type f \
            -name "$(basename "$LLVM_ARCHIVE_PART_PREFIX")*" \
            -print |
            sort -V
    )

    if [[ ${#LLVM_ARCHIVE_PARTS[@]} -eq 0 ]]; then
        echo "[FAIL] No toolchain archive or archive parts found" >&2
        echo "       Expected:" >&2
        echo "       $LLVM_ARCHIVE" >&2
        echo "       or:" >&2
        echo "       $LLVM_ARCHIVE_PART_PREFIX*" >&2
        exit 1
    fi

    STAGING_DIR=$(
        mktemp -d "$REPO_ROOT/toolchain/.install-stage.XXXXXX"
    )

    MERGED_ARCHIVE="$STAGING_DIR/ame_toolchain.tar.gz"

    echo "[INFO] Reconstructing toolchain archive from parts"

    cat "${LLVM_ARCHIVE_PARTS[@]}" > "$MERGED_ARCHIVE"

    verify_checksum "$MERGED_ARCHIVE"
    ARCHIVE_TO_EXTRACT="$MERGED_ARCHIVE"
fi

#
# ----------------------------------------------------------------------
# 2. Check archive integrity
# ----------------------------------------------------------------------
#

echo "[INFO] Checking toolchain archive integrity"

tar -tzf "$ARCHIVE_TO_EXTRACT" >/dev/null

#
# ----------------------------------------------------------------------
# 3. Extract into staging directory
# ----------------------------------------------------------------------
#

if [[ -z "$STAGING_DIR" ]]; then
    STAGING_DIR=$(
        mktemp -d "$REPO_ROOT/toolchain/.install-stage.XXXXXX"
    )
fi

echo "[INFO] Extracting AME LLVM RISC-V toolchain package"

tar -xzf "$ARCHIVE_TO_EXTRACT" -C "$STAGING_DIR"

#
# If the archive was reconstructed inside staging, remove the temporary
# merged archive before installing the directory.
#
if [[ -n "$MERGED_ARCHIVE" && -f "$MERGED_ARCHIVE" ]]; then
    rm -f -- "$MERGED_ARCHIVE"
    MERGED_ARCHIVE=""
fi

# ----------------------------------------------------------------------
# 4. Atomically install the toolchain
# ----------------------------------------------------------------------
#

echo "[INFO] Installing toolchain"

if [[ -e "$INSTALL_DIR" ]]; then
    BACKUP_DIR=$(
        mktemp -d "$REPO_ROOT/toolchain/.install-backup.XXXXXX"
    )

    rmdir -- "$BACKUP_DIR"

    mv -- "$INSTALL_DIR" "$BACKUP_DIR"
fi

mv -- "$STAGING_DIR" "$INSTALL_DIR"
STAGING_DIR=""

if [[ -n "$BACKUP_DIR" ]]; then
    rm -rf -- "$BACKUP_DIR"
    BACKUP_DIR=""
fi

echo
echo "[PASS] toolchain installed"
echo "       Install: $INSTALL_DIR"
