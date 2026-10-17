#!/usr/bin/env bash
#
# Build the Spike used by the open06 Ztt regression.  The build directory is
# deliberately supplied by the workflow so that a run never reuses a binary
# from an earlier commit.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD_ROOT="${1:-${ZTT_CI_BUILD_ROOT:-}}"
JOBS="${ZTT_CI_BUILD_JOBS:-2}"

if [[ -z "$BUILD_ROOT" ]]; then
  BUILD_ROOT="$ROOT/.ci-ztt-build"
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ZTT_CI_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
if [[ -e "$BUILD_ROOT" ]]; then
  echo "refusing to reuse existing build directory: $BUILD_ROOT" >&2
  exit 2
fi

mkdir -p "$BUILD_ROOT"

if [[ -f "$ROOT/.gitmodules" ]]; then
  git -C "$ROOT" submodule sync -- core-math
  git -C "$ROOT" submodule update --init --recursive core-math
  expected_core_math="$(
    git -C "$ROOT" ls-tree HEAD core-math | awk '$2 == "commit" { print $3 }'
  )"
  actual_core_math="$(git -C "$ROOT/core-math" rev-parse HEAD)"
  if [[ -z "$expected_core_math" || "$actual_core_math" != "$expected_core_math" ]]; then
    echo "core-math checkout does not match the source gitlink" >&2
    echo "expected: ${expected_core_math:-missing}" >&2
    echo "actual: $actual_core_math" >&2
    exit 2
  fi
fi

BUILD_DIR="$BUILD_ROOT/build"
INSTALL_DIR="$BUILD_ROOT/install"
mkdir -p "$BUILD_DIR/spike" "$INSTALL_DIR"

if commit="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"; then
  :
else
  commit="${GITHUB_SHA:-unknown}"
fi

{
  echo "source_root=$ROOT"
  echo "commit=$commit"
  echo "build_root=$BUILD_ROOT"
  echo "jobs=$JOBS"
  echo "submodule_core_math=$(git -C "$ROOT/core-math" rev-parse HEAD 2>/dev/null || echo unavailable)"
} | tee "$BUILD_ROOT/build-metadata.env"

(
  cd "$BUILD_DIR/spike"
  "$ROOT/configure" --prefix="$INSTALL_DIR"
  make -j"$JOBS"
  make check
  make install install-hdrs-list.h
  "$INSTALL_DIR/bin/spike" -h >/dev/null
)

test -x "$INSTALL_DIR/bin/spike"
sha256sum "$INSTALL_DIR/bin/spike" | tee "$BUILD_ROOT/spike.sha256"
printf '%s\n' "$BUILD_ROOT" > "$BUILD_ROOT/build-root.txt"
