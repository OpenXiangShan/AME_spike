#!/usr/bin/env bash
#
# Build the Spike used by the open06 Ztt regression.  The build directory is
# deliberately supplied by the workflow so that a run never reuses a binary
# from an earlier commit.  Open06 sets ZTT_CI_CORE_MATH_MIRROR to a
# maintainer-owned read-only mirror; ordinary local builds retain submodule
# behavior when that variable is unset.
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

expected_core_math="$(
  git -C "$ROOT" ls-tree HEAD core-math | awk '$2 == "commit" { print $3 }'
)"
if [[ ! "$expected_core_math" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source checkout has no valid core-math gitlink" >&2
  echo "expected: ${expected_core_math:-missing}" >&2
  exit 2
fi

if [[ -n "${ZTT_CI_CORE_MATH_MIRROR:-}" ]]; then
  core_math_mirror="$ZTT_CI_CORE_MATH_MIRROR"
  if [[ ! -d "$core_math_mirror" || ! -r "$core_math_mirror" || ! -x "$core_math_mirror" ]]; then
    echo "core-math mirror is missing or unreadable: $core_math_mirror" >&2
    exit 2
  fi
  if ! git -c "safe.directory=$core_math_mirror" \
    -C "$core_math_mirror" rev-parse --git-dir >/dev/null; then
    echo "core-math mirror is not a readable Git repository: $core_math_mirror" >&2
    exit 2
  fi
  if ! git -c "safe.directory=$core_math_mirror" \
    -C "$core_math_mirror" cat-file -e "${expected_core_math}^{commit}"; then
    echo "core-math mirror does not contain the source gitlink commit" >&2
    echo "mirror: $core_math_mirror" >&2
    echo "required: $expected_core_math" >&2
    exit 2
  fi

  if [[ -e "$ROOT/core-math" ]]; then
    if [[ ! -d "$ROOT/core-math" ]]; then
      echo "refusing to replace non-directory core-math path: $ROOT/core-math" >&2
      exit 2
    fi
    if [[ -n "$(find "$ROOT/core-math" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      core_math_toplevel="$(git -C "$ROOT/core-math" rev-parse --show-toplevel 2>/dev/null || true)"
      core_math_path="$(cd "$ROOT/core-math" && pwd -P)"
      core_math_git_dir_raw="$(
        git -C "$ROOT/core-math" rev-parse --absolute-git-dir 2>/dev/null || true
      )"
      core_math_git_dir=""
      if [[ -n "$core_math_git_dir_raw" ]]; then
        core_math_git_dir="$(cd "$core_math_git_dir_raw" && pwd -P)"
      fi
      superproject_git_dir="$(
        cd "$(git -C "$ROOT" rev-parse --absolute-git-dir)" && pwd -P
      )"
      if [[ "$core_math_toplevel" != "$core_math_path" ||
        ! -f "$ROOT/core-math/.git" ||
        "$core_math_git_dir" != "$superproject_git_dir/modules/core-math" ]]; then
        echo "refusing to replace a non-empty path that is not an initialized core-math submodule: $ROOT/core-math" >&2
        exit 2
      fi
    fi
  fi
  temp_git_config="$(mktemp)"
  chmod 600 "$temp_git_config"
  trap 'rm -f -- "$temp_git_config"' EXIT
  git config -f "$temp_git_config" --add safe.directory "$core_math_mirror"
  git config -f "$temp_git_config" --add protocol.file.allow always
  git config -f "$temp_git_config" \
    --add submodule.core-math.url "$core_math_mirror"
  GIT_CONFIG_GLOBAL="$temp_git_config" \
    git submodule update --init --recursive core-math
elif [[ -f "$ROOT/.gitmodules" ]]; then
  git -C "$ROOT" submodule sync -- core-math
  git -C "$ROOT" submodule update --init --recursive core-math
fi

actual_core_math="$(git -C "$ROOT/core-math" rev-parse HEAD)"
if [[ "$actual_core_math" != "$expected_core_math" ]]; then
  echo "core-math checkout does not match the source gitlink" >&2
  echo "expected: $expected_core_math" >&2
  echo "actual: $actual_core_math" >&2
  exit 2
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
