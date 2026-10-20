#!/usr/bin/env bash
#
# Create or validate an independent bare core-math mirror for CI.
#
# This script never updates an existing destination.  To publish a newer
# mirror, create and validate a new destination, then let the maintainer
# switch the deployed path during a maintenance window.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  init-core-math-mirror.sh --source SOURCE --destination PATH [--commit SHA]

Create an independent bare Git mirror, or validate an existing one.

Options:
  --source SOURCE       Existing local Git repository or accessible Git URL.
  --destination PATH    Bare mirror path to create or validate.
  --commit SHA          Require this full 40- or 64-character commit.
  -h, --help            Show this help.

The destination is created atomically through a temporary sibling directory.
An existing destination is never overwritten or deleted.  Local repositories
are cloned with --no-local so the result has no dependency on hardlinks,
alternates, or the source repository.

Examples:
  init-core-math-mirror.sh \
    --source /path/to/core-math \
    --destination /srv/ci-deps/core-math.git \
    --commit 708e86ef634414431070336099bb5745f061dc21

  init-core-math-mirror.sh \
    --source https://gitlab.inria.fr/core-math/core-math.git \
    --destination /srv/ci-deps/core-math.git
EOF
}

die() {
  echo "init-core-math-mirror.sh: $*" >&2
  exit 2
}

source_repo=
destination=
required_commit=

while (($#)); do
  case "$1" in
    --source)
      (($# >= 2)) || die "--source requires a value"
      source_repo=$2
      shift 2
      ;;
    --destination)
      (($# >= 2)) || die "--destination requires a value"
      destination=$2
      shift 2
      ;;
    --commit)
      (($# >= 2)) || die "--commit requires a value"
      required_commit=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -n "$source_repo" ]] || die "--source is required"
[[ -n "$destination" ]] || die "--destination is required"
if [[ -n "$required_commit" ]] &&
  [[ ! "$required_commit" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]]; then
  die "--commit must be a full lowercase 40- or 64-character SHA"
fi

destination_parent="$(dirname -- "$destination")"
destination_name="$(basename -- "$destination")"
mkdir -p -- "$destination_parent"
destination_parent="$(cd -- "$destination_parent" && pwd -P)"
destination="$destination_parent/$destination_name"

verify_mirror() {
  local mirror=$1
  if [[ ! -d "$mirror" || ! -r "$mirror" || ! -x "$mirror" ]]; then
    die "destination is missing or unreadable: $mirror"
  fi
  [[ "$(git -C "$mirror" rev-parse --is-bare-repository 2>/dev/null)" == true ]] ||
    die "destination is not a bare Git repository: $mirror"
  if [[ -e "$mirror/objects/info/alternates" ]]; then
    die "destination uses Git alternates and is not independent: $mirror"
  fi
  git -C "$mirror" fsck --full --no-progress >/dev/null ||
    die "destination Git object database is incomplete: $mirror"
  if [[ -n "$required_commit" ]] &&
    ! git -C "$mirror" cat-file -e "${required_commit}^{commit}" 2>/dev/null; then
    die "destination does not contain required commit $required_commit: $mirror"
  fi
}

if [[ -e "$destination" || -L "$destination" ]]; then
  verify_mirror "$destination"
  if [[ -n "$required_commit" ]]; then
    echo "validated $destination contains $required_commit"
  else
    echo "validated independent mirror: $destination"
  fi
  exit 0
fi

local_source=false
if [[ -d "$source_repo" ]] &&
  git -C "$source_repo" rev-parse --git-dir >/dev/null 2>&1; then
  local_source=true
fi

temporary=
cleanup() {
  local status=$?
  if [[ -n "$temporary" && ( -e "$temporary" || -L "$temporary" ) ]]; then
    rm -rf -- "$temporary"
  fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

temporary="$(mktemp -d --tmpdir="$destination_parent" ".${destination_name}.tmp.XXXXXX")"
rmdir -- "$temporary"
if [[ "$local_source" == true ]]; then
  git clone --mirror --no-local -- "$source_repo" "$temporary"
else
  git clone --mirror -- "$source_repo" "$temporary"
fi

# Do not leave a personal source path as the mirror's implicit update source.
# Future updates are performed by an administrator with an explicit source.
git -C "$temporary" config --unset-all remote.origin.url || true
verify_mirror "$temporary"
if [[ -e "$destination" || -L "$destination" ]]; then
  die "destination appeared while creating mirror; refusing to replace it: $destination"
fi
mv -- "$temporary" "$destination"
temporary=
if [[ -n "$required_commit" ]]; then
  echo "created $destination containing $required_commit"
else
  echo "created independent mirror: $destination"
fi
