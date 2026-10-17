# CI dependencies and open06 deployment

## Prepare the core-math mirror

The mirror is maintained by the CI administrator.  CI runners need only
read and traverse permission; they must not be able to update the mirror.

With network access, run the following on a maintenance host:

```bash
ci-tests/init-core-math-mirror.sh \
  --source https://gitlab.inria.fr/core-math/core-math.git \
  --destination /nfs/share/ci-runner-RAISA/spike_run_files/ci-deps/core-math.git \
  --commit 708e86ef634414431070336099bb5745f061dc21
```

Without network access, use any existing local core-math Git repository as the
source:

```bash
ci-tests/init-core-math-mirror.sh \
  --source /path/to/existing/core-math \
  --destination /nfs/share/ci-runner-RAISA/spike_run_files/ci-deps/core-math.git \
  --commit 708e86ef634414431070336099bb5745f061dc21
```

The script creates a complete independent bare mirror through a temporary
sibling and atomically publishes it.  It never overwrites or deletes an
existing destination, never changes the source repository, and rejects Git
alternates.  Re-running it against an existing mirror validates the repository
and the requested commit.

After creation, verify it as the Runner user:

```bash
git -C /nfs/share/ci-runner-RAISA/spike_run_files/ci-deps/core-math.git \
  cat-file -e 708e86ef634414431070336099bb5745f061dc21^{commit}
git -C /nfs/share/ci-runner-RAISA/spike_run_files/ci-deps/core-math.git \
  rev-parse --is-bare-repository
```

For a future Gitlink commit, create a new temporary destination with
`init-core-math-mirror.sh --commit <new-commit>`, validate it, then have the
administrator switch the deployed mirror path during a maintenance window.
Do not update or replace a mirror while CI jobs are using it.

## Configure open06

The open06 Build job sets `ZTT_CI_CORE_MATH_MIRROR` to the maintainer-owned
mirror.  Change that job-level environment value when deploying to another
site; no personal home-directory path is required.  The build script uses the
mirror only when this variable is set.  A normal developer build with the
variable unset continues to initialize `core-math` from the URL in
`.gitmodules`.  The CI override is a single-command `git -c` setting and
does not modify `.gitmodules` or global Git configuration.

`ZTT_CI_LOG_ROOT` selects the shared per-run storage.  If it is not supplied,
open06 uses:

```text
/nfs/share/ci-runner-RAISA/spike_run_files/ci-results
```

For another deployment, set `ZTT_CI_LOG_ROOT` in the Runner environment (or
change the open06 workflow default) to a shared directory visible to all
six regression Runners and the aggregate Runner.

Both the Build and Regression jobs isolate data below
`open06-ztt/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT`.  The Build job publishes one
Spike installation there; six Matrix jobs read that installation and write
their reports there.  Aggregate reads the same directory on a self-hosted
Runner.  Artifact upload/download is not part of the correctness path.

The Runner account needs read/execute access to the core-math mirror and
read/write/execute access to `ZTT_CI_LOG_ROOT`.  The mirror parent and mirror
contents should be writable only by CI maintainers.

## Pull requests and security boundary

Other developers do not need to prepare or update core-math for a PR.  The
maintainer-owned mirror must already contain the PR's exact Gitlink commit;
otherwise Build fails with an explicit error.

Open06 is a self-hosted Runner that executes checked-out PR code, including
code from external Fork PRs.  Enable this workflow only on isolated,
maintainer-approved runners with no sensitive credentials, and require the
normal maintainer approval policy before allowing untrusted PR code to run.
