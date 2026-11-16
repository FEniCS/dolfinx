# Consolidated FEniCSx EESSI release build

[../workflows/eessi.yml](../workflows/eessi.yml) is the single EESSI workflow
for the Basix, UFL, FFCx and DOLFINx release stack. Their independent EESSI
workflows, dependency bundles and Bash build/test scripts have been removed.
Other CI workflows in those repositories are unchanged.

## Dependency resolution

[fenicsx.yml](fenicsx.yml) lists the four Python package recipes as roots.
One `eb --robot --easystack=...` invocation resolves their complete dependency
graph, orders the builds, and reuses shared dependencies. Jobs are split only
by runner architecture, so the components share an installation on each runner.
Build commands, tests, installation and sanity checks remain in EasyBuild.

Only the DOLFINx C++ and Python easyconfigs are stored locally. The workflow
checks out `gkaf89/FEniCS-EESSI-demo` at commit
`d05588835643a1782777fce313299ec42b4e3575` for the other recipes and Basix
smoke test. Robot searches the local directory first, the demo checkout second,
then the default EasyBuild recipe directories. The trailing colon preserves
those default search paths.

EasyBuild runs from the demo directory because its Basix recipe references
the smoke test relative to the working directory. The easystack and local
recipe directory are passed by absolute path. The job needs no shell build
script, virtual environment, manual dependency loop, pip/CMake build stage,
or shell-based logging wrapper.

## Sources and tests

This builds release archives with their recipe checksums:

| Component | Release | Recipe checks |
| --- | --- | --- |
| UFL | 2026.1.0 | Configured pytest suite |
| Basix | 0.11.0 | C++ installation checks and Python smoke test |
| FFCx | 0.11.0 | Poisson example, selected pytest tests and version check |
| DOLFINx | 0.11.0.post0 | C++ installation checks and one-rank Python import |

This is release packaging validation, not CI of the development source
checkouts. In particular, it does not run the full DOLFINx serial/two-rank
unit suites or the development FFCx Numba demos. EasyBuild may reuse already
available modules without rerunning their tests. The existing
`fenicsx-refs.env` files are not used.

The workflow runs on DOLFINx pushes to `main`, tags, pull requests targeting
`main`, merge queues, and manual dispatch. Changes to the other repositories
do not trigger it automatically. Manual dispatch requires this workflow to
be available on the DOLFINx repository's default branch.

## Runners and reports

The default runners remain `ubuntu-24.04` and `ubuntu-24.04-arm`, initialized
with the EESSI action for EESSI 2025.06. The `EESSI_RUNNERS` repository variable
can supply a JSON array of alternative single runner labels. A label does not
install EESSI or grant runner access. This setup still requires Ubuntu/CVMFS
support for the setup action; a Rocky Linux HPC runner needs separate adaptation
to initialize its preinstalled EESSI environment.

Installation and build paths are isolated by workflow run ID and attempt under
`runner.temp`. Directory configuration is passed as step environment variables.
EasyBuild writes its normal log files; temporary logs and logs installed with
packages are uploaded for seven days even after failures. There are no custom
JUnit reports or standalone wheel uploads, and no publishing to PyPI/CVMFS.

## Local execution

From the DOLFINx repository root, after initializing EESSI 2025.06:

```bash
local_recipes="$PWD/.github/eessi"
git clone https://github.com/gkaf89/FEniCS-EESSI-demo.git .eessi-dependencies/recipes
git -C .eessi-dependencies/recipes checkout d05588835643a1782777fce313299ec42b4e3575
cd .eessi-dependencies/recipes
module load EESSI-extend/2025.06-easybuild
unset EASYBUILD_DEBUG
eb --robot --robot-paths="$local_recipes:$PWD:" \
  --easystack="$local_recipes/fenicsx.yml"
```

Add `--dry-run` to inspect dependency resolution without building packages.
Runtime validation requires EESSI; YAML and Bash checks alone cannot establish
that these release recipes build successfully on the intended runner.
