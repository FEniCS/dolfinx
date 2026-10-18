# FEniCSx release guide

## Overview

The release proceeds bottom up: UFL, Basix, FFCx, DOLFINx. The examples below
release DOLFINx, Basix and FFCx `0.12.0` and UFL `2026.3.0`.

1. Update version numbers and dependencies on the `release` branches.
2. Run integration tests, ensuring that the `release` branches work together.
3. Make git tags on the tip of `release` (not permanent).
4. Produce the documentation and Docker images.
5. Update version numbers on `main`.
6. Make GitHub releases (not permanent).
7. Make PyPI releases (permanent!), a few days after tagging.
8. Announce the release.
9. Update downstream packages.

Git tags and GitHub releases can be deleted and remade. PyPI releases cannot.
Waiting a few days before publishing to PyPI leaves time to find errors.

## Prerequisites

Check that all CIs on `main` are running green.

Check that the `main` documentation looks reasonable at
https://docs.fenicsproject.org.

Clone UFL, Basix, FFCx and DOLFINx.

In DOLFINx, merge a PR to `main` that:

- removes `(draft)` from the `python/doc/source/release_notes.md` heading,
  fills in the number of merged pull requests and contributors, and points the
  full diff link at the new tag, e.g. `compare/v0.11.0...v0.12.0`,
- updates `version` and `date-released` in `CITATION.cff`,
- adds the new version to the options in
  `.github/ISSUE_TEMPLATE/bug_report.yml`.

These then reach `release` with the merge below.

## Running workflows

Most steps run GitHub Actions workflows with `gh workflow run`, which prints
the run URL. Wait for a run with

    gh run watch --exit-status -R FEniCS/<repo> <run-id>

before starting a step that depends on it.

## Version number bumping

At the current phase of development (<1.0) FEniCSx components are typically
bumped an entire minor version i.e. `0.+1.0`.

UFL uses year-based numbering and releases are made by both FEniCS and
Firedrake teams. The version number should be bumped as follows:

1. If the calendar year has changed, bump the year. Restart the minor and patch
   versions at `1` and `0` respectively.
2. Otherwise, always bump the minor version `0.+1.0`. A minor series belongs
   to either FEniCS or Firedrake teams.
3. If making a patch release, bump the patch version `0.0.+1` -- check
   compatibility through CI runs before releasing.

Release dependency ranges are capped at the next minor version, e.g.
`fenics-basix>=0.12.0,<0.13.0` and `fenics-ufl>=2026.3.0,<2026.4.0`.

CMake versions cannot carry `.dev0`. On `main` they have a fourth component
instead, e.g. `0.12.0.0` for Python `0.12.0.dev0`, which the release drops: both
become `0.12.0`. `UFCX_VERSION` in `ufcx.h` follows the same rule through
`UFCX_VERSION_IS_RELEASE`.

### Merging `main` into `release`

Each component starts by merging `origin/main` into `release`, taking `main`'s
version of every file:

    git pull
    git checkout release
    git merge --no-commit origin/main
    git read-tree -u --reset origin/main
    git diff origin/main

`git read-tree` also discards release-only changes that merge cleanly, e.g. a
previous `UFCX_VERSION_IS_RELEASE true`, and removes files deleted on `main`.
`git checkout --theirs .` does neither. After the merge `git diff origin/main`
should be empty.

Then point the refs in `.github/workflows/fenicsx-refs.env` at the `release`
branches, make the component's changes below, check `git diff origin/main` for
obvious errors, and commit and push.

### ABI checks

A patch release must not change any interface below. A minor release may, but
`ufcx.h` changes more slowly than FFCx and is moving towards its own version,
bumped only when its ABI changes.

| Component | Interface | Check |
| --- | --- | --- |
| UFL | Python API | griffe |
| Basix | `libbasix` ABI, Python API, element definitions | `abidiff`, griffe, element comparison |
| FFCx | `ufcx.h` ABI, Python API | `abidiff`, griffe |
| DOLFINx | `libdolfinx` ABI and headers, Python API, file formats | `abidiff`, old demos against the new library, griffe |

#### C and C++ ABI

Use libabigail's `abidiff`, e.g. in a Fedora container. `ufcx.h` is
header-only, so compile it into a small library that keeps all its types in
the debug information:

    mkdir old new
    git show v0.11.0:ffcx/codegeneration/ufcx.h > old/ufcx.h
    cp ffcx/codegeneration/ufcx.h new/ufcx.h
    podman run --rm -v "$PWD":/w:Z -w /w docker.io/library/fedora bash -c '
      dnf -y -q install gcc libabigail >/dev/null 2>&1 &&
      for d in old new; do
        echo "#include \"ufcx.h\"" > $d/abi.c &&
        gcc -shared -fPIC -g -fno-eliminate-unused-debug-types -I$d \
          -o $d/libufcx.so $d/abi.c
      done &&
      abidiff --non-reachable-types old/libufcx.so new/libufcx.so'

Exit status `0` means no ABI change. Otherwise `abidiff` lists the changed
types, and `12` means an incompatible change. The SHA1 signature of `ufcx.h` is
no substitute, since it changes with the version macros on every release.

For Basix and DOLFINx, build the previous and new tags with
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` and compare the installed libraries, e.g.
`abidiff --headers-dir1 old/include --headers-dir2 new/include
old/lib/libbasix.so new/lib/libbasix.so`. The headers options limit the report
to the public API. Most of DOLFINx is templates in its headers, which
`abidiff` does not see, so also build the previous release's C++ demos against
the old install and run them against the new `libdolfinx` without rebuilding.

#### Python API

From each repository at the new tag, compare with the previous release using
[griffe](https://mkdocstrings.github.io/griffe/):

    uvx griffe check ufl --search . --against 2026.3.0
    uvx griffe check ffcx --search . --against v0.12.0
    uvx griffe check basix --search python --against v0.12.0
    uvx griffe check dolfinx --search python --against v0.12.0

griffe lists removed objects and changed signatures, and exits non-zero if it
finds any. Additions are not reported. It reads Python source and stub files,
not the compiled modules `_basixcpp` and `dolfinx.cpp`.

#### Basix elements

FFCx bakes Basix's tabulated values into generated kernels, and its JIT cache
key does not include the Basix version. A Basix patch release that changes an
element would leave cached kernels silently using the old values. Compare every
element's hash and tabulated values, since the hash alone can miss changes:

    cat > element_hashes.py <<'EOF'
    import hashlib
    import itertools

    import numpy as np

    import basix

    for family, cell, degree in itertools.product(
        basix.ElementFamily, basix.CellType, range(5)
    ):
        try:
            e = basix.create_element(family, cell, degree)
            pts, _ = basix.make_quadrature(cell, 4)
        except RuntimeError:
            continue
        t = np.round(e.tabulate(1, pts), 10)
        sha = hashlib.sha1(t.tobytes()).hexdigest()[:12]
        print(family.name, cell.name, degree, e.hash(), sha)
    EOF
    python3 -m venv old new
    old/bin/pip install -q fenics-basix==0.12.0 numpy
    new/bin/pip install -q numpy \
        "fenics-basix @ git+https://github.com/FEniCS/basix.git@v0.12.1"
    old/bin/python element_hashes.py > old.txt
    new/bin/python element_hashes.py > new.txt
    diff old.txt new.txt

Any output means an element changed.

#### Build environment

Within a release series, build the wheels with the same nanobind version,
since `dolfinx.cpp` takes Basix's nanobind objects directly. Do not change the
MPI implementation, the manylinux image or the `cp312-abi3` tag either.

### UFL version bump

1. Merge `main` into `release`.

2. Update the version number in `pyproject.toml`, e.g. `2026.3.0`.

### Basix version bump

1. Merge `main` into `release`.

2. Update the version numbers in `pyproject.toml`, `python/pyproject.toml`,
   `CMakeLists.txt` and `cpp/CMakeLists.txt`, e.g. `0.12.0`.

3. In `pyproject.toml` and `python/pyproject.toml` replace the `fenics-ufl`
   lower bound of the `optional` extra with a range.

### FFCx version bump

1. Merge `main` into `release`.

2. Update the version number in `pyproject.toml`, e.g. `0.12.0`.

3. In `pyproject.toml` replace the `fenics-basix` and `fenics-ufl` lower bounds
   with ranges.

4. Check the `ufcx.h` ABI against the previous release, see
   [ABI checks](#abi-checks). The ufcx version is in `cmake/CMakeLists.txt` and
   the `ufcx.h` macros. If the ABI changed, bump its minor version, e.g. from
   `0.11.0` to `0.12.0`. Otherwise keep the previous release's, e.g. `0.11.0`.
   Set `UFCX_VERSION_IS_RELEASE` to `true`.

### DOLFINx version bump

1. Merge `main` into `release`.

2. Update the version number in `cpp/CMakeLists.txt`, e.g. `0.12.0`.

3. Check the `find_package(Basix)` call in `cpp/CMakeLists.txt` requires the
   release's minor version, e.g. `0.12`. Check `find_package(ufcx)` and
   `find_package(UFCxPython)` require the ufcx version chosen in the FFCx bump,
   e.g. `0.11` and `0.11...<0.12`.

4. In `python/pyproject.toml` update the version, e.g. `0.12.0`, and replace
   the `fenics-basix`, `fenics-ffcx` and `fenics-ufl` lower bounds with ranges.

## Integration testing

Although lengthy, integration testing is highly effective at discovering issues
and mistakes before they reach tagged versions.

Pushes to `release` do not run CI. Once every `release` branch is updated, run
the integration tests on `release` with:

    gh workflow run fenicsx-tests.yml -R FEniCS/ufl --ref release
    gh workflow run ffcx.yml -R FEniCS/basix --ref release
    gh workflow run dolfinx.yml -R FEniCS/basix --ref release
    gh workflow run dolfinx.yml -R FEniCS/ffcx --ref release
    gh workflow run ccpp.yml -R FEniCS/dolfinx --ref release

Then list the latest run of each:

    for w in ufl:fenicsx-tests.yml basix:ffcx.yml basix:dolfinx.yml \
        ffcx:dolfinx.yml dolfinx:ccpp.yml; do
      gh run list -R FEniCS/${w%%:*} -w ${w#*:} -b release -L 1
    done

## Tagging

Tag each repository in order: UFL, Basix, FFCx, DOLFINx. UFL tags have no `v`
prefix, e.g. `2026.3.0`. The others do, e.g. `v0.12.0`.

1. For Basix, FFCx and DOLFINx, point the refs of the components below it in
   `.github/workflows/fenicsx-refs.env` on `release` at their new tags, e.g.
   `ufl_ref=2026.3.0` and `basix_ref=v0.12.0`, then commit and push. The Basix
   wheel tests install UFL from `ufl_ref`. If it is left at `main` they can pick
   up a UFL outside the `fenics-ufl` range.

2. Tag the tip of `release` and push the tag:

       git tag v0.12.0
       git push origin v0.12.0

Tag pushes run each component's full CI and publish its documentation. Wait for
the UFL, Basix and FFCx documentation to be published before tagging DOLFINx.
The DOLFINx tag CI tests the tagged stack, so check it passes before producing
artifacts.

## Artifacts

### Documentation

Documentation is pushed automatically to
[FEniCS/docs](https://github.com/FEniCS/docs) when tags are created. Update
its `README.md` by hand.

### Docker containers

Image tags match the DOLFINx tag. Run each workflow from the DOLFINx tag so the
Dockerfiles match the release.

First create the tagged development and test environment images:

    gh workflow run docker-dev-test-env.yml -R FEniCS/dolfinx --ref v0.12.0 \
        -f tag_prefix=v0.12.0

Then create the tagged end-user images on top of the development image:

    gh workflow run docker-end-user.yml -R FEniCS/dolfinx --ref v0.12.0 \
        -f tag_prefix=v0.12.0 \
        -f baseimage=docker.io/dolfinx/dev-env:v0.12.0 \
        -f dolfinx_ref=v0.12.0 -f basix_ref=v0.12.0 \
        -f ffcx_ref=v0.12.0 -f ufl_ref=2026.3.0

The images are pushed to Docker Hub and the GitHub Container Registry:

    docker run -ti dolfinx/dolfinx:v0.12.0

Finally link `:stable` to the new images:

    gh workflow run docker-update-stable.yml -R FEniCS/dolfinx --ref v0.12.0 \
        -f end-user-tag=v0.12.0 -f dev-env-tag=v0.12.0

## Post-release

Check for any changes on `release` that should be cherry-picked back onto
`main` via a PR. Do not cherry-pick version numbers or
`.github/workflows/fenicsx-refs.env`, which stays at `main` on `main`.

    git checkout main
    git diff release
    git log
    git cherry-pick 914ae4

Bump the version numbers on the `main` branches of UFL, Basix, FFCx and
DOLFINx to the next development version via PRs, in the files listed above. On
`main`:

- Python versions are e.g. `0.13.0.dev0` and CMake versions e.g. `0.13.0.0`.
- Dependencies on other components are lower bounds only, e.g.
  `fenics-basix>=0.13.0.dev0` and `fenics-ufl>=2026.4.0.dev0`.
- In `ufcx.h` set `UFCX_VERSION_IS_RELEASE` to `false`, and give the ufcx
  version in `cmake/CMakeLists.txt` its fourth component, e.g. `0.11.0.0`. Do
  not bump the ufcx version itself. The first PR that changes the `ufcx.h` ABI
  bumps its minor version instead.
- The DOLFINx `find_package(Basix)` call requires the next minor version, e.g.
  `0.13`. Basix only satisfies requests for its own minor version, so once Basix
  `main` is bumped DOLFINx `main` no longer configures. Make the Basix and
  DOLFINx bumps paired PRs, each pointing `fenicsx-refs.env` at the other's
  branch, and merge them together. An FFCx PR that bumps the ufcx version needs
  a paired DOLFINx PR updating `find_package(ufcx)` and
  `find_package(UFCxPython)` in the same way.
- `python/doc/source/release_notes.md` gets a new section, e.g.
  `## v0.13.0 (draft)`.

## GitHub releases

Create a release for each tag with automatically generated notes:

    gh release create 2026.3.0 -R FEniCS/ufl --verify-tag --generate-notes \
        --notes-start-tag 2026.2.0
    gh release create v0.12.0 -R FEniCS/basix --verify-tag --generate-notes \
        --notes-start-tag v0.11.0
    gh release create v0.12.0 -R FEniCS/ffcx --verify-tag --generate-notes \
        --notes-start-tag v0.11.0
    gh release create v0.12.0 -R FEniCS/dolfinx --verify-tag --generate-notes \
        --notes-start-tag v0.11.0

Always pass `--notes-start-tag`. Release tags are on `release`, not `main`, so
GitHub's automatic choice can be far older than the previous release. For
Basix, FFCx and DOLFINx start from the previous minor's `.0` tag, not a `.postN`
tag. For UFL start from the previous UFL tag, even when Firedrake made it.

Check the "Full Changelog" link in the generated notes. The notes can still be
edited after the release, e.g. with `gh release edit`.

## PyPI

Publish to PyPI a few days after tagging. Publish in the order UFL, Basix,
FFCx, since each package's dependency ranges only resolve once the packages
below it are on PyPI. Run each workflow from its tag.

First build and test the wheels without publishing:

    gh workflow run build-wheels.yml -R FEniCS/ufl --ref 2026.3.0 \
        -f ufl_ref=2026.3.0
    gh workflow run build-wheels.yml -R FEniCS/basix --ref v0.12.0 \
        -f basix_ref=v0.12.0
    gh workflow run build-wheels.yml -R FEniCS/ffcx --ref v0.12.0 \
        -f ffcx_ref=v0.12.0

Then publish to Test PyPI, one component at a time:

    gh workflow run build-wheels.yml -R FEniCS/ufl --ref 2026.3.0 \
        -f ufl_ref=2026.3.0 -f test_pypi_publish=true
    gh workflow run build-wheels.yml -R FEniCS/basix --ref v0.12.0 \
        -f basix_ref=v0.12.0 -f test_pypi_publish=true
    gh workflow run build-wheels.yml -R FEniCS/ffcx --ref v0.12.0 \
        -f ffcx_ref=v0.12.0 -f test_pypi_publish=true

Check the Test PyPI packages install together in a clean environment:

    python -m venv test-env
    test-env/bin/pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        fenics-ufl==2026.3.0 fenics-basix==0.12.0 fenics-ffcx==0.12.0

Finally publish to PyPI, one component at a time:

    gh workflow run build-wheels.yml -R FEniCS/ufl --ref 2026.3.0 \
        -f ufl_ref=2026.3.0 -f pypi_publish=true
    gh workflow run build-wheels.yml -R FEniCS/basix --ref v0.12.0 \
        -f basix_ref=v0.12.0 -f pypi_publish=true
    gh workflow run build-wheels.yml -R FEniCS/ffcx --ref v0.12.0 \
        -f ffcx_ref=v0.12.0 -f pypi_publish=true

The DOLFINx wheel builder is experimental and is not used in the release
process at this time.

## Announcements

Once the packages are on PyPI, announce the release:

- Website: open a PR to [FEniCS/web](https://github.com/FEniCS/web) updating
  the version in `_config.yml` and the documentation links in
  `documentation/index.md`.
- Slack: post in the FEniCS Slack, linking the GitHub releases.
- LinkedIn: post on the FEniCS Project page.

## Downstream packages

### Debian/Ubuntu

Contact Drew Parsons.

### conda-forge

Conda-forge bots typically pick up new releases automatically. Can also
contact @minrk.

### Spack

The FEniCS overlay [FEniCS/spack-fenics](https://github.com/FEniCS/spack-fenics)
should differ from upstream
[spack/spack-packages](https://github.com/spack/spack-packages) `develop` only
by fixes not yet merged upstream. An agentic LLM can do all of the steps.

1. Diff each overlay package against its `builtin` copy on upstream `develop`,
   using both histories to see which side changed each hunk.

2. In a spack-fenics PR, take upstream's newer changes and cosmetic choices,
   restore any fix either side lost, keep overlay fixes, and add the new
   versions. The CI must pass before and after.

3. Upstream each remaining fix and new version from a personal fork of
   spack/spack-packages, one PR per package, e.g. branch
   `<username>/fenics-dolfinx-0.12.0`. Link the passing spack-fenics CI. If PRs
   are slow to merge ask in Slack Spack `#pull-requests`.

4. Once upstream PRs merge, drop the matching changes from the overlay.

### EasyBuild/EESSI

Contact Jack Hale.

## Mistakes

If something doesn't work, or other issues/bugs are identified during the
release process you can either:

1. Make changes on `main` via the usual PR workflow, then `git cherry-pick` or
   `git merge` the commit back onto `release`.

2. Manually make commits on the `release` branch.

If you want the same change to be reflected on `main` option 1. is preferred.

A tag can be deleted and recreated on the same working day, and a GitHub
release can be recreated. Once packages are on PyPI, fix mistakes with a patch
release, or with a `.post0` release, e.g. version and tag `v0.12.0.post0`, if
only packaging or documentation changed.

## Bug fix patches

Bug fix versions e.g. `v0.12.1` are made on `release` without merging `main`:

1. Cherry-pick the fixes from `main` onto `release`.
2. Bump the patch version in the files listed above. Keep the dependency
   ranges, and point `fenicsx-refs.env` at the tags the patch release is meant
   to work with. Check against the previous tag in the series that no
   interface in [ABI checks](#abi-checks) changed.
3. Run the integration tests, then follow the steps from Tagging onwards. Use
   the previous tag in the series as `--notes-start-tag`, e.g. `v0.12.0`, and
   only publish the components that changed.
