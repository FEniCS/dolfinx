# FEniCSx release guide

## Prerequisites

Check out all of the FEniCSx components on the `release` branch.

Check that all CIs on `main` are running green.

Check that the `main` documentation looks reasonable at
https://docs.fenicsproject.org.

The release proceeds in a bottom up manner (Basix, UFL, FFCx, DOLFINx). pypa
packages cannot be deleted and should be made a number of days after the
creation of git tags so that errors can be fixed. GitHub releases can have their
version notes updated, and can be deleted and remade on new tags (not recommended).

The release process consists of the following steps:

1. Update version numbers and dependencies on the `release` branches.
2. Run integration tests, ensuring that the `release` branches work together.
3. Make git tags on the tip of `release`.
4. Organise production of release artifacts.
5. Update version numbers on `main`.
6. Make GitHub releases (not permanent) pypa releases (permanent!).

## Version bumping

At the current phase of development (<1.0) FEniCSx components are typically
bumped an entire minor version i.e. `0.+1.0`.

UFL still runs on the year-based release scheme.

### Basix version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git pull
       git checkout release
       git merge --no-commit origin/main
       git checkout --theirs origin/main . # files deleted on `main` must be manually `git add`ed
       git diff origin/main

2. Update version numbers in `pyproject.toml`, `python/pyproject.toml`,
   `CMakeLists.txt` and `cpp/CMakeLists.txt`.

4. In `pyproject.toml` update the `fenics-ufl` optional dependency version. On
   `main` this is often pointing at the git repo, it needs to be changed to a
   version bound e.g. `>=2024.1.0,<2024.2.0`.

5. Commit and push.

6. Check `git diff origin/main` for obvious errors.

### UFL version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git pull
       git checkout release
       git merge --no-commit main
       git checkout --theirs origin/main . # files deleted on `main` must be manually git `add`ed
       git diff main

2. Update the version number in `pyproject.toml`, e.g. `2022.2.0`.

3. Commit and push.

4. Check `git diff origin/main` for obvious errors.

### FFCx version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git pull
       git checkout release
       git merge --no-commit origin/main
       git checkout --theirs origin/main . # files deleted on `main` must be manually git `add`ed
       git diff origin/main

2. Update the version number in `pyproject.toml`, e.g. `0.5.0`.

3. Update the dependency versions for `fenics-basix` and `fenics-ufl` in
   `pyproject.toml`.

4. If necessary, update the version number in `cmake/CMakeLists.txt`, e.g.
   `0.5.0`.

5. Update the version number macros in `ffcx/codegeneration/ufcx.h`. Typically
   this should match the Python version number. Remember to change the
   `UFCX_VERSION_RELEASE` to `1`.

6. Commit and push.

7. Check `git diff origin/main` for obvious errors.

### DOLFINx

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git pull
       git checkout release
       git merge --no-commit origin/main
       git checkout --theirs origin/main . # files deleted on `main` must be manually git `add`ed
       git diff origin/main

2. In `cpp/CMakeLists.txt` change the version number e.g. `0.5.0`.

3. In `cpp/CMakeLists.txt` change the version number in the
   `find_package(ufcx)` and `find_package(UFCx)` calls.

4. In `python/pyproject.toml` update the version to e.g. `0.5.0` and
   update the dependency versions for `fenics-ffcx` and `fenics-ufl`.

5. In `CITATION.md` update the version number `version: 0.5.0` and the release
   date `date-released: 2022-03-14`.

6. In `.github/ISSUE_TEMPLATE/bug_report.yml` add a new option to the version
   numbers.

7. Commit and push.

8. Check `git diff origin/main` for obvious errors.

## Integration testing

Although lengthy, integration testing is highly effective at discovering issues
and mistakes before they reach tagged versions.

At each of the following links run the GitHub Action Workflow manually using
the `release` branch in all fields, including the . *Only proceed to tagging
once all tests pass.*

Basix with FFCx: https://github.com/FEniCS/basix/actions/workflows/ffcx-tests.yml

Basix with DOLFINx: https://github.com/FEniCS/basix/actions/workflows/dolfinx-tests.yml

UFL with FEniCSx: https://github.com/FEniCS/ufl/actions/workflows/fenicsx-tests.yml

FFCx with DOLFINx: https://github.com/FEniCS/ffcx/actions/workflows/dolfinx-tests.yml

Full stack: https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml


## Tagging

Make appropriate version tags in each repository. UFL does not use the `v` prefix.

    git tag v0.5.0
    git push --tags origin

## Artifacts

### Documentation

Documentation should be pushed automatically to `FEniCS/docs` on the creation
of tags. You will need to manually update the `README.md`.

### Docker containers

First create tagged development and test environment images, e.g. `v0.5.0`:

https://github.com/FEniCS/dolfinx/actions/workflows/docker-dev-test-env.yml

Then create tagged end-user images setting the base image as the tagged
development image:

https://github.com/FEniCS/dolfinx/actions/workflows/docker-end-user.yml

The tag prefix should be the same as the DOLFINx tag e.g. `v0.5.0`. Git refs
should be appropriate tags for each component.

Tagged Docker images will be pushed to Dockerhub and GitHub.

    docker run -ti dolfinx/dolfinx:v0.5.0

Use the *Docker update stable* tag workflow to update/link `:stable` to e.g.
`v0.5.0`.

https://github.com/FEniCS/dolfinx/actions/workflows/docker-update-stable.yml

### pypa

Wheels can be made using the following actions:

https://github.com/FEniCS/basix/actions/workflows/build-wheels.yml

https://github.com/FEniCS/ufl/actions/workflows/build-wheels.yml

https://github.com/FEniCS/ffcx/actions/workflows/build-wheels.yml

Both the workflow and the ref should be set to the appropriate tags for each
component.

It is recommended to first build without publishing, then to test pypa, then to
the real pypa. Publishing to pypa cannot be revoked.

The DOLFINx wheel builder is experimental and is not used in the release
process at this time.

### Mistakes

If something doesn't work, or other issues/bugs are identified during the
release process you can either:

1. Make changes on `main` via the usual PR workflow, then `git cherry-pick` or
   `git merge` the commit back onto `release`.

2. Manually make commits on the `release` branch.

If you want the change to be reflected on `main` long term 1. is preferred.

If a mistake is noticed soon after making a tag then you can delete the tag and
recreate it. It is also possible to recreate GitHub releases. After pypa
packages are pushed you must create .post0 tags or make minor version bumps, as
pypa is immutable.

### GitHub releases

Releases can be made at the following links using the appropriate tag. The
automatic release notes should be checked. The release notes can still be
edited after the release is finalised.

https://github.com/FEniCS/basix/releases/new

https://github.com/FEniCS/ufl/releases/new

https://github.com/FEniCS/ffcx/releases/new

https://github.com/FEniCS/dolfinx/releases/new

## Post-release

Check for any changes on `release` that should be cherry-picked back onto
`main` via a PR.

     git checkout main
     git diff release

Bump the version numbers on the `main` branch via a PR.

## Bug fix patches

Bug fix versions e.g. `v0.5.1` can be made by cherry picking commits off of
`main` and bumping the minor version number. Remember to run the DOLFINx
integration tests on a proposed set of tags as it is easy to make an error.

### Debian/Ubuntu

Contact Drew Parsons.

### Conda Forge

Conda Forge bots typically pickup new releases automatically. Can also contact
@minrk.

### Spack

Update the Spack recipe for the FEniCSx components on the fork
[FEniCS/spack](https://github.com/FEniCS/spack) using a branch e.g.
`updates/dolfinx-<version>`. Create a pull request to the Spack mainline
repository.
