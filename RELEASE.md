# FEniCSx release guide

## Prerequisites

Check out all of the FEniCSx components on the `release` branch.

Check that all CIs on `main` are running green.

The release proceeds in a bottom up manner (Basix, UFL, FFCx, DOLFINx).
GitHub Releases and pypa packages cannot be deleted and should be made a number
of days after the creation of git tags so that errors can be fixed.

The process is consists of the following steps:

1. Update version numbers and dependencies on the `release` branches.
2. Run integration tests, ensuring that the `release` branches work together.
3. Make git tags on the tip of `release`.
4. Organise production of release artifacts.
5. Update version numbers on `main`.
6. Make GitHub and pypa releases (permanent!).

## Version bumping

At the current phase of development (<1.0) FEniCSx components are typically
bumped an entire minor version i.e. `0.+1.0`.

UFL still runs on the year-based release scheme.

### Basix version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git merge -Xtheirs main

2. Update version numbers, e.g.

       python3 update_versions.py -v 0.5.0

3. Inspect automatic version updates.

       git diff

4. Commit and push.

5. Check `git diff main` for obvious errors.

### UFL version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git merge -Xtheirs main

2. Update the version number in `setup.cfg`, e.g. `2022.2.0`.

3. Commit and push.

4. Check `git diff main` for obvious errors.

### FFCx version bump

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git merge -Xtheirs main

2. Update the version number in `setup.cfg`, e.g. `0.5.0`.

3. Update the dependency versions for `fenics-basix` and `fenics-ufl` in `setup.cfg`.

4. Update the version number macros in `ffcx/code_generation/ufcx.h`. Typically this
   should match the Python version number. Remember to change the
   `UFCX_VERSION_RELEASE` to `1`.

5. Commit and push.

6. Check `git diff main` for obvious errors.

### DOLFINx

1. Merge `main` into `release` resolving all conflicts in favour of `main`.

       git merge -Xtheirs main

2. In `cpp/CMakeLists.txt` change the version number near the top of the file,
   e.g. `0.5.0`.

3. In `cpp/CMakeLists.txt` check the `find_package(ufcx)` and
   `find_package(UFCx)` calls. If the DOLFINx and UFCx versions match then
   there is no need to change anything here. However, if they don't match, you
   need to manually specify the appropriate UFCx version.

4. In `python/setup.cfg` change the `VERSION` variable to e.g. `0.5.0` and
   update the depedency versions for `fenics-ffcx` and `fenics-ufl`.

5. Commit and push.

6. Check `git diff main` for obvious errors.

## Integration testing

Although lengthy, integration testing is highly effective at discovering issues
and mistakes before they reach tagged versions.

At each of the following links run the GitHub Action Workflow manually using
the `develop` branch in all fields. *Only proceed to tagging once all tests pass.*

### Basix integration

Basix with FFCx: https://github.com/FEniCS/basix/actions/workflows/ffcx-tests.yml

Basix with DOLFINx: https://github.com/FEniCS/basix/actions/workflows/dolfin-tests.yml

UFL with FEniCSx (TODO): https://github.com/FEniCS/ufl/actions/workflows/fenicsx-tests.yml

FFCx with DOLFINx: https://github.com/FEniCS/ffcx/actions/workflows/dolfin-tests.yml

Full stack: https://github.com/FEniCS/dolfinx/actions/workflows/ccpp.yml

## Tagging


## Artifacts

### Documentation

### pypa 

### Docker containers

### GitHub releases

## Post-release

1. Did you make any changes on release that should be ported back onto main?

   git checkout main
   git diff release

### Ubuntu

### conda 
