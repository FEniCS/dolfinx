# DOLFINx C++ demos

Each subdirectory is a self-contained demo that builds against an installed
DOLFINx. [FFCx](https://github.com/FEniCS/ffcx) is required to compile the UFL
forms (`.py` files) that accompany most demos.

## Building

To build all demos:

    cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer -B build -S .
    cmake --build build

To build a single demo, use its directory as the source, e.g. `-S poisson`, and
run the resulting executable directly.

Configuring this directory additionally registers tests for each demo on 1, 2
and 3 MPI processes:

    ctest --test-dir build          # all demos
    ctest --test-dir build -R demo_poisson_np_3

The demos can also be built as part of DOLFINx by configuring `cpp/` with
`-DDOLFINX_BUILD_DEMOS=ON`. The `demos` target then builds them all, and their
tests are added to the `test` target alongside the unit tests.

## Documentation

The demo sources are commented in Markdown and rendered into the DOLFINx
documentation with jupytext and Sphinx, so keep the prose comments in
`main.cpp` in step with the code.
