# AGENTS.md

Style and conventions for working in the DOLFINx repository. This
complements [CONTRIBUTING.md](CONTRIBUTING.md), which covers the PR/AI
disclosure process.

## Repository layout

- `cpp/dolfinx/` — the C++ library, organized by sub-namespace
  (`common`, `mesh`, `fem`, `graph`, `la`, `io`, `nls`, `refinement`,
  `geometry`).
- `cpp/test/` — C++ unit tests (Catch2 3).
- `cpp/demo/` — C++ demo programs.
- `python/dolfinx/` — the Python package.
- `python/dolfinx/wrappers/` — nanobind C++/Python bindings.
- `python/demo/`, `python/test/` — Python demos and pytest suite.

## C++ style

- **Standard**: Modern C++20. Use concepts (`std::floating_point T`,
  `std::integral`, `std::ranges` etc.) to constrain templates rather than
  SFINAE. Don't use C-style casts. `const`-correctness is encouraged.
  Consider using `constexpr` and `consteval`.
- **Avoid overusing the `auto` keyword`**: The `auto` keyword should not be
  used on simple-to-reason-about types, e.g. `std::int32_t` and
  `std::vector<T>` as it reduces code readability.
- **Formatting**: enforced by `.clang-format` (LLVM-derived, 2-space
  indent, 80-column limit, Allman braces). Always run `clang-format -i`
  on touched `.cpp`/`.h` files before considering a change done; CI
  runs `clang-format --dry-run --Werror` and will fail otherwise.
- **File header**: every source file starts with

  ```cpp
  // Copyright (C) <years> <author(s)>
  //
  // This file is part of DOLFINx (https://www.fenicsproject.org)
  //
  // SPDX-License-Identifier:    LGPL-3.0-or-later
  ```

  Add your name/year to the existing copyright line when you make a
  substantive change; don't replace existing authors.
- **Headers**: use `#pragma once`, not include guards. Includes are
  sorted (`clang-format`'s `SortIncludes`) — the paired header first
  (in `.cpp` files), then project (`<dolfinx/...>`) headers, then
  standard-library headers, alphabetically within each group.
- **Include What You Use (IWYU)**: follow IWYU best practice down to
  including 'trivial' headers such as `<cstdint>` and `<iterators>`
  directly, rather than relying on transitive includes -- IWYU is
  not currently enforced systematically via testing, so inspect the
  modified files touched and make suggestions.
- **Namespaces**: library code lives in `dolfinx::<module>` (e.g.
  `dolfinx::io::hdf5`). In `.cpp` files, prefer `using namespace
  dolfinx;` at the top and qualify definitions with the remaining
  namespace path, e.g. `io::hdf5::open_file(...)` rather than fully
  qualifying every symbol.
- **Naming**: `snake_case` for functions and variables, `PascalCase`
  for types/classes, private/protected data members prefixed with an
  underscore (`_dofmap`, `_index_map_bs`). Free functions and class
  methods both use `snake_case`.
- **Parameters**: pass read-only strings as `std::string_view`, not
  `const std::string&`. Use `std::span` for contiguous read-only array
  views, and `mdspan` for read-only multi-dimensional views. Reserve
  `std::vector`/`std::string` for parameters that are stored, mutated
  in place, or are container element types.
- **Documentation**: Doxygen-style `///` comments above declarations.
  Use `@brief` for anything non-trivial, `@param[in]`/`@param[in,out]`
  per parameter, `@return`, `@note`. Keep `@param` names in sync with
  the actual parameter names — this is checked manually in review, not
  by tooling, so a rename must be applied to the declaration, the
  definition, and any doc comment together.
- **Comments**: LLM-generated comments tend to be rather verbose;
  after the first comment draft, compress comments to their essence
  using concise technical language.
- **Errors and invariants**: throw `std::runtime_error` with a
  descriptive message for user-facing/API-boundary errors when the
  check is O(1). For more expensive single-line checks, use `assert`.
  For more expensive multi-line checks, throw an exception but guard
  the check so it only runs in debug builds (e.g. `#ifndef NDEBUG`).
  `assert` itself is for internal invariants that indicate a library
  bug, not bad user input. Do not add exceptions inside hot loops.
  Prefer `spdlog::debug`/`info`/`warn` for logging
  over `std::cout`/`std::cerr`.
- **Move/copy semantics**: Moving is preferred over copying, unless
  the object is very lightweight. Many DOLFINx classes disable move
  constructors. `std::move` is used systematically on incoming
  `std::shared_ptr` to avoid unnecessary copies of `std::shared_ptr`
  and also to avoid copies when returning with a `std::pair{}`.
- **String formatting**: use `std::format` (`<format>`) to build
  formatted/error strings rather than `printf`-style, `std::ostringstream`
  concatenation, or the `fmt` library.
- **Function pointers over lambdas**: when a free function's signature
  already matches a callback/`std::function` parameter exactly, pass
  the function directly (e.g. `graph::reorder_rcm`) rather than
  wrapping it in a trivial forwarding lambda.
- **Algorithms**: prefer `<algorithm>`/`<ranges>` (`std::ranges::...`)
  over hand-written loops where it doesn't hurt clarity or
  performance; flattened row-major storage is the default convention
  for multi-dimensional data passed as flat buffers.
- **Windows**: Windows is continuously tested on GitHub with the
  most important missing feature being the lack of C99 `_Complex`
  support denoted by existence of `DOLFINX_NO_STDC_COMPLEX_KERNELS`
  macro variable.
- **PETSc support is optional**: If possible, tests should be
  written without needing PETSc functionality and PETSc-related
  functionality in the main library should be isolated. The test
  suite include a simple conjugate-gradient solver for small
  problems, for example. 

## Python style

- **Formatting/linting**: `ruff` (both `ruff check` and `ruff format
  --check`), configured in `python/pyproject.toml`. Line length 100,
  4-space indent. Rule set includes pydocstyle (`D`, Google
  convention), pycodestyle, pyflakes, isort, pyupgrade,
  flake8-import-conventions, and NumPy-specific rules.
- **Import order** (via ruff's isort): future → standard-library →
  `mpi4py`/`petsc4py` (own `mpi` section) → third-party → first-party
  (`basix`, `dolfinx`, `ffcx`, `ufl`) → local-folder.
- **Docstrings**: Google style (`Args:`, `Returns:`, etc.), module and
  public API documented; test/demo files are exempt from some
  pydocstyle rules (see `per-file-ignores`).
- **Type hints**: required on the public API; checked with `mypy`
  (`python/pyproject.toml` `[tool.mypy]` config, run over `dolfinx`,
  `test`, and `demo`). PETSc-related type checking is disabled on a
  per-line basis until upstream petsc4py type work is finished.
- **File header**: same SPDX/copyright block as C++, adapted to `#`
  comments, followed by a module docstring.

## nanobind wrapper style (`python/dolfinx/wrappers/`)

- One file per C++ module (`fem.cpp`, `mesh.cpp`, `la.cpp`, ...),
  wired together from `dolfinx.cpp`.
- Bind free functions with `m.def(...)`, giving named arguments via
  `nb::arg("name")` matching the C++ parameter name. Check the
  ordering matches as it's easy to make a mistake.
- Wrap C++ types with `nb::class_<T>(m, "Name", "docstring")`, chaining
  `.def(...)`, `.def_prop_ro(...)`, `.def_ro(...)`.
- These files are still C++: `clang-format` applies to them too (CI
  checks `python/dolfinx/wrappers` separately).
- Do not use default argument values.
- The nanobind wrappers are further wrapped into a pure-Python
  interface which contains the user facing API. Users and developers
  are discouraged from using `dolfinx.cpp` directly.

## CMake style

- Formatted with `gersemi` (2-space indent, see `.gersemirc`); CI runs
  `gersemi --check .`.

## Demos

- C++ demos are written with Markdown comments for subsequent
  postprocessing with jupytext and sphinx.
- Python demos are written with light format and Markdown for
  subsequent postprocessing with jupytext and sphinx.
- Demo text should be checked for clarity, brevity, mathematical
  correctness (e.g. missing definitions) and misalignment with the
  presented solver code.

## Testing

- **Developer vs Release build mode**: Both C++
  `cmake ... -DCMAKE_BUILD_TYPE=Developer` and Python parts
  `pip ... -Ccmake.build-type=Developer` must be built in Developer mode
  which enables hardened debugging/correctness checks. Performance
  profiling must be done on a build built with `Release` mode.
- **C++**: Catch2 3, in `cpp/test/`. FFCx-generated forms are compiled
  as part of the test build (see `cpp/test/CMakeLists.txt`).
- **Python**: `pytest`, in `python/test/`. Use `mpi4py.MPI` fixtures
  for parallel-aware tests where relevant.
- Run the relevant formatter/linter and the affected test suite before
  calling a change done — don't rely on CI to catch formatting.

## Verifying changes locally

- C++: configure once with CMake+Ninja against an installed Basix,
  then `ninja` in the build directory is enough to catch real
  compilation errors (the project builds with `-Werror`). Don't trust
  editor/clangd diagnostics alone — they're frequently noise from
  incomplete include paths, not real errors.
- Python: To build the Python interface the C++ interface must be
  `ninja install`ed.
- Prefer `clang-format --dry-run --Werror` / `ruff check` / `ruff
  format --check` / `gersemi --check` locally to match exactly what CI
  enforces, rather than eyeballing style.
