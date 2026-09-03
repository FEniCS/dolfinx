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
  `std::integral`, `std::ranges` etc.) to constrain templates rather
  than SFINAE. Don't use C-style casts. `const`-correctness is
  encouraged. Use `constexpr` and `consteval` where possible.
- **Avoid overusing the `auto` keyword**: The `auto` keyword should not be
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
  not currently enforced systematically via testing, so check touched
  files for missed opportunities and suggest fixes.
- **Namespaces**: library code lives in `dolfinx::<module>` (e.g.
  `dolfinx::io::hdf5`). In `.cpp` files, prefer `using namespace
  dolfinx;` at the top and qualify definitions with the remaining
  namespace path, e.g. `io::hdf5::open_file(...)` rather than fully
  qualifying every symbol.
- **Naming**: `snake_case` for functions and variables, `PascalCase`
  for types/classes, private/protected data members prefixed with an
  underscore (`_dofmap`, `_index_map_bs`).
- **Integer types**: `std::int32_t` for process-local indices and local
  offsets, `std::int64_t` for global indices and global offsets, `int`
  for MPI ranks, counts and displacements, `std::size_t` for `.size()`
  results. The 32-bit local index is a deliberate commitment — a rank is
  not expected to exceed 2^31 entities, and the narrow type halves index
  array memory traffic — so local indices must not be silently widened
  in storage or interfaces. Type a variable by the role of its value,
  not by the expression that initialises it.
- **PETSc/SLEPc index types**: in the thin PETSc/SLEPc wrappers
  (`la::petsc`, `la::slepc`, `nls::petsc`), an index or count passed to
  or obtained from a PETSc/SLEPc call is `PetscInt`, including where it
  is returned to the caller or passed on to a callback, matching the
  `PetscScalar`/`Mat`/`Vec` already in those signatures. The fixed-width
  types above are for DOLFINx-meaningful quantities, such as the sizes
  and ranges returned by `petsc::Vector`.
- **Iterator distances**: store `std::ranges::distance` results, a
  signed `difference_type`, in `std::size_t` when used as a container
  offset. They are non-negative by construction here, and
  `-Wsign-compare` is `-Werror`, so `std::ptrdiff_t` would force a cast
  at every comparison against `.size()`.
- **Narrowing conversions**: neither `-Wconversion` nor
  `-Wshorten-64-to-32` is enabled, so implicit 64-to-32 narrowing is
  legal and widespread. `static_cast` only where the narrowing is the
  point — a public API returning a local index or count
  (`IndexMap::size_local`, `IndexMap::num_ghosts`) — not elsewhere.
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
- **Comments**: LLM-generated comments tend to be rather verbose; after
  the first comment draft, compress comments to their essence using
  concise technical language.
- **Errors and invariants**: For user-facing/API-boundary errors, throw
  `std::invalid_argument` for a bad argument or violated parameter
  precondition, `std::out_of_range` for an index/lookup-key failure, and
  `std::runtime_error` for other runtime/state/IO/MPI failures. Do not
  introduce a custom exception hierarchy. Use a descriptive message —
  unconditionally when the check is O(1), or guarded behind
  `#ifndef NDEBUG` when the check is more expensive, so it's skipped in
  release builds. For internal
  invariants that indicate a library bug rather than bad user input, use
  `assert` when the check fits in a single expression, or a
  `#ifndef NDEBUG`-guarded block with an explicit throw/abort when it
  needs multiple statements. Do not add exceptions inside hot loops.
  Prefer `spdlog::debug`/`info`/`warn` for logging over
  `std::cout`/`std::cerr`.
- **MPI collectives**: collective operations (`MPI_Allreduce`,
  neighbourhood collectives, etc.) must be reached by every rank in the
  communicator — an error path, early return, or exception on one rank
  must not skip a collective that other ranks still call, or the
  mismatch deadlocks. Validate/throw before entering a code path with
  collectives, not conditionally partway through it.
- **Move/copy semantics**: Moving is preferred over copying, unless
  the object is very lightweight. Many DOLFINx classes disable
  copying; none disable moving. `std::move` is used systematically on
  incoming `std::shared_ptr` to avoid unnecessary copies of
  `std::shared_ptr` and also to avoid copies when returning with a
  `std::pair{}`.
- **Special member functions**: a class that declares any of the five
  (copy constructor, move constructor, destructor, copy assignment,
  move assignment) declares all five explicitly, `public`, as a
  contiguous block immediately after the named constructors, in that
  order — see `mesh/Mesh.h`, `fem/DofMap.h`, `common/Table.h`. Write
  `= default` rather than relying on implicit generation; declaring
  only some of the five silently suppresses the others.
- **Deleting copies**: delete both copy operations for classes owning
  an external handle (MPI communicator, PETSc/SLEPc object,
  ADIOS2/HDF5 file) and for 'heavy' data classes where an accidental
  deep copy is a performance bug (`common::IndexMap`,
  `fem::FunctionSpace`, `fem::Function`). Classes that are cheap to
  copy explicitly but should not be copied into an existing object
  delete copy assignment only, keeping a defaulted copy constructor
  (`mesh::Mesh`, `mesh::Topology`, `mesh::Geometry`).
- **Never delete moves**: no class in the library deletes a move
  operation. If a class caches `std::span`s or other pointers into its
  own members, explain in a `@note` why moving remains valid instead
  of disabling it (see the deleted copy constructor and defaulted move
  constructor of `fem::Form`).
- **Destructors**: `= default` unless a raw handle must be released
  (`common::Comm`, `la::petsc::Vector`, `io::VTKFile`). Mark the
  destructor `virtual` only when the class is actually used as a base
  class.
- **`noexcept`**: used on hand-written move operations of handle owners
  (`common::Comm`, `la::petsc::*`); defaulted moves are left
  unannotated as they are implicitly `noexcept`.
- **Documenting special members**: the conventional wordings are
  `/// Copy constructor`, `/// Move constructor`, `/// Destructor`,
  `/// Copy assignment` and `/// Move assignment`; `@param` is
  normally omitted. Deleted members use a non-Doxygen `//` comment
  with `(deleted)` appended so they stay out of the generated
  documentation.
- **String formatting**: use `std::format` (`<format>`) to build
  formatted/error strings rather than `printf`-style,
  `std::ostringstream` concatenation, or the `fmt` library.
- **Function pointers over lambdas**: when a free function's signature
  already matches a callback/`std::function` parameter exactly, pass
  the function directly (e.g. `graph::reorder_rcm`) rather than
  wrapping it in a trivial forwarding lambda.
- **Lambdas**: no `[=]`/`[&]` — list captures explicitly, e.g. `[&v]`.
  Capture by reference for lambdas invoked in place; by value (cheap
  scalars, or `[v = std::move(v)]` to move a container) for lambdas that
  escape the scope, where a captured reference or `span` into a local
  dangles silently. `[this]` only if the lambda cannot outlive the
  object. Never capture a container or `shared_ptr` by value for
  convenience: the copy happens at capture and again whenever the lambda
  or its `std::function` is copied.
  Prefer explicit parameter types over `auto` (`auto&&` in generic code)
  in non-generic code; add an explicit return type when the deduced one
  is non-obvious or must not decay. Avoid `mutable`. Promote long or
  reused lambdas to a free function in an anonymous namespace. A
  by-reference capture is `const` only if the captured variable is, so
  declare read-only locals `const`.
- **Algorithms**: prefer `std::ranges` algorithms (`std::ranges::...`)
  over both hand-written loops and their pre-ranges `<algorithm>`
  equivalents, where it doesn't hurt clarity or performance. Flattened
  row-major storage is the default convention for multi-dimensional
  data passed as flat buffers.
- **`std::distance`/`std::advance`/`std::next`/`std::prev` on a
  generic, template-parameterized range**: these legacy `<iterator>`
  algorithms dispatch on `std::iterator_traits<It>::iterator_category`,
  not the C++20 iterator concepts. Views such as
  `std::ranges::iota_view` satisfy `std::random_access_iterator` but
  not the legacy `LegacyRandomAccessIterator` (`operator*` returns a
  prvalue, not a reference), so their `iterator_category` degrades to
  `input_iterator_tag` and `std::distance` silently falls back to an
  O(n) count instead of an O(1) subtraction, turning an O(n) loop into
  O(n²) in unoptimised (Debug) builds only. When indexing into a
  generic range parameter inside a loop, use
  `std::ranges::distance`/`std::ranges::advance`, which dispatch on
  the C++20 concept and stay O(1) for these views, or track the index
  with a plain counter incremented alongside the iterator.
- **Windows**: Windows is continuously tested on GitHub with the most
  important missing feature being the lack of C99 `_Complex` support
  denoted by existence of `DOLFINX_NO_STDC_COMPLEX_KERNELS` macro
  variable.
- **PETSc support is optional**: Tests should avoid depending on PETSc
  unless the functionality under test is PETSc-specific; PETSc-related
  functionality in the main library should be isolated so it can be
  excluded from non-PETSc builds. For example, the test suite includes a
  simple conjugate-gradient solver for small problems rather than
  depending on a PETSc `KSP` solver.

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
- Do not give bound arguments Python-visible default values
  (`nb::arg("x") = value`) in the C++ nanobind wrapping code — defaults
  belong in the pure-Python layer.
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
- Dependency groups (`build`, `docs`, `lint`, `test`, `ci` in
  `python/pyproject.toml`) use PEP 735 syntax and require `pip >= 25.1`
  (or another PEP 735-compliant build frontend) for the `--group` flag.
  `demo`, `optional`, `petsc4py`, and `typing` remain real
  `[project.optional-dependencies]` extras since they are user-facing
  runtime features, or (in the case of `test`) are installed against
  built wheels where dependency groups are unavailable.

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
