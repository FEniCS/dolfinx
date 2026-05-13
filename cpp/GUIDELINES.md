## DOLFINx C++ API design notes

This is a guide to the design idioms used in the DOLFINx C++ library. The
goal is to keep new code consistent with the patterns already established
under `cpp/dolfinx/`. Formatting and lint rules are handled by
`.clang-format` and `.clang-tidy` and are not repeated here.

## Templates and concepts

Public templated code is constrained with concepts, not SFINAE. Use the
project concepts where they exist before reaching for `std::*`:

- `dolfinx::scalar` (`common/types.h`) — real or complex floating point.
  Use this on anything that can hold field values (matrix entries,
  function coefficients).
- `std::floating_point U` — geometry / coordinate scalars only.
- `dolfinx::scalar_value_t<T>` — extracts the real value type from `T`,
  whether `T` is real or `std::complex<...>`.
- `dolfinx::md` is the project alias for the mdspan namespace; never spell
  out `MDSPAN_IMPL_STANDARD_NAMESPACE` in API code.

The canonical class signature on a field-bearing template is:

```cpp
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
class Function { ... };
```

Define new concepts for callback / kernel parameters rather than taking
raw `std::function` everywhere. See `fem::FEkernel`, `fem::DofTransformKernel`,
`fem::MDSpan2`, `la::VectorPackKernel`. A concept makes the call signature
discoverable and avoids the `std::function` indirection for hot kernels.

Constrain template parameters at the point of use too — `requires
std::is_convertible_v<std::remove_cvref_t<V>, Geometry<T>>` on the
forwarding `Mesh` constructor, `mesh::CellRange auto&&` on the
interpolation overloads in `Function.h`.

## Container parameters

Choose by ownership and shape:

- **`std::span<const T>`** for non-owning, contiguous input. This is the
  default for "give me a list of indices/values".
- **`std::span<T>`** for non-owning, contiguous output to a buffer the
  caller already owns.
- **`md::mdspan<...>`** for multi-dimensional non-owning views. Prefer
  `md::dextents<std::size_t, N>` for runtime extents and
  `md::extents<std::size_t, K, md::dynamic_extent>` when one extent is
  known at compile time (e.g. the geometric dimension `3`).
- **`std::vector<T>`** only as a return type (the function owns and
  hands ownership over) or for class data members.
- **`std::ranges::range auto&&`** / domain concepts like `mesh::CellRange`
  when an algorithm should accept any forward range — see the
  `Function::interpolate` overloads which accept both
  `std::ranges::iota_view` and explicit `std::vector<std::int32_t>`.

Do not take `const std::vector<T>&` in new APIs. Span-ify it.

## Return values

Return by value. Multi-valued returns use `std::pair`, `std::tuple`, or
`std::array`, unpacked at the call site with structured bindings:

```cpp
auto [V, map] = _function_space->collapse();
auto [unique_end, range_end] = std::ranges::unique(vertices);
```

`std::optional<T>` is used for "may be absent" returns and optional
parameters (e.g. `max_facet_to_cell_links`); do not use sentinel values or
empty containers to mean "not provided". `nullptr` is reserved for
optional shared/unique pointer arguments.

Document tuple-returns in the `@return` block by numbering: "(0) ..., (1)
..., (2) ...". This is the house convention — see `stack_index_maps`,
`create_sub_index_map`, `compute_vertex_coords_boundary`.

Use `std::move` on the constituents when assembling a returned aggregate
to make the move explicit:

```cpp
return {std::move(entities), std::move(x_vertices), std::move(vertex_to_pos)};
```

## Ownership and lifetime

- Long-lived collaborators held by a class are `std::shared_ptr<const T>`
  members. Constructors take them by `std::shared_ptr<const T>` value.
  The `const` in the template signals that the object is observed, not
  mutated, through this reference.
- Mutable shared state uses `std::shared_ptr<T>` deliberately (e.g.
  `Mesh::_topology`, with a `topology_mutable()` accessor whose existence
  is explained in a comment).
- Value types that are cheap to construct (`la::Vector`, `mesh::Mesh`,
  `IndexMap`) are non-copyable but moveable. Spell every special member
  out: `= default` / `= delete` / `= default` for the move pair, with a
  doc comment on each:

  ```cpp
  Function(const Function&) = delete;
  Function(Function&&) = default;
  Function& operator=(const Function&) = delete;
  Function& operator=(Function&&) = default;
  ~Function() = default;
  ```

- Constructors that store a heavy aggregate use perfect-forwarding with a
  `requires` clause (see `Mesh(MPI_Comm, shared_ptr<Topology>, V&&)`).
- `MPI_Comm` is wrapped in `dolfinx::MPI::Comm` for RAII duplication; raw
  `MPI_Comm` is fine at API boundaries (returned by `.comm()`).

## Free functions over methods

Operations that combine objects of multiple types live as free functions
in the relevant namespace (`fem::create_dofmap`, `fem::create_form`,
`mesh::create_mesh`, `common::create_sub_index_map`,
`la::norm`). Reserve member functions for queries and mutators of a
single object's state.

Constructors of public templated classes are intentionally minimal; the
heavyweight "build a thing from a description" work goes into a
`create_<thing>` free function. This keeps the class header small and
lets the factory live in the `.cpp` where possible.

Use `using` aliases to give a name to recurring callback types
(`mesh::CellPartitionFunction`, `mesh::CellReorderFunction`) instead of
re-spelling the `std::function<...>` signature in every overload.

## Algorithms — prefer the standard library

Reach for `std::ranges::*` first, then `std::*`. New code should not be
writing index loops where an algorithm exists.

Common patterns in the codebase:

```cpp
std::ranges::sort(vertices);
auto [first, last] = std::ranges::unique(vertices);
vertices.erase(first, last);

std::ranges::copy(dofs00, macro_dofs0.begin());

auto it = std::ranges::find(cell_vertices, v);
std::size_t local_pos = std::distance(cell_vertices.begin(), it);

int dim_sum = std::accumulate(entity_dofs[dim].begin(), entity_dofs[dim].end(),
                              0, [](int c, auto v) { return c + v.size(); });

std::transform(idx_first, idx_last, out_first,
               [in_first](auto p) { return *std::next(in_first, p); });
```

Iota ranges are used to express "all cells" without materialising a
vector:

```cpp
interpolate(f, std::ranges::iota_view(0, num_cells));
```

When working with iterators received via a concept (the
`VectorPackKernel` pattern), advance with `std::next(it, n)` rather than
`it + n` — it works for non-random-access iterators too.

## Error handling

- `throw std::runtime_error("...")` for argument / state misuse that a
  caller could plausibly hit. Messages are user-facing English sentences.
- `assert(...)` for internal invariants and pointer non-null checks.
  These are stripped in release builds and document the contract
  inline.
- `static_assert(...)` for compile-time container/value-type agreement
  (see the `Container::value_type == T` assertion in `la::Vector`).
- No exception specifications, no error codes, no `std::expected` yet.
- `noexcept` is used only where it's actually load-bearing (the
  `MPI::Comm` move/copy operations).

## Documentation

Doxygen comments on every public declaration:

- `@brief` one-liner, then a blank `///` line, then any longer
  discussion.
- `@param[in]` / `@param[out]` / `@param[in,out]` with direction always
  specified.
- `@tparam` for every template parameter.
- `@pre` for caller-side preconditions (`indices` sorted, no duplicates,
  facets on the boundary, …). Match these with `assert`s where cheap.
- `@note Collective.` on every collective MPI entry point.
- `@warning` for "internal use only" constructors and footguns.
- LaTeX math in `\f[ ... \f]` / `\f$ ... \f$` is welcome on the
  user-facing classes (see `fem::Function`).

`@private` / `@cond` / `@endcond` hide helpers from generated docs;
`impl` sub-namespaces inside headers do not need to be in the public
docs and should be marked accordingly.

## Logging and timing

- `spdlog::debug("Counting entity dofs, dim={}: {}", dim, dim_sum);` is
  the in-library logging API. `info`, `warn`, `debug`, `trace` levels are
  all used. Don't use `std::cout`.
- Format strings use the `{}` syntax. For local string formatting use
  `std::format` (header is included as `<format>` in `fem/utils.h` and
  `mesh/utils.h`).
- Wrap timed sections with `common::Timer` rather than rolling your own
  `std::chrono` measurement.

## PETSc interop

PETSc is an optional dependency. Code that touches PETSc lives in
`dolfinx/la/petsc.{h,cpp}` (generic Vec/Mat/KSP wrappers) and
`dolfinx/fem/petsc.{h,cpp}` (assembly into PETSc data structures).
Anything else stays PETSc-agnostic and works with the templated
`la::Vector`, `la::MatrixCSR`, etc.

### Compilation guard

Every PETSc header and translation unit is wrapped:

```cpp
#ifdef HAS_PETSC
// ...
#endif
```

The guard goes immediately after `#pragma once` / the file header, and
closes at the end of the file. Do not sprinkle inner `#ifdef HAS_PETSC`
blocks inside otherwise-portable headers; either the whole unit is
PETSc-only or it doesn't touch PETSc.

### Ownership: raw handles out, RAII wrappers optional

Free functions return raw `Vec` / `Mat` / `IS` / `MatNullSpace` handles
and the Doxygen explicitly states who destroys them:

```cpp
/// @note Caller is responsible for destroying the returned object
Vec create_vector(const common::IndexMap& map, int bs);
```

For convenience there are thin RAII wrappers — `la::petsc::Vector`,
`la::petsc::Matrix`, `la::petsc::Operator`, `la::petsc::KrylovSolver` —
that own a PETSc handle and call `*Destroy` in the destructor. They all
follow the same shape:

- Non-copyable, move-only (same special-member pattern as elsewhere in
  the library).
- A "construct from scratch" constructor that creates the PETSc object.
- A "wrap existing" constructor taking the raw handle plus a
  `bool inc_ref_count` flag. When `true`, the wrapper bumps the PETSc
  reference count; either way the destructor calls `*Destroy`, which
  decrements it. This is how you safely adopt a `Vec`/`Mat` handed in
  from elsewhere.
- A `.vec()` / `.mat()` / `.ksp()` accessor returning the raw handle for
  callers who need the full PETSc API.

Wrappers are deliberately thin — do not grow them into a full
re-implementation of the PETSc API. The escape hatch is `.vec()` etc.

### Error translation

PETSc returns `PetscErrorCode`. The house pattern is to call the PETSc
function, capture the code, and route non-zero codes through
`la::petsc::error`, which logs via `spdlog` and throws
`std::runtime_error`. In `.cpp` files this is wrapped by a
`CHECK_ERROR(name)` macro defined at the top of the file:

```cpp
PetscErrorCode ierr;
ierr = VecCreateGhost(comm, local_size, PETSC_DETERMINE, _ghosts.size(),
                      _ghosts.data(), &x);
CHECK_ERROR("VecCreateGhost");
```

Inside hot inner lambdas (the matrix `set_fn` family) the check is
gated on `#ifndef NDEBUG` so release builds skip the branch entirely.
Use that pattern when the call is on the assembly fast path; otherwise
check unconditionally.

### Insertion lambdas for assembly

Assemblers do not call `MatSetValuesLocal` directly. They take a "set
function" with signature

```cpp
int(std::span<const std::int32_t> rows,
    std::span<const std::int32_t> cols,
    std::span<const PetscScalar> vals)
```

built by static factories on `la::petsc::Matrix`: `set_fn`,
`set_block_fn`, `set_block_expand_fn`. Each returns a `mutable` lambda
with a `std::vector<PetscInt>` captured by value as a per-call scratch
buffer:

```cpp
return [A, mode, cache = std::vector<PetscInt>()](
           std::span<const std::int32_t> rows,
           std::span<const std::int32_t> cols,
           std::span<const PetscScalar> vals) mutable -> int { ... };
```

The cache exists to handle the `PETSC_USE_64BIT_INDICES` build — when
`PetscInt` is 64-bit we have to widen the 32-bit row/column spans into
the cache before calling PETSc; otherwise the spans are forwarded
directly. Guard with `#ifdef PETSC_USE_64BIT_INDICES`. Write new
insertion-style adapters this way rather than allocating per call.

### Type and scalar agreement

The PETSc API uses the configured `PetscScalar`. DOLFINx templates that
target PETSc fix the scalar to `PetscScalar` at the API boundary:

```cpp
template <std::floating_point T>
Mat create_matrix(const Form<PetscScalar, T>& a,
                  std::optional<std::string> type = std::nullopt);
```

The geometry type `T` (real `std::floating_point`) remains a template
parameter. Do not template over `PetscScalar` — pin it.

Use `std::optional<std::string> type = std::nullopt` for "use the PETSc
default Mat type". For nested matrix types use
`std::optional<std::vector<std::vector<std::optional<std::string>>>>`.
That is the idiomatic way to express "no opinion" at every level of a
nested matrix description.

### Options

The PETSc options database is wrapped in `la::petsc::options::{set,
clear}` free functions. `set` is a function template that lexical-casts
the value to a string, prepends `-` if the user omitted it, and routes
errors through `petsc::error`. Prefer this to calling
`PetscOptionsSetValue` directly so error handling stays uniform.

Wrapper classes expose `set_options_prefix` / `get_options_prefix` /
`set_from_options` mirroring the PETSc names so PETSc users can find
them.

### Exposing IndexMaps to PETSc

When a DOLFINx `IndexMap` (or a stack of them) needs to be exposed to
PETSc, use `la::petsc::create_index_sets` and the
`std::vector<std::pair<std::reference_wrapper<const common::IndexMap>,
int>>` `(map, bs)` pair-of-references shape that PETSc-facing functions
all take. The `reference_wrapper` lets the same map appear multiple
times without copying and works with structured bindings.

## Files and namespaces

- One class per public header; companion `.cpp` for non-template
  definitions. Implementation-detail headers end in `_impl.h` (see
  `fem/assemble_matrix_impl.h`).
- Each module exposes a single `dolfinx_<module>.h` umbrella header that
  pulls in the public API.
- Namespaces mirror directories: `dolfinx::fem`, `dolfinx::mesh`,
  `dolfinx::la`, `dolfinx::common`, `dolfinx::MPI`. Nested `impl` or
  anonymous namespaces hide helpers — keep them out of the public
  namespace.
- Use `namespace dolfinx::fem { ... }` directly; don't nest `namespace
  dolfinx { namespace fem { ... } }`.
