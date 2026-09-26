// ```text
// Copyright (C) 2024-2026 Jack S. Hale and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Custom cell kernel assembly
//
// Finite element assembly normally starts from a variational form written in
// UFL. FFCx generates element kernels from the form, and DOLFINx executes the
// kernels over the mesh and inserts their output into a global tensor. For
// applications with specialised or very small kernels, it can be useful to
// supply the element kernel directly and, where appropriate, bypass the
// higher-level {cpp:class}`dolfinx::fem::Form` interface.
//
// This demo compares three ways to assemble the same operator and vector:
//
// 1. Hand-written kernels wrapped in a {cpp:class}`dolfinx::fem::Form` and
//    passed to the standard assembly functions.
// 2. The same hand-written kernels passed as concrete lambda types to the
//    low-level cell assembly functions. The compiler can specialise the cell
//    loop for the lambda and inline the kernel.
// 3. FFCx-generated kernels called directly from lambdas and passed to the
//    same low-level cell assembly functions. Including the generated source
//    in this translation unit makes those kernel bodies available for
//    inlining too.
//
// ```{note}
// This is an advanced demo. Most applications should express forms in UFL
// and use the public {cpp:class}`dolfinx::fem::Form` assembly interface.
// ```
//
// ```{warning}
// The direct assembly route uses internal `fem::impl` and UFCx interfaces.
// It is a performance-oriented example, not a stable user-facing API.
// ```
//
// ## Problem definition
//
// Let $V_h$ be the scalar, continuous, piecewise linear Lagrange space on a
// triangular mesh of $\Omega = [0, 1]^2$. We assemble the mass matrix and the
// load vector for a unit source:
//
// \begin{align*}
//   A_{ij} &= \int_{\Omega} \phi_i \phi_j \, \mathrm{d}x, \\
//   b_i &= \int_{\Omega} \phi_i \, \mathrm{d}x.
// \end{align*}
//
// On an affine cell $K$, the hand-written kernels obtain the local tensors
// from reference-cell tensors by multiplying by the absolute Jacobian
// determinant:
//
// \begin{align*}
//   A^K &= |\det J_K| \widehat{A}, \\
//   b^K &= |\det J_K| \widehat{b}.
// \end{align*}
//
// Basix supplies the quadrature rule and tabulated P1 basis functions used to
// compute $\widehat{A}$ and $\widehat{b}$. Each assembly variant returns a
// squared norm of its assembled tensor, allowing the results to be checked
// against the conventional form-based route.
//
// Running this demo requires the files:
// {download}`demo_custom_kernel/main.cpp`,
// {download}`demo_custom_kernel/mass.py` and
// {download}`demo_custom_kernel/CMakeLists.txt`.
//
// ## UFL forms
//
// The forms used to generate the third pair of kernels are defined in
// {download}`demo_custom_kernel/mass.py`. They are mathematically identical
// to the hand-written kernels below. Each integral carries
// `ffcx_kernel_name` metadata, which fixes the name of the generated kernel
// function so that it can be called directly from C++. Without it, FFCx
// derives the name from the Python variable holding the form.
//
// ````{admonition} UFL forms implemented in Python
// :class: dropdown
// ![ufl-code]
// ````
//
// ## C++ program
//
// Normal DOLFINx solvers compile the FFCx-generated C source in a separate
// translation unit, so the assembler can reach the kernels only through
// function pointers. Here the generated source is included in `main.cpp` so
// that its function definitions and the templated DOLFINx cell loops are
// visible to the optimiser in the same translation unit, which allows the
// kernels to be inlined. The generated header declares the named kernels.
// `restrict` is a C keyword used by the generated kernels, so it is mapped to
// the corresponding compiler extension while the generated source is parsed as
// C++.

#include "mass.h"

// Include the generated C source to make its kernels visible to the compiler.
#if defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict __restrict__
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "mass.c"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#undef restrict

#include <algorithm>
#include <array>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ## Compile-time shapes
//
// The low-level assembly functions are templates over their `mdspan`
// arguments. Encoding dimensions that are fixed by the problem in the
// `mdspan` type makes those values available while the cell loop is compiled,
// which allows the compiler extra scope for optimisation, e.g. fully unrolling
// loops over cell degrees-of-freedom.
// For a scalar P1 triangle there are three coordinate degrees-of-freedom and
// three field degrees-of-freedom per cell. DOLFINx stores each geometry point
// in three components, including for a two-dimensional mesh. Only the number
// of cells and the number of quadrature points remain dynamic here.

using namespace dolfinx;
template <typename T, std::size_t n0, std::size_t n1>
using mdspan2_t = md::mdspan<T, std::extents<std::size_t, n0, n1>>;
constexpr std::size_t p1_triangle_dofs_per_cell = 3;
using p1_triangle_dofmap_t = mdspan2_t<const std::int32_t, md::dynamic_extent,
                                       p1_triangle_dofs_per_cell>;
template <typename T>
using triangle_points_t
    = md::mdspan<T, md::extents<std::size_t, md::dynamic_extent, 2>>;
template <typename T>
using p1_triangle_basis_t
    = md::mdspan<T, md::extents<std::size_t, 1, md::dynamic_extent,
                                p1_triangle_dofs_per_cell, 1>>;
static_assert(p1_triangle_dofmap_t::static_extent(1)
              == p1_triangle_dofs_per_cell);

// ## Reference-cell tensors
//
// We first compute the tensors on the reference triangle. If $w_q$ and
// $\widehat{\phi}_i(X_q)$ are the quadrature weights and tabulated basis
// values, respectively, then
//
// \begin{align*}
//   \widehat{A}_{ij}
//     &= \sum_q w_q \widehat{\phi}_i(X_q)\widehat{\phi}_j(X_q), \\
//   \widehat{b}_i &= \sum_q w_q \widehat{\phi}_i(X_q).
// \end{align*}
//
// The basis view has shape `(1, num_points, 3, 1)`: one derivative entry
// because only values are tabulated, three P1 basis functions and one scalar
// value component.

/// @brief Compute the P1 element mass matrix on the reference cell.
/// @tparam T Scalar type.
/// @param phi Basis functions.
/// @param w Integration weights.
/// @return Element reference matrix (row-major storage).
template <typename T>
std::array<T, p1_triangle_dofs_per_cell * p1_triangle_dofs_per_cell>
A_ref(p1_triangle_basis_t<const T> phi, std::span<const T> w)
{
  std::array<T, p1_triangle_dofs_per_cell * p1_triangle_dofs_per_cell> A_b{};
  mdspan2_t<T, p1_triangle_dofs_per_cell, p1_triangle_dofs_per_cell> A(
      A_b.data());
  for (std::size_t k = 0; k < phi.extent(1); ++k)   // quadrature point
    for (std::size_t i = 0; i < A.extent(0); ++i)   // row i
      for (std::size_t j = 0; j < A.extent(1); ++j) // column j
        A(i, j) += w[k] * phi(0, k, i, 0) * phi(0, k, j, 0);
  return A_b;
}

/// @brief Compute the P1 RHS vector for f=1 on the reference cell.
/// @tparam T Scalar type.
/// @param phi Basis functions.
/// @param w Integration weights.
/// @return RHS reference vector.
template <typename T>
std::array<T, p1_triangle_dofs_per_cell> b_ref(p1_triangle_basis_t<const T> phi,
                                               std::span<const T> w)
{
  std::array<T, p1_triangle_dofs_per_cell> b{};
  for (std::size_t k = 0; k < phi.extent(1); ++k) // quadrature point
    for (std::size_t i = 0; i < b.size(); ++i)    // row i
      b[i] += w[k] * phi(0, k, i, 0);
  return b;
}

template <std::floating_point T>
void check_norm(double norm, double reference)
{
  const double tol = 1.0e4 * std::numeric_limits<T>::epsilon()
                     * std::max({1.0, std::abs(norm), std::abs(reference)});
  if (std::abs(norm - reference) > tol)
    throw std::runtime_error("Assembly variants have different norms.");
}

// ## Assembly through `fem::Form`
//
// The first route packages a kernel and the cells on which it is active as
// integral data for a {cpp:class}`dolfinx::fem::Form`. This is the usual
// DOLFINx abstraction: the public assembly functions obtain the mesh,
// dofmaps, kernels and integration domains from the form. A custom callable
// is stored behind the form's type-erased kernel interface.
//
// Matrix assembly additionally needs a sparsity pattern. It is populated from
// the test and trial dofmaps before the matrix is created. After local cell
// contributions have been inserted, `scatter_rev` accumulates contributions
// for shared degrees-of-freedom on their owning MPI ranks.

/// @brief Assemble a matrix operator using a `std::function` kernel
/// function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return Frobenius norm squared of the matrix.
template <std::floating_point T>
double assemble_matrix0(std::shared_ptr<const fem::FunctionSpace<T>> V,
                        auto kernel, const std::vector<std::int32_t>& cells)
{
  // Kernel data (ID, kernel function, cell indices to execute over)
  std::map integrals{
      std::pair{std::tuple{fem::IntegralType::cell, 0, 0},
                fem::integral_data<T>(kernel, cells, std::vector<int>{})}};

  fem::Form<T, T> a({V, V}, integrals, V->mesh(), {}, {}, false, {});
  auto dofmap = V->dofmap();
  auto sp = la::SparsityPattern(
      V->mesh()->comm(), {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sp, std::pair{cells, cells}, {*dofmap, *dofmap});
  sp.finalize();
  la::MatrixCSR<T> A(sp);
  common::Timer timer("Assembler0 std::function (matrix)");
  assemble_matrix(A.mat_add_values(), a, {});
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using a `std::function` kernel
/// function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <std::floating_point T>
double assemble_vector0(std::shared_ptr<const fem::FunctionSpace<T>> V,
                        auto kernel, const std::vector<std::int32_t>& cells)
{
  auto mesh = V->mesh();
  std::map integrals{
      std::pair{std::tuple{fem::IntegralType::cell, 0, 0},
                fem::integral_data<T>(kernel, cells, std::vector<int>{})}};
  fem::Form<T> L({V}, integrals, mesh, {}, {}, false, {});
  auto dofmap = V->dofmap();
  la::Vector<T> b(dofmap->index_map, 1);
  common::Timer timer("Assembler0 std::function (vector)");
  fem::assemble_vector(b.array(), L);
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

// ## Direct cell assembly
//
// The second route calls the implementation-level cell loops directly. The
// caller must now provide all data that a `Form` would normally organise:
// geometry and field dofmaps, active cells, degree-of-freedom transformations,
// coefficients, constants, permutation data and correctly sized work buffers.
// This demo has no coefficients, constants, boundary conditions or required
// permutations, so the corresponding views and callbacks are empty.
//
// The advantage is that `kernel` retains its concrete type as it passes into
// `assemble_cells_matrix` or `assemble_cells`. Together with the static
// `mdspan` extents and compile-time block size, this allows the compiler to
// specialise the cell loop for this element and kernel.

/// @brief Assemble a matrix operator using a lambda kernel function.
///
/// The lambda function can be inlined in the assembly code, which can
/// be important for performance for lightweight kernels.
///
/// @tparam T Scalar type.
/// @param g mesh geometry.
/// @param dofmap dofmap.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return Frobenius norm squared of the matrix.
template <std::floating_point T>
double assemble_matrix1(const mesh::Geometry<T>& g, const fem::DofMap& dofmap,
                        auto kernel, std::span<const std::int32_t> cells)
{
  auto sp = la::SparsityPattern(dofmap.index_map->comm(),
                                {dofmap.index_map, dofmap.index_map},
                                {dofmap.index_map_bs(), dofmap.index_map_bs()});
  fem::sparsitybuild::cells(sp, std::pair{cells, cells}, {dofmap, dofmap});
  sp.finalize();
  la::MatrixCSR<T> A(sp);
  auto ident = [](auto, auto, auto, auto) {}; // DOF permutation not required
  common::Timer timer("Assembler1 lambda (matrix)");

  // `Geometry` and `DofMap` expose views with a dynamic second extent. Check
  // the P1 triangle invariant, then rewrap the same memory in views whose
  // three entries per cell are part of the type. The coordinate array has
  // three stored components per geometry point.
  const auto x_dofmap0 = g.dofmaps().front();
  assert(x_dofmap0.extent(1) == p1_triangle_dofs_per_cell);
  p1_triangle_dofmap_t x_dofmap(x_dofmap0.data_handle(), x_dofmap0.extent(0));
  const auto dmap0 = dofmap.map();
  assert(dmap0.extent(1) == p1_triangle_dofs_per_cell);
  p1_triangle_dofmap_t dmap(dmap0.data_handle(), dmap0.extent(0));
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>> x(
      g.x().data(), g.x().size() / 3, 3);

  // The direct assembler does not allocate inside the cell loop. Supply
  // storage for the 3-by-3 element matrix and the three geometry points,
  // each of which has three coordinate components. The dofmap tuples contain
  // the dofmap, its block size and the cell indices. The scalar block size is
  // represented by `integral_constant`, making it available at compile time.
  std::array<T, 3 * p1_triangle_dofs_per_cell> cdofs_b;
  std::array<T, p1_triangle_dofs_per_cell * p1_triangle_dofs_per_cell> Ab;
  fem::impl::assemble_cells_matrix<false>(
      A.mat_add_values(), x_dofmap, x, cells,
      std::tuple{dmap, std::integral_constant<int, 1>{}, cells}, ident,
      std::tuple{dmap, std::integral_constant<int, 1>{}, cells}, ident, {}, {},
      kernel, {}, {}, {}, {}, std::span<T>(Ab), std::span<T>(cdofs_b));
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using a lambda kernel function.
///
/// The lambda function can be inlined in the assembly code, which can
/// be important for performance for lightweight kernels.
///
/// @tparam T Scalar type.
/// @param g mesh geometry.
/// @param dofmap dofmap.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <std::floating_point T>
double assemble_vector1(const mesh::Geometry<T>& g, const fem::DofMap& dofmap,
                        auto kernel, const std::vector<std::int32_t>& cells)
{
  la::Vector<T> b(dofmap.index_map, 1);

  // As above, expose the known cell-local dimensions through the view types.
  const auto x_dofmap0 = g.dofmaps().front();
  assert(x_dofmap0.extent(1) == p1_triangle_dofs_per_cell);
  p1_triangle_dofmap_t x_dofmap(x_dofmap0.data_handle(), x_dofmap0.extent(0));
  const auto dmap0 = dofmap.map();
  assert(dmap0.extent(1) == p1_triangle_dofs_per_cell);
  p1_triangle_dofmap_t dmap(dmap0.data_handle(), dmap0.extent(0));
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>> x(
      g.x().data(), g.x().size() / 3, 3);
  common::Timer timer("Assembler1 lambda (vector)");

  // Vector assembly needs a three-entry element vector rather than a 3-by-3
  // element matrix. Geometry storage is unchanged.
  std::array<T, 3 * p1_triangle_dofs_per_cell> cdofs_b;
  std::array<T, p1_triangle_dofs_per_cell> be_b;
  fem::impl::assemble_cells(
      [](auto, auto, auto, auto) {}, b.array(), x_dofmap, x, cells,
      std::tuple{dmap, std::integral_constant<int, 1>{}, cells}, kernel, {}, {},
      {}, std::span<T>(be_b), std::span<T>(cdofs_b));
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

// ## Defining the kernels
//
// We now construct the mesh, element and quadrature rule shared by all three
// assembly routes. The hand-written kernels use the UFCx cell-kernel calling
// convention. In order, their arguments are the output tensor, coefficient
// values, constants, coordinate degrees-of-freedom, local facet information,
// quadrature permutation data and optional custom data. Only the output and
// coordinates are needed for these cell integrals.

/// @brief Assemble a P1 mass matrix and load vector by three kernel routes.
///
/// @tparam T Scalar type.
/// @param comm MPI communicator to assemble over.
template <std::floating_point T>
void assemble(MPI_Comm comm)
{
  // Use a linear triangular mesh, so the coordinate map is affine on every
  // cell and its Jacobian determinant is constant on the cell.
  auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_rectangle<T>(
      comm, {{{0, 0}, {1, 1}}}, {516, 116}, mesh::CellType::triangle));

  // Create the Basix P1 Lagrange element used both by the function space and
  // to tabulate the reference basis functions.
  constexpr int order = 1;
  basix::FiniteElement e = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), order,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // The mass integrand is a polynomial of degree two, so a degree-two rule
  // integrates both the mass and load reference tensors exactly.
  constexpr int max_degree = 2 * order;
  auto quadrature_type = basix::quadrature::get_default_rule(
      basix::cell::type::triangle, max_degree);
  auto [X_b, weights] = basix::quadrature::make_quadrature<T>(
      quadrature_type, basix::cell::type::triangle,
      basix::polyset::type::standard, max_degree);
  triangle_points_t<const T> X(X_b.data(), weights.size());

  // Create the scalar finite element function space.
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
      mesh,
      std::make_shared<fem::FiniteElement<T>>(e, mesh->geometry().dim())));

  // Assemble over the cells owned by this rank. Ghost-cell contributions are
  // not assembled; reverse scatter later accumulates shared-dof entries.
  std::int32_t size_local
      = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
  std::vector<std::int32_t> cells(size_local);
  std::iota(cells.begin(), cells.end(), 0);

  // Tabulate basis values (derivative order zero) at the quadrature points.
  // Verify the dimensions supplied by Basix before assigning the buffer a
  // type with static derivative, dof and value-component extents.
  const std::array<std::size_t, 4> e_shape
      = e.tabulate_shape(0, weights.size());
  assert((e_shape
          == std::array<std::size_t, 4>{1, weights.size(),
                                        p1_triangle_dofs_per_cell, 1}));
  std::vector<T> phi_b(weights.size() * p1_triangle_dofs_per_cell);
  p1_triangle_basis_t<T> phi(phi_b.data(), weights.size());
  e.tabulate(0, X, phi);

  // Compute |det J_K| from the first two components of the three stored
  // coordinate components. For the affine triangles in this mesh, multiplying
  // by this value maps a reference-cell integral to the physical cell.
  auto detJ = [](mdspan2_t<const T, 3, 3> x)
  {
    return std::abs((x(0, 0) - x(1, 0)) * (x(2, 1) - x(1, 1))
                    - (x(0, 1) - x(1, 1)) * (x(2, 0) - x(1, 0)));
  };

  // Compute the reference mass matrix once. Capturing a statically shaped
  // view avoids recomputing quadrature inside every kernel invocation.
  std::array<T, p1_triangle_dofs_per_cell * p1_triangle_dofs_per_cell> A_hat_b
      = A_ref<T>(phi, weights);
  auto kernel_a
      = [A_hat
         = mdspan2_t<T, p1_triangle_dofs_per_cell, p1_triangle_dofs_per_cell>(
             A_hat_b.data()),
         detJ](T* A, const T*, const T*, const T* x, const int*, const uint8_t*,
               void*)
  {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    mdspan2_t<T, p1_triangle_dofs_per_cell, p1_triangle_dofs_per_cell> _A(A);
    for (std::size_t i = 0; i < A_hat.extent(0); ++i)
      for (std::size_t j = 0; j < A_hat.extent(1); ++j)
        _A(i, j) = scale * A_hat(i, j);
  };

  // The load kernel follows the same pattern, now capturing the reference
  // vector by value. The unused parameters are part of the UFCx kernel ABI.
  auto kernel_L = [b_hat = b_ref<T>(phi, weights),
                   detJ](T* b, const T*, const T*, const T* x, const int*,
                         const uint8_t*, void*)
  {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    for (std::size_t i = 0; i < p1_triangle_dofs_per_cell; ++i)
      b[i] = scale * b_hat[i];
  };

  // Route 1: wrap each hand-written kernel in a Form and use public assembly.
  // The Form stores kernels as type-erased `std::function` objects, so each
  // cell incurs an indirect call that the compiler cannot inline. For small
  // kernels like these the call overhead is significant relative to the work.
  const double norm_A0 = assemble_matrix0<T>(V, kernel_a, cells);
  const double norm_b0 = assemble_vector0<T>(V, kernel_L, cells);

  // Route 2: pass the concrete lambda types to the cell assembly templates.
  // This removes kernel type erasure and permits inlining into the cell loop.
  const double norm_A1
      = assemble_matrix1<T>(mesh->geometry(), *V->dofmap(), kernel_a, cells);
  const double norm_b1
      = assemble_vector1<T>(mesh->geometry(), *V->dofmap(), kernel_L, cells);
  check_norm<T>(norm_A1, norm_A0);
  check_norm<T>(norm_b1, norm_b0);

  // Route 3: repeat the direct assembly using the kernels generated from
  // mass.py. FFCx generates kernels for a single scalar type, float64 by
  // default, so they take `double*` arguments and this route is compiled only
  // for T = double. Float32 kernels would need a second FFCx run with
  // `--scalar_type float32` and different kernel names.
  if constexpr (std::is_same_v<T, double>)
  {
    // The lambdas call the generated functions by the names fixed in mass.py.
    // Because mass.c was included above, the optimiser can see and inline
    // their definitions.
    auto kernel_a_ffcx
        = [](T* A, const T* w, const T* c, const T* coordinate_dofs,
             const int* entity_local_index,
             const uint8_t* quadrature_permutation, void* d)
    {
      tabulate_tensor_mass(A, w, c, coordinate_dofs, entity_local_index,
                           quadrature_permutation, d);
    };
    auto kernel_L_ffcx
        = [](T* b, const T* w, const T* c, const T* coordinate_dofs,
             const int* entity_local_index,
             const uint8_t* quadrature_permutation, void* d)
    {
      tabulate_tensor_load(b, w, c, coordinate_dofs, entity_local_index,
                           quadrature_permutation, d);
    };
    const double norm_A2 = assemble_matrix1<T>(mesh->geometry(), *V->dofmap(),
                                               kernel_a_ffcx, cells);
    const double norm_b2 = assemble_vector1<T>(mesh->geometry(), *V->dofmap(),
                                               kernel_L_ffcx, cells);
    check_norm<T>(norm_A2, norm_A0);
    check_norm<T>(norm_b2, norm_b0);
  }

  // The norm checks establish that the three routes produce the same global
  // tensors. Timings distinguish the Form-based and direct cell loops.
  list_timings(comm);
}

// Run the hand-written routes for both supported real scalar types. The
// generated-kernel route is additionally run for double precision.
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);
  assemble<float>(MPI_COMM_WORLD);
  assemble<double>(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
