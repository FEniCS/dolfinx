// ```text
// Copyright (C) 2024 Jack S. Hale and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Custom cell kernel assembly
//
// This demo shows various methods to define custom cell kernels in C++
// and have them assembled into DOLFINx linear algebra data structures.

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cmath>
#include <concepts>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <map>
#include <stdint.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
template <typename T, std::size_t ndim>
using mdspand_t = md::mdspan<T, md::dextents<std::size_t, ndim>>;
template <typename T, std::size_t n0, std::size_t n1>
using mdspan2_t = md::mdspan<T, std::extents<std::size_t, n0, n1>>;

/// @brief Compute the P1 element mass matrix on the reference cell.
/// @tparam T Scalar type.
/// @param phi Basis functions.
/// @param w Integration weights.
/// @return Element reference matrix (row-major storage).
template <typename T>
std::array<T, 9> A_ref(mdspand_t<const T, 4> phi, std::span<const T> w)
{
  std::array<T, 9> A_b{};
  mdspan2_t<T, 3, 3> A(A_b.data());
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
std::array<T, 3> b_ref(mdspand_t<const T, 4> phi, std::span<const T> w)
{
  std::array<T, 3> b{};
  for (std::size_t k = 0; k < phi.extent(1); ++k) // quadrature point
    for (std::size_t i = 0; i < b.size(); ++i)    // row i
      b[i] += w[k] * phi(0, k, i, 0);
  return b;
}

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
      std::pair{std::tuple{fem::IntegralType::cell, -1, 0},
                fem::integral_data<T>(kernel, cells, std::vector<int>{})}};

  fem::Form<T, T> a({V, V}, integrals, V->mesh(), {}, {}, false, {});
  auto dofmap = V->dofmap();
  auto sp = la::SparsityPattern(
      V->mesh()->comm(), {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sp, {cells, cells}, {*dofmap, *dofmap});
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
      std::pair{std::tuple{fem::IntegralType::cell, -1, 0},
                fem::integral_data<T>(kernel, cells, std::vector<int>{})}};
  fem::Form<T> L({V}, integrals, mesh, {}, {}, false, {});
  auto dofmap = V->dofmap();
  la::Vector<T> b(dofmap->index_map, 1);
  common::Timer timer("Assembler0 std::function (vector)");
  fem::assemble_vector(b.mutable_array(), L);
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

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
  fem::sparsitybuild::cells(sp, {cells, cells}, {dofmap, dofmap});
  sp.finalize();
  la::MatrixCSR<T> A(sp);
  auto ident = [](auto, auto, auto, auto) {}; // DOF permutation not required
  common::Timer timer("Assembler1 lambda (matrix)");
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>> x(
      g.x().data(), g.x().size() / 3, 3);
  fem::impl::assemble_cells<T>(
      A.mat_add_values(), g.dofmap(), x, cells, {dofmap.map(), 1, cells}, ident,
      {dofmap.map(), 1, cells}, ident, {}, {}, kernel, {}, {}, {}, {});
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using using a lambda kernel function.
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
  md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>> x(
      g.x().data(), g.x().size() / 3, 3);
  common::Timer timer("Assembler1 lambda (vector)");
  fem::impl::assemble_cells<T, 1>([](auto, auto, auto, auto) {},
                                  b.mutable_array(), g.dofmap(), x, cells,
                                  {dofmap.map(), 1, cells}, kernel, {}, {}, {});
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

/// @brief Assemble P1 mass matrix and a RHS vector using element kernel
/// approaches.
///
/// Function demonstrates how hand-coded element kernels can be executed
/// in assembly over cells.
///
/// @tparam T Scalar type.
/// @param comm MPI communicator to assembler over.
template <std::floating_point T>
void assemble(MPI_Comm comm)
{
  // Create mesh
  auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_rectangle<T>(
      comm, {{{0, 0}, {1, 1}}}, {516, 116}, mesh::CellType::triangle));

  // Create Basix P1 Lagrange element. This will be used to construct
  // basis functions inside the custom cell kernel.
  constexpr int order = 1;
  basix::FiniteElement e = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), order,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // Construct quadrature rule
  constexpr int max_degree = 2 * order;
  auto quadrature_type = basix::quadrature::get_default_rule(
      basix::cell::type::triangle, max_degree);
  auto [X_b, weights] = basix::quadrature::make_quadrature<T>(
      quadrature_type, basix::cell::type::triangle,
      basix::polyset::type::standard, max_degree);
  mdspand_t<const T, 2> X(X_b.data(), weights.size(), 2);

  // Create a scalar function space
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
      mesh, std::make_shared<fem::FiniteElement<T>>(e)));

  // Build list of cells to assembler over (all cells owned by this
  // rank)
  std::int32_t size_local
      = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
  std::vector<std::int32_t> cells(size_local);
  std::iota(cells.begin(), cells.end(), 0);

  // Tabulate basis functions at quadrature points
  auto e_shape = e.tabulate_shape(0, weights.size());
  std::size_t length
      = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
  std::vector<T> phi_b(length);
  mdspand_t<T, 4> phi(phi_b.data(), e_shape);
  e.tabulate(0, X, phi);

  // Utility function to compute det(J) for an affine triangle cell
  // (geometry is 3D)
  auto detJ = [](mdspan2_t<const T, 3, 3> x)
  {
    return std::abs((x(0, 0) - x(1, 0)) * (x(2, 1) - x(1, 1))
                    - (x(0, 1) - x(1, 1)) * (x(2, 0) - x(1, 0)));
  };

  // Finite element mass matrix kernel function
  std::array<T, 9> A_hat_b = A_ref<T>(phi, weights);
  auto kernel_a
      = [A_hat = mdspan2_t<T, 3, 3>(A_hat_b.data()),
         detJ](T* A, const T*, const T*, const T* x, const int*, const uint8_t*)
  {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    mdspan2_t<T, 3, 3> _A(A);
    for (std::size_t i = 0; i < A_hat.extent(0); ++i)
      for (std::size_t j = 0; j < A_hat.extent(1); ++j)
        _A(i, j) = scale * A_hat(i, j);
  };

  // Finite element RHS (f=1) kernel function
  auto kernel_L
      = [b_hat = b_ref<T>(phi, weights),
         detJ](T* b, const T*, const T*, const T* x, const int*, const uint8_t*)
  {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    for (std::size_t i = 0; i < 3; ++i)
      b[i] = scale * b_hat[i];
  };

  // Assemble matrix and vector using std::function kernel
  assemble_matrix0<T>(V, kernel_a, cells);
  assemble_vector0<T>(V, kernel_L, cells);

  // Assemble matrix and vector using lambda kernel. This version
  // supports efficient inlining of the kernel in the assembler. This
  // can give a significant performance improvement for lightweight
  // kernels.
  assemble_matrix1<T>(mesh->geometry(), *V->dofmap(), kernel_a, cells);
  assemble_vector1<T>(mesh->geometry(), *V->dofmap(), kernel_L, cells);

  list_timings(comm);
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);
  assemble<float>(MPI_COMM_WORLD);
  assemble<double>(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
