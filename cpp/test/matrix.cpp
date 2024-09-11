// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::MatrixCSR

#include "poisson.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <mpi.h>
#include <span>

using namespace dolfinx;

namespace
{
/// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] x Output vector
template <typename T>
void spmv_impl(std::span<const T> values,
               std::span<const std::int64_t> row_begin,
               std::span<const std::int64_t> row_end,
               std::span<const std::int32_t> indices, std::span<const T> x,
               std::span<T> y)
{
  assert(row_begin.size() == row_end.size());
  for (std::size_t i = 0; i < row_begin.size(); i++)
  {
    double vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

// The matrix A is distributed across P  processes by blocks of rows:
//  A = |   A_0  |
//      |   A_1  |
//      |   ...  |
//      |  A_P-1 |
//
// Each submatrix A_i is owned by a single process "i" and can be further
// decomposed into diagonal (Ai[0]) and off diagonal (Ai[1]) blocks:
//  Ai = |Ai[0] Ai[1]|
//
// If A is square, the diagonal block Ai[0] is also square and contains
// only owned columns and rows. The block Ai[1] contains ghost columns
// (unowned dofs).

// Likewise, a local vector x can be decomposed into owned and ghost blocks:
// xi = |   x[0]  |
//      |   x[1]  |
//
// So the product y = Ax can be computed into two separate steps:
//  y[0] = |Ai[0] Ai[1]| |   x[0]  | = Ai[0] x[0] + Ai[1] x[1]
//                       |   x[1]  |
//
/// Computes y += A*x for a parallel CSR matrix A and parallel dense vectors x,y
/// @param[in] A Parallel CSR matrix
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
void spmv(la::MatrixCSR<T>& A, la::Vector<T>& x, la::Vector<T>& y)
{
  // start communication (update ghosts)
  x.scatter_fwd_begin();

  const std::int32_t nrowslocal = A.num_owned_rows();
  std::span<const std::int64_t> row_ptr(A.row_ptr().data(), nrowslocal + 1);
  std::span<const std::int32_t> cols(A.cols().data(), row_ptr[nrowslocal]);
  std::span<const std::int64_t> off_diag_offset(A.off_diag_offset().data(),
                                                nrowslocal);
  std::span<const T> values(A.values().data(), row_ptr[nrowslocal]);

  std::span<const T> _x = x.array();
  std::span<T> _y = y.mutable_array();

  std::span<const std::int64_t> row_begin(row_ptr.data(), nrowslocal);
  std::span<const std::int64_t> row_end(row_ptr.data() + 1, nrowslocal);

  // First stage:  spmv - diagonal
  // yi[0] += Ai[0] * xi[0]
  spmv_impl<T>(values, row_begin, off_diag_offset, cols, _x, _y);

  // finalize ghost update
  x.scatter_fwd_end();

  // Second stage:  spmv - off-diagonal
  // yi[0] += Ai[1] * xi[1]
  spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
}

/// @brief Create a matrix operator
/// @param comm The communicator to builf the matrix on
/// @return The assembled matrix
la::MatrixCSR<double> create_operator(MPI_Comm comm)
{
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {12, 12, 12},
                       mesh::CellType::tetrahedron, part));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, 2,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(mesh, element, {}));

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<fem::Constant<double>>(2.0);
  auto a = std::make_shared<fem::Form<double, double>>(
      fem::create_form<double, double>(*form_poisson_a, {V, V}, {},
                                       {{"kappa", kappa}}, {}, {}));

  la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
  sp.finalize();
  la::MatrixCSR<double> A(sp);
  fem::assemble_matrix(A.mat_add_values(), *a, {});
  A.scatter_rev();

  return A;
}

[[maybe_unused]] void test_matrix_norm()
{
  la::MatrixCSR A0 = create_operator(MPI_COMM_SELF);
  la::MatrixCSR A1 = create_operator(MPI_COMM_WORLD);
  CHECK(A1.squared_norm() == Catch::Approx(A0.squared_norm()).epsilon(1e-8));
}

[[maybe_unused]] void test_matrix_apply()
{
  MPI_Comm comm = MPI_COMM_WORLD;
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {12, 12, 12},
                       mesh::CellType::tetrahedron, part));

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, 2,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(mesh, element, {}));

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<fem::Constant<double>>(2.0);
  auto ui = std::make_shared<fem::Function<double, double>>(V);

  // Define variational forms
  auto a = std::make_shared<fem::Form<double, double>>(
      fem::create_form<double, double>(*form_poisson_a, {V, V}, {},
                                       {{"kappa", kappa}}, {}, {}));

  // Create sparsity pattern
  la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
  sp.finalize();

  // Assemble matrix
  la::MatrixCSR<double> A(sp);
  fem::assemble_matrix(A.mat_add_values(), *a, {});
  A.scatter_rev();
  CHECK((V->dofmap()->index_map->size_local() == A.num_owned_rows()));

  // Get compatible vectors
  auto col_map = A.index_map(1);

  la::Vector<double> x(col_map, 1);
  la::Vector<double> y(col_map, 1);

  std::size_t col_size = col_map->size_local() + col_map->num_ghosts();
  CHECK(x.array().size() == col_size);

  // Fill x vector with 1 (Constant)
  std::ranges::fill(x.mutable_array(), 1);

  // Matrix A represents the action of the Laplace operator, so when
  // applied to a constant vector the result should be zero
  spmv(A, x, y);

  std::ranges::for_each(y.array(),
                        [](auto a) { REQUIRE(std::abs(a) < 1e-13); });
}

void test_matrix()
{
  auto map0 = std::make_shared<common::IndexMap>(MPI_COMM_WORLD, 8);
  la::SparsityPattern p(MPI_COMM_WORLD, {map0, map0}, {1, 1});
  p.insert(0, 0);
  p.insert(4, 5);
  p.insert(5, 4);
  p.finalize();

  using T = float;
  la::MatrixCSR<T> A(p);
  A.add(std::vector<decltype(A)::value_type>{1}, std::vector{0},
        std::vector{0});
  A.add(std::vector<decltype(A)::value_type>{2.3}, std::vector{4},
        std::vector{5});

  const std::vector Adense0 = A.to_dense();

  // Note: we cut off the ghost rows by intent here! But therefore we are not
  // able to work with the dimensions of Adense0 to compute indices, these
  // contain the ghost rows, which also vary between processes.
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T,
      MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
          std::size_t, 8, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
      Adense(Adense0.data(), 8, A.index_map(1)->size_global());

  std::vector<T> Aref_data(8 * A.index_map(1)->size_global(), 0);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
             std::size_t, 8, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
      Aref(Aref_data.data(), 8, A.index_map(1)->size_global());

  auto to_global_col = [&](auto col)
  {
    std::array<std::int64_t, 1> tmp;
    A.index_map(1)->local_to_global(std::vector<std::int32_t>{col}, tmp);
    return tmp[0];
  };
  Aref(0, to_global_col(0)) = 1;
  Aref(4, to_global_col(5)) = 2.3;

  for (std::size_t i = 0; i < Adense.extent(0); ++i)
    for (std::size_t j = 0; j < Adense.extent(1); ++j)
      CHECK(Adense(i, j) == Aref(i, j));

  Aref(4, to_global_col(4)) = 2.3;
  CHECK(Adense(4, to_global_col(4)) != Aref(4, to_global_col(4)));
}

} // namespace

TEST_CASE("Linear Algebra CSR Matrix", "[la_matrix]")
{
  CHECK_NOTHROW(test_matrix());
  CHECK_NOTHROW(test_matrix_apply());
  CHECK_NOTHROW(test_matrix_norm());
}
