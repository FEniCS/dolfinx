// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::MatrixCSR

#include "poisson.h"
#include <catch2/catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

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
void spmv_impl(xtl::span<const T> values,
               xtl::span<const std::int32_t> row_begin,
               xtl::span<const std::int32_t> row_end,
               xtl::span<const std::int32_t> indices, xtl::span<const T> x,
               xtl::span<T> y)
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
// If A is square, the diagonal block Ai[0] is also square and countains
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
  xtl::span<const std::int32_t> row_ptr(A.row_ptr().data(), nrowslocal + 1);
  xtl::span<const std::int32_t> cols(A.cols().data(), row_ptr[nrowslocal]);
  xtl::span<const std::int32_t> off_diag_offset(A.off_diag_offset().data(),
                                                nrowslocal);
  xtl::span<const T> values(A.values().data(), row_ptr[nrowslocal]);

  xtl::span<const T> _x = x.array();
  xtl::span<T> _y = y.mutable_array();

  xtl::span<const std::int32_t> row_begin(row_ptr.data(), nrowslocal);
  xtl::span<const std::int32_t> row_end(row_ptr.data() + 1, nrowslocal);

  // First stage:  spmv - diagonal
  // yi[0] += Ai[0] * xi[0]
  spmv_impl<T>(values, row_begin, off_diag_offset, cols, _x, _y);

  // finalize ghost update
  x.scatter_fwd_end();

  // Second stage:  spmv - off-diagonal
  // yi[0] += Ai[1] * xi[1]
  spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
};

void test_matrix_apply()
{
  MPI_Comm comm = MPI_COMM_WORLD;
  auto mesh = std::make_shared<mesh::Mesh>(
      mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {12, 12, 12},
                       mesh::CellType::tetrahedron, mesh::GhostMode::none));

  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<fem::Constant<double>>(2.0);
  auto ui = std::make_shared<fem::Function<double>>(V);

  // Define variational forms
  auto a = std::make_shared<fem::Form<double>>(fem::create_form<double>(
      *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));

  // Create sparsity pattern
  la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
  sp.assemble();

  // Assemble matrix
  la::MatrixCSR<double> A(sp);
  fem::assemble_matrix(A.mat_add_values(), *a, {});
  A.finalize();

  CHECK((V->dofmap()->index_map->size_local() == A.num_owned_rows()));

  // Get compatible vectors
  auto maps = A.index_maps();

  la::Vector<double> x(maps[1], 1);
  la::Vector<double> y(maps[1], 1);

  std::size_t col_size = maps[1]->size_local() + maps[1]->num_ghosts();
  CHECK(x.array().size() == col_size);

  // Fill x vector with 1 (Constant)
  std::fill(x.mutable_array().begin(), x.mutable_array().end(), 1);

  // Matrix A represents the action of the Laplace operator, so when applied to
  // a constant vector the result should be zero
  spmv(A, x, y);

  std::for_each(y.array().begin(), y.array().end(),
                [](auto a) { REQUIRE(std::abs(a) < 1e-13); });
}

void test_matrix()
{
  auto map0 = std::make_shared<common::IndexMap>(MPI_COMM_SELF, 8);
  la::SparsityPattern p(MPI_COMM_SELF, {map0, map0}, {1, 1});
  p.insert(std::vector{0}, std::vector{0});
  p.insert(std::vector{4}, std::vector{5});
  p.insert(std::vector{5}, std::vector{4});
  p.assemble();

  la::MatrixCSR<float> A(p);
  A.add(std::vector<decltype(A)::value_type>{1}, std::vector{0},
        std::vector{0});
  A.add(std::vector<decltype(A)::value_type>{2.3}, std::vector{4},
        std::vector{5});

  const std::vector Adense0 = A.to_dense();
  auto Adense = xt::adapt(Adense0, {8, 8});

  xt::xtensor<float, 2> Aref = xt::zeros<float>({8, 8});
  Aref(0, 0) = 1;
  Aref(4, 5) = 2.3;
  CHECK((Adense == Aref));

  Aref(4, 4) = 2.3;
  CHECK((Adense != Aref));
}

} // namespace

TEST_CASE("Linear Algebra CSR Matrix", "[la_matrix]")
{
  CHECK_NOTHROW(test_matrix());
  CHECK_NOTHROW(test_matrix_apply());
}