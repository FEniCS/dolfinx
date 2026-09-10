// Copyright (C) 2021-2026 Chris Richardson, Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::Vector

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/la/Vector.h>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

using namespace dolfinx;

namespace
{
template <typename T>
void test_vector()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  const std::vector<int> global_ghost_owner(ghosts.size(),
                                            (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::ranges::fill(v.array(), 1.0);

  double norm2 = la::squared_norm(v);
  CHECK(norm2 == mpi_size * size_local);

  std::ranges::fill(v.array(), mpi_rank);

  double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(la::squared_norm(v) == sumn2);
  CHECK(la::norm(v, la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(la::norm(v, la::Norm::linf) == static_cast<T>(mpi_size - 1));
}

void test_vector_cast()
{
  using T = double;
  using U = float;

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::ranges::fill(v.array(), 1);

  la::Vector<U, std::vector<U>, std::vector<std::int64_t>> v1(v);

  U norm2 = la::squared_norm(v1);
  CHECK(norm2 == mpi_size * size_local);

  std::ranges::fill(v1.array(), mpi_rank);

  U sumn2 = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(la::squared_norm(v1) == sumn2);
  CHECK(la::norm(v1, la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v1, v1) == sumn2);
  CHECK(la::norm(v1, la::Norm::linf) == static_cast<U>(mpi_size - 1));
}

void test_vector_scatter_rev()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  const std::vector<int> global_ghost_owner(ghosts.size(),
                                            (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<double> v(index_map, 1);
  std::ranges::fill(v.array(), 0.0);
  std::fill(std::next(v.array().begin(), size_local), v.array().end(), 2.0);

  // Scatter ghost values to owning ranks and accumulate into owned
  // entries via Vector::get_unpack_op
  v.scatter_rev(std::plus<>{});
  const double sum0 = std::reduce(v.array().begin(),
                                  std::next(v.array().begin(), size_local));
  CHECK(sum0 == 2.0 * num_ghosts);

  // Repeat, to check accumulation onto the non-zero values from the
  // first scatter, rather than overwriting them
  v.scatter_rev(std::plus<>{});
  const double sum1 = std::reduce(v.array().begin(),
                                  std::next(v.array().begin(), size_local));
  CHECK(sum1 == 2 * sum0);
}
void test_vector_shared_scatterer()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  constexpr int bs = 3;

  // Ghost entries owned by the next process
  const int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;
  const std::vector<int> owners(ghosts.size(), (mpi_rank + 1) % mpi_size);
  auto map = std::make_shared<const common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, owners);

  auto sct = std::make_shared<const common::Scatterer<>>(*map, bs);
  CHECK(sct.use_count() == 1);
  {
    // One scatterer, two vectors, two scalar types
    la::Vector<double> u(map, bs, sct);
    la::Vector<std::int8_t> v(map, bs, sct);
    CHECK(sct.use_count() == 3);
    CHECK(u.index_map() == map);
    CHECK(u.bs() == bs);

    std::ranges::fill_n(u.array().begin(), bs * size_local,
                        static_cast<double>(mpi_rank));
    u.scatter_fwd();
    const double owner = static_cast<double>((mpi_rank + 1) % mpi_size);
    for (int i = 0; i < bs * num_ghosts; ++i)
      CHECK(u.array()[bs * size_local + i] == owner);

    std::ranges::fill_n(v.array().begin(), bs * size_local, std::int8_t(7));
    v.scatter_fwd();
    for (int i = 0; i < bs * num_ghosts; ++i)
      CHECK(v.array()[bs * size_local + i] == std::int8_t(7));
  }
  CHECK(sct.use_count() == 1);
}
} // namespace

TEMPLATE_TEST_CASE("Linear Algebra Vector", "[la_vector]", double,
                   std::complex<double>)
{
  CHECK_NOTHROW(test_vector<TestType>());
}

TEST_CASE("Linear Algebra Vector", "[la_vector]")
{
  CHECK_NOTHROW(test_vector_cast());
}

TEST_CASE("Linear Algebra Vector scatter reverse", "[la_vector]")
{
  CHECK_NOTHROW(test_vector_scatter_rev());
}

TEST_CASE("Linear Algebra Vector shared scatterer", "[la_vector]")
{
  CHECK_NOTHROW(test_vector_shared_scatterer());
}
