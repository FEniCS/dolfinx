// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::Vector

#include <catch2/catch.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>
#include <xtensor/xtensor.hpp>

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
  const auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::fill(v.mutable_array().begin(), v.mutable_array().end(), 1.0);

  const double norm2 = la::squared_norm(v);
  CHECK(norm2 == mpi_size * size_local);

  std::fill(v.mutable_array().begin(), v.mutable_array().end(), mpi_rank);

  const double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(la::squared_norm(v) == sumn2);
  CHECK(la::norm(v, la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(la::norm(v, la::Norm::linf) == static_cast<T>(mpi_size - 1));
}

} // namespace

TEMPLATE_TEST_CASE("Linear Algebra Vector", "[la_vector]", double,
                   std::complex<double>)
{
  CHECK_NOTHROW(test_vector<TestType>());
}
