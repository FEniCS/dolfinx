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
  std::vector<int> src_ranks = global_ghost_owner;
  std::sort(src_ranks.begin(), src_ranks.end());
  src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                  src_ranks.end());
  auto dest_ranks
      = dolfinx::MPI::compute_graph_edges_nbx(MPI_COMM_WORLD, src_ranks);
  const auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, dest_ranks, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::fill(v.mutable_array().begin(), v.mutable_array().end(), 1.0);

  const double norm2 = v.squared_norm();
  CHECK(norm2 == mpi_size * size_local);

  std::fill(v.mutable_array().begin(), v.mutable_array().end(), mpi_rank);

  const double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(v.squared_norm() == sumn2);
  CHECK(v.norm(la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(v.norm(la::Norm::linf) == static_cast<T>(mpi_size - 1));
}

} // namespace

TEMPLATE_TEST_CASE("Linear Algebra Vector", "[la_vector]", double,
                   std::complex<double>)
{
  CHECK_NOTHROW(test_vector<TestType>());
}
