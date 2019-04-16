// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <vector>

using namespace dolfin;

namespace
{
void test_scatter()
{
  // const std::size_t mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);
  // const std::size_t mpi_rank = dolfin::MPI::rank(MPI_COMM_WORLD);

  // int nlocal = 100;

  // // Create some ghost entries on next process
  // int nghost = (mpi_size - 1) * 3;
  // std::vector<std::size_t> ghosts(nghost);
  // for (int i = 0; i < nghost; ++i)
  //   ghosts[i] = (mpi_rank + 1) % mpi_size * nlocal + i;

  // // Create some local data
  // const std::int64_t val = 11;
  // std::vector<std::int64_t> data_local(nlocal, val * mpi_rank);
  // std::vector<std::int64_t> data_ghost(nghost);

  // common::IndexMap idx_map(MPI_COMM_WORLD, nlocal, ghosts, 1);

  // // Check value has been pushed over from other processes
  // idx_map.scatter_fwd(data_local, data_ghost);
  // for (std::size_t i = 0; i < data_ghost.size(); ++i)
  // {
  //   if (data_ghost[i] == val * ((mpi_rank + 1) % mpi_size))
  //     continue;
  //   else
  //     throw std::runtime_error("Received data incorrect.");
  // }

  // std::vector<PetscScalar> data;
  // data.insert(data.end(), data_local.begin(), data_local.end());
  // data.insert(data.end(), data_ghost.begin(), data_ghost.end());

  // // Send ghost values back to origin
  // idx_map.scatter_rev(data);
  // for (int i = 0; i < nghost; ++i)
  // {
  //   if (data[i] == 2 * val * mpi_rank)
  //     continue;
  //   else
  //     throw std::runtime_error("Received unexpected data (2).");
  // }

} // namespace
} // namespace

TEST_CASE("Scatter using IndexMap", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter());
}
