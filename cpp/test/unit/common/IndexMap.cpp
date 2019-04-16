// Copyright (C) 2018 Chris N. Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//

#include <catch.hpp>
#include <dolfin.h>
#include <dolfin/common/IndexMap.h>
#include <petscvec.h>

#include <iostream>
#include <sstream>

using namespace dolfin;

namespace
{
void test_scatter()
{
  const std::size_t mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);
  const std::size_t mpi_rank = dolfin::MPI::rank(MPI_COMM_WORLD);

  int nlocal = 100;

  // Create some ghost entries on next process
  int nghost = (mpi_size - 1) * 3;
  std::vector<std::size_t> ghosts(nghost);
  for (int i = 0; i < nghost; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * nlocal + i;

  // Create some local data
  const double val = 1.23;
  std::vector<PetscScalar> data_local(nlocal, val * mpi_rank);
  std::vector<PetscScalar> data_ghost(nghost);

  common::IndexMap idx_map(MPI_COMM_WORLD, nlocal, ghosts, 1);

  idx_map.scatter_fwd(data_local, data_ghost);

  // Check value has been pushed over from other processes
  for (int i = 0; i < nghost; ++i)
  {
    assert(data_ghost[i] == val * ((mpi_rank + 1) % mpi_size));
  }

  // // Send ghost values back to origin
  // idx_map.scatter_rev(data);

  // std::stringstream s;
  // for (int i = 0; i < nghost; ++i)
  // {
  //   assert(data[i] == 2 * val * mpi_rank);
  //   s << mpi_rank << "] " << i << " " << data[i] << "\n";
  // }

  // std::cout << s.str() << "\n";
}
} // namespace

TEST_CASE("Scatter using IndexMap", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter());
}
