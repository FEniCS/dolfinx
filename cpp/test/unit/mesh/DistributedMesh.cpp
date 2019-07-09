// Copyright (C) 2015 Garth N. Wells
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
// First added:  2008-09-30
// Last changed: 2012-08-21
//
// Unit tests for SubSystemsManager

#include <catch.hpp>
#include <cmath>
#include <dolfin.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/PartitionData.h>
#include <dolfin/mesh/Partitioning.h>

using namespace dolfin;

namespace
{
void test_distributed_mesh()
{
  int argc = 0;
  char** argv = nullptr;
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  auto mpi_comm = dolfin::MPI::Comm(MPI_COMM_WORLD);
  int mpi_size = mpi_comm.size();

  // Create sub-communicator
  int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  MPI_Comm subset_comm = dolfin::MPI::SubsetComm(MPI_COMM_WORLD, subset_size);

  // Create mesh using all processes
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 0.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{64, 64}}, mesh::CellType::Type::triangle,
      mesh::GhostMode::none));

  // Save mesh in XDMF format
  io::XDMFFile file(MPI_COMM_WORLD, "mesh.xdmf");
  file.write(*mesh);

  mesh::CellType::Type cell_type;
  EigenRowArrayXXd points;
  EigenRowArrayXXi64 cells;
  std::vector<std::int64_t> global_cell_indices;

  // Save mesh in XDMF format
  io::XDMFFile infile(MPI_COMM_WORLD, "mesh.xdmf");
  std::tie(cell_type, points, cells, global_cell_indices)
      = infile.read_mesh_data(subset_comm);

  // Partition mesh into nparts using local mesh data and subset of
  // communicators
  int nparts = mpi_size;
  std::string partitioner = "SCOTCH";
  mesh::PartitionData cell_partition = mesh::Partitioning::partition_cells(
      subset_comm, nparts, cell_type, cells, partitioner);

  // Build mesh from local mesh data, ghost mode, and provided cell partition
  auto ghost_mode = mesh::GhostMode::none;
  auto new_mesh
      = std::make_shared<mesh::Mesh>(mesh::Partitioning::build_from_partition(
          mpi_comm.comm(), cell_type, cells, points, global_cell_indices,
          ghost_mode, cell_partition));

  CHECK(dolfin::MPI::max(mpi_comm.comm(), mesh->hmax())
        == dolfin::MPI::max(mpi_comm.comm(), new_mesh->hmax()));

  CHECK(dolfin::MPI::min(mpi_comm.comm(), mesh->hmin())
        == dolfin::MPI::min(mpi_comm.comm(), new_mesh->hmin()));
}
} // namespace

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  CHECK_NOTHROW(test_distributed_mesh());
}
