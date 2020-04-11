// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <catch.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>

using namespace dolfinx;

namespace
{
void test_distributed_mesh()
{
  // auto mpi_comm = dolfinx::MPI::Comm(MPI_COMM_WORLD);
  // int mpi_size = dolfinx::MPI::size(mpi_comm.comm());

  // // Create sub-communicator
  // int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  // MPI_Comm subset_comm = dolfinx::MPI::SubsetComm(MPI_COMM_WORLD, subset_size);

  // // Create mesh using all processes
  // std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
  //                                   Eigen::Vector3d(1.0, 1.0, 0.0)};
  // auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
  //     MPI_COMM_WORLD, pt, {{64, 64}}, mesh::CellType::triangle,
  //     mesh::GhostMode::none));

  // int dim = mesh->geometry().dim();

  // // Save mesh in XDMF format
  // io::XDMFFile file(MPI_COMM_WORLD, "mesh.xdmf");
  // file.write(*mesh);

  // // Read in mesh in mesh data from XDMF file
  // io::XDMFFile infile(MPI_COMM_WORLD, "mesh.xdmf");
  // const auto [cell_type, points, cells, global_cell_indices]
  //     = infile.read_mesh_data(subset_comm);

  // // Partition mesh into nparts using local mesh data and subset of
  // // communicators
  // int nparts = mpi_size;
  // mesh::GhostMode ghost_mode = mesh::GhostMode::none;
  // mesh::PartitionData cell_partition = mesh::Partitioning::partition_cells(
  //     subset_comm, nparts, cell_type, cells, mesh::Partitioner::scotch,
  //     ghost_mode);

  // // Build mesh from local mesh data, ghost mode, and provided cell partition
  // auto new_mesh
  //     = std::make_shared<mesh::Mesh>(mesh::Partitioning::build_from_partition(
  //         mpi_comm.comm(), cell_type, points, cells, global_cell_indices,
  //         ghost_mode, cell_partition));

  // // Check mesh features
  // CHECK(dolfinx::MPI::max(mpi_comm.comm(), mesh->hmax())
  //       == dolfinx::MPI::max(mpi_comm.comm(), new_mesh->hmax()));

  // CHECK(dolfinx::MPI::min(mpi_comm.comm(), mesh->hmin())
  //       == dolfinx::MPI::min(mpi_comm.comm(), new_mesh->hmin()));

  // CHECK(mesh->num_entities_global(0) == new_mesh->num_entities_global(0));

  // CHECK(mesh->num_entities_global(dim) == new_mesh->num_entities_global(dim));
}
} // namespace

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  // CHECK_NOTHROW(test_distributed_mesh());
}
