// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include "cmap.h"
#include <catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/kahip.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/graphbuild.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{

void create_mesh_file()
{
  // Create mesh using all processes and save xdmf
  auto cmap = fem::create_coordinate_map(create_coordinate_map_cmap);
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}}, {32, 32}, cmap,
      mesh::GhostMode::shared_facet));

  // Save mesh in XDMF format
  io::XDMFFile file(MPI_COMM_WORLD, "mesh.xdmf", "w");
  file.write_mesh(*mesh);
}

void test_distributed_mesh(mesh::CellPartitionFunction partitioner)
{
  MPI_Comm mpi_comm{MPI_COMM_WORLD};
  int mpi_size = dolfinx::MPI::size(mpi_comm);

  // Create a communicator with subset of the original group of processes
  int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  std::vector<int> ranks(subset_size);
  std::iota(ranks.begin(), ranks.end(), 0);

  MPI_Group comm_group;
  MPI_Comm_group(mpi_comm, &comm_group);

  MPI_Group new_group;
  MPI_Group_incl(comm_group, subset_size, ranks.data(), &new_group);

  MPI_Comm subset_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &subset_comm);

  // Create coordinate map
  auto cmap = fem::create_coordinate_map(create_coordinate_map_cmap);

  // read mesh data
  Eigen::Array<double, -1, -1, Eigen::RowMajor> x(0, 3);
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells(0, dolfinx::mesh::num_cell_vertices(cmap.cell_shape()));
  graph::AdjacencyList<std::int32_t> dest(0);

  if (subset_comm != MPI_COMM_NULL)
  {
    int nparts{mpi_size};
    io::XDMFFile infile(subset_comm, "mesh.xdmf", "r");
    cells = infile.read_topology_data("mesh");
    x = infile.read_geometry_data("mesh");
    auto [data, offsets] = graph::create_adjacency_data(cells);
    dest = partitioner(
        subset_comm, nparts, cmap.cell_shape(),
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offsets)),
        mesh::GhostMode::shared_facet);
  }

  auto [data, offsets] = graph::create_adjacency_data(cells);
  graph::AdjacencyList<std::int64_t> cells_topology(std::move(data),
                                                    std::move(offsets));

  // Distribute cells to destination ranks
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::build::distribute(mpi_comm, cells_topology, dest);

  dolfinx::mesh::Topology topology = mesh::create_topology(
      mpi_comm, cell_nodes, original_cell_index, ghost_owners,
      cmap.cell_shape(), mesh::GhostMode::shared_facet);
  int tdim = topology.dim();
  dolfinx::mesh::Geometry geometry
      = mesh::create_geometry(mpi_comm, topology, cmap, cell_nodes, x);

  auto mesh = std::make_shared<dolfinx::mesh::Mesh>(
      mpi_comm, std::move(topology), std::move(geometry));

  CHECK(mesh->topology().index_map(tdim)->size_global() == 2048);
  CHECK(mesh->topology().index_map(tdim)->size_local() > 0);

  CHECK(mesh->topology().index_map(0)->size_global() == 1089);
  CHECK(mesh->topology().index_map(0)->size_local() > 0);

  CHECK(mesh->geometry().x().shape[0]
        == mesh->topology().index_map(0)->size_local()
               + mesh->topology().index_map(0)->num_ghosts());
}
} // namespace

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  create_mesh_file();

  SECTION("SCOTCH")
  {
    CHECK_NOTHROW(test_distributed_mesh(
        static_cast<graph::AdjacencyList<std::int32_t> (*)(
            MPI_Comm, int, const mesh::CellType,
            const graph::AdjacencyList<std::int64_t>&, mesh::GhostMode)>(
            &mesh::partition_cells_graph)));
  }

#ifdef HASKIP
  SECTION("KAHIP with Lambda")
  {
    auto kahip
        = [](MPI_Comm mpi_comm, int nparts, const mesh::CellType cell_type,
             const graph::AdjacencyList<std::int64_t>& cells,
             mesh::GhostMode ghost_mode) {
            const auto [dual_graph, graph_info]
                = mesh::build_dual_graph(mpi_comm, cells, cell_type);
            bool ghosting = (ghost_mode != mesh::GhostMode::none);
            return graph::kahip::partition(mpi_comm, nparts, dual_graph, -1,
                                           ghosting);
          };
    CHECK_NOTHROW(test_distributed_mesh(kahip));
  }
#endif
}
