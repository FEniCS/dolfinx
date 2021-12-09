// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <basix/finite-element.h>
#include <catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/graphbuild.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{

constexpr int N = 4;

void create_mesh_file()
{
  // Create mesh using all processes and save xdmf
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}}, {N, N},
      mesh::CellType::triangle, mesh::GhostMode::shared_facet));

  // Save mesh in XDMF format
  io::XDMFFile file(MPI_COMM_WORLD, "mesh.xdmf", "w");
  file.write_mesh(*mesh);
}

void test_distributed_mesh(mesh::CellPartitionFunction partitioner)
{
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  const int mpi_size = dolfinx::MPI::size(mpi_comm);

  // Create a communicator with subset of the original group of processes
  const int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  std::vector<int> ranks(subset_size);
  std::iota(ranks.begin(), ranks.end(), 0);

  MPI_Group comm_group;
  MPI_Comm_group(mpi_comm, &comm_group);

  MPI_Group new_group;
  MPI_Group_incl(comm_group, subset_size, ranks.data(), &new_group);

  MPI_Comm subset_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &subset_comm);

  // Create coordinate map
  auto e = std::make_shared<basix::FiniteElement>(basix::create_element(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::equispaced, false));
  fem::CoordinateElement cmap(e);

  // read mesh data
  xt::xtensor<double, 2> x({0, 3});
  xt::xtensor<std::int64_t, 2> cells(
      {0, static_cast<std::size_t>(
              dolfinx::mesh::num_cell_vertices(mesh::CellType::triangle))});
  graph::AdjacencyList<std::int32_t> dest(0);
  if (subset_comm != MPI_COMM_NULL)
  {
    int nparts = mpi_size;
    io::XDMFFile infile(subset_comm, "mesh.xdmf", "r");
    cells = infile.read_topology_data("mesh");
    x = infile.read_geometry_data("mesh");
    auto [data, offsets] = graph::create_adjacency_data(cells);
    const int tdim = mesh::cell_dim(mesh::CellType::triangle);
    dest = partitioner(
        subset_comm, nparts, tdim,
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

  CHECK(mesh->topology().index_map(tdim)->size_global() == 2 * N * N);
  CHECK(mesh->topology().index_map(tdim)->size_local() > 0);

  CHECK(mesh->topology().index_map(0)->size_global() == (N + 1) * (N + 1));
  CHECK(mesh->topology().index_map(0)->size_local() > 0);

  CHECK(mesh->geometry().x().shape(0)
        == mesh->topology().index_map(0)->size_local()
               + mesh->topology().index_map(0)->num_ghosts());

  MPI_Comm_free(&subset_comm);
  MPI_Group_free(&new_group);
  MPI_Group_free(&comm_group);
}
} // namespace

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  create_mesh_file();

  SECTION("SCOTCH")
  {
    CHECK_NOTHROW(test_distributed_mesh(create_cell_partitioner()));
  }

#ifdef HAS_KAHIP
  SECTION("KAHIP with Lambda")
  {
    auto partfn = graph::kahip::partitioner();

    CellPartitionFunction kahip
        = [&](MPI_Comm comm, int nparts, int tdim,
              const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
              dolfinx::mesh::GhostMode ghost_mode)
    {
      LOG(INFO) << "Compute partition of cells across ranks (KaHIP).";
      // Compute distributed dual graph (for the cells on this process)
      const auto [dual_graph, num_ghost_edges]
          = mesh::build_dual_graph(comm, cells, tdim);

      // Just flag any kind of ghosting for now
      bool ghosting = (ghost_mode != mesh::GhostMode::none);

      // Compute partition
      return partfn(comm, nparts, dual_graph, num_ghost_edges, ghosting);
    };

    CHECK_NOTHROW(test_distributed_mesh(kahip));
  }
#endif
}
