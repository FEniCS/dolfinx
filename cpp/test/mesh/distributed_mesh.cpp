// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <basix/finite-element.h>
#include <catch2/catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/graphbuild.h>

using namespace dolfinx;

namespace
{

constexpr int N = 8;

void create_mesh_file()
{
  // Create mesh using all processes and save xdmf
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh>(
      mesh::create_rectangle(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {N, N},
                             mesh::CellType::triangle, part));

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
      basix::element::family::P, basix::cell::type::triangle, 1, false));
  fem::CoordinateElement cmap(e);

  // read mesh data
  std::vector<double> x;
  std::array<std::size_t, 2> xshape = {0, 2};
  std::vector<std::int64_t> cells;
  std::array<std::size_t, 2> cshape = {0, 3};
  graph::AdjacencyList<std::int32_t> dest(0);
  if (subset_comm != MPI_COMM_NULL)
  {
    io::XDMFFile infile(subset_comm, "mesh.xdmf", "r");
    std::tie(cells, cshape) = infile.read_topology_data("mesh");
    std::tie(x, xshape) = infile.read_geometry_data("mesh");

    int nparts = mpi_size;
    const int tdim = mesh::cell_dim(mesh::CellType::triangle);
    dest = partitioner(subset_comm, nparts, tdim,
                       graph::regular_adjacency_list(cells, cshape[1]));
  }
  CHECK(xshape[1] == 2);

  // Distribute cells to destination ranks
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::build::distribute(
          mpi_comm, graph::regular_adjacency_list(cells, cshape[1]), dest);

  // FIXME: improve way to find 'external' vertices
  // Count the connections of all vertices on owned cells. If there are 6
  // connections (on a regular triangular mesh) then it is 'internal'.
  int num_local_cells = cell_nodes.num_nodes() - ghost_owners.size();
  int ghost_offset = cell_nodes.offsets()[num_local_cells];
  std::vector<std::int64_t> external_vertices(
      cell_nodes.array().begin(), cell_nodes.array().begin() + ghost_offset);
  std::sort(external_vertices.begin(), external_vertices.end());
  std::vector<int> counts;
  auto it = external_vertices.begin();
  while (it != external_vertices.end())
  {
    auto it2 = std::find_if(it, external_vertices.end(),
                            [&](std::int64_t val) { return (val != *it); });
    counts.push_back(std::distance(it, it2));
    it = it2;
  }
  external_vertices.erase(
      std::unique(external_vertices.begin(), external_vertices.end()),
      external_vertices.end());
  for (std::size_t i = 0; i < counts.size(); ++i)
    if (counts[i] == 6)
      external_vertices[i] = -1;
  std::sort(external_vertices.begin(), external_vertices.end());
  it = std::find_if(external_vertices.begin(), external_vertices.end(),
                    [](std::int64_t i) { return (i != -1); });
  external_vertices.erase(external_vertices.begin(), it);

  mesh::Topology topology = mesh::create_topology(
      mpi_comm, cell_nodes, original_cell_index, ghost_owners,
      cmap.cell_shape(), external_vertices);
  int tdim = topology.dim();

  mesh::Geometry geometry = mesh::create_geometry(mpi_comm, topology, cmap,
                                                  cell_nodes, x, xshape[1]);

  auto mesh = std::make_shared<mesh::Mesh>(mpi_comm, std::move(topology),
                                           std::move(geometry));

  CHECK(mesh->topology().index_map(tdim)->size_global() == 2 * N * N);
  CHECK(mesh->topology().index_map(tdim)->size_local() > 0);

  CHECK(mesh->topology().index_map(0)->size_global() == (N + 1) * (N + 1));
  CHECK(mesh->topology().index_map(0)->size_local() > 0);

  CHECK((int)mesh->geometry().x().size() / 3
        == mesh->topology().index_map(0)->size_local()
               + mesh->topology().index_map(0)->num_ghosts());

  MPI_Group_free(&comm_group);
  MPI_Group_free(&new_group);
  if (subset_comm != MPI_COMM_NULL)
    MPI_Comm_free(&subset_comm);
}
} // namespace

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  create_mesh_file();

  SECTION("SCOTCH")
  {
    CHECK_NOTHROW(test_distributed_mesh(mesh::create_cell_partitioner()));
  }

// #ifdef HAS_KAHIP
  // SECTION("KAHIP with Lambda")
  // {
  //   auto partfn = graph::kahip::partitioner();
  //   mesh::CellPartitionFunction kahip
  //       = [&](MPI_Comm comm, int nparts, int tdim,
  //             const graph::AdjacencyList<std::int64_t>& cells)
  //   {
  //     LOG(INFO) << "Compute partition of cells across ranks (KaHIP).";
  //     // Compute distributed dual graph (for the cells on this process)
  //     const graph::AdjacencyList<std::int64_t> dual_graph
  //         = mesh::build_dual_graph(comm, cells, tdim);

  //     // Compute partition
  //     return partfn(comm, nparts, dual_graph, true);
  //   };

  //   CHECK_NOTHROW(test_distributed_mesh(kahip));
  // }
// #endif
#ifdef HAS_PARMETIS
  SECTION("parmetis")
  {
    auto partfn = graph::parmetis::partitioner();
    CHECK_NOTHROW(test_distributed_mesh(
        mesh::create_cell_partitioner(mesh::GhostMode::none, partfn)));
  }
#endif
}
