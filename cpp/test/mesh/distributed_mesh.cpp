// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <algorithm>
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/graphbuild.h>
#include <memory>

using namespace dolfinx;

namespace
{
constexpr int N = 8;

/// @brief Create a mesh and save to the file 'mesh.xdmf'
[[maybe_unused]] void create_mesh_file(MPI_Comm comm)
{
  // Create mesh using all processes and save xdmf
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_rectangle(comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {N, N},
                             mesh::CellType::triangle, part));

  // Save mesh in XDMF format
  io::XDMFFile file(MPI_COMM_SELF, "mesh.xdmf", "w");
  file.write_mesh(*mesh);
}

[[maybe_unused]] void test_create_box(mesh::CellPartitionFunction part)
{
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // Create sub-communicator on even ranks
  const int mpi_rank = dolfinx::MPI::rank(comm);
  int color = mpi_rank % 2 ? MPI_UNDEFINED : 1;
  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(comm, color, mpi_rank, &subcomm);

  // Create mesh on comm and distribute to all ranks in comm
  mesh::Mesh<double> mesh0
      = mesh::create_box(comm, comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
                         {12, 12, 12}, mesh::CellType::hexahedron, part);
  int tdim = mesh0.topology()->dim();
  mesh0.topology()->create_entities(tdim - 1);

  // Create mesh on even ranks (subcomm) and distribute to all ranks in
  // comm
  mesh::Mesh<double> mesh1
      = mesh::create_box(comm, subcomm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
                         {12, 12, 12}, mesh::CellType::hexahedron, part);
  int tdim1 = mesh1.topology()->dim();
  mesh1.topology()->create_entities(tdim1 - 1);

  // Check that mesh communicators are the same
  int equal;
  MPI_Comm_compare(mesh0.comm(), mesh1.comm(), &equal);
  CHECK(equal != MPI_UNEQUAL);

  // Check global sizes for topology and geometry
  auto t0 = mesh0.topology();
  auto t1 = mesh1.topology();
  CHECK(t0->index_map(tdim)->size_global()
        == t1->index_map(tdim)->size_global());
  CHECK(t0->index_map(tdim - 1)->size_global()
        == t1->index_map(tdim - 1)->size_global());
  CHECK(t0->index_map(0)->size_global() == t1->index_map(0)->size_global());
  CHECK(mesh0.geometry().index_map()->size_global()
        == mesh1.geometry().index_map()->size_global());

  if (subcomm != MPI_COMM_NULL)
    MPI_Comm_free(&subcomm);
  MPI_Comm_free(&comm);
}

void test_distributed_mesh(mesh::CellPartitionFunction partitioner)
{
  using T = double;

  MPI_Comm comm = MPI_COMM_WORLD;
  const int mpi_size = dolfinx::MPI::size(comm);

  // -- Create a communicator with subset of the original group of
  // processes

  // Get group of processes on comm
  MPI_Group comm_group;
  int ierr = MPI_Comm_group(comm, &comm_group);
  if (ierr != MPI_SUCCESS)
    throw std::runtime_error("MPI_Comm_group failed");

  // Create a group of processes (lower 'half' of processes by rank)
  std::vector<int> ranks(std::max(mpi_size / 2, 1));
  std::iota(ranks.begin(), ranks.end(), 0);
  MPI_Group new_group;
  ierr = MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &new_group);
  if (ierr != MPI_SUCCESS)
    throw std::runtime_error("MPI_Group_incl failed");

  // Create new communicator
  MPI_Comm subset_comm = MPI_COMM_NULL;
  ierr = MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &subset_comm);
  if (ierr != MPI_SUCCESS)
    throw std::runtime_error("MPI_Comm_create_group failed");

  // Create coordinate map
  auto e = std::make_shared<basix::FiniteElement<T>>(basix::create_element<T>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false));
  fem::CoordinateElement<T> cmap(e);

  // Read mesh data from file on sub-communicator
  std::vector<T> x;
  std::array<std::size_t, 2> xshape = {0, 2};
  std::vector<std::int64_t> cells;
  std::array<std::size_t, 2> cshape = {0, 3};
  graph::AdjacencyList<std::int32_t> dest(0);
  if (subset_comm != MPI_COMM_NULL)
  {
    io::XDMFFile infile(subset_comm, "mesh.xdmf", "r");
    std::tie(cells, cshape) = infile.read_topology_data("mesh");
    auto [_x, _xshape] = infile.read_geometry_data("mesh");
    x = std::move(std::get<std::vector<T>>(_x));
    int nparts = mpi_size;
    int tdim = mesh::cell_dim(mesh::CellType::triangle);
    dest = partitioner(subset_comm, nparts, tdim,
                       graph::regular_adjacency_list(cells, cshape[1]));
  }
  CHECK(xshape[1] == 2);

  // -- Distribute cells to destination ranks
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::build::distribute(
          comm, graph::regular_adjacency_list(cells, cshape[1]), dest);

  // FIXME: improve way to find 'external' vertices
  // Count the connections of all vertices on owned cells. If there are 6
  // connections (on a regular triangular mesh) then it is 'internal'.
  std::vector<std::int64_t> external_vertices;
  std::vector<int> cell_group_offsets;
  {
    int num_local_cells = cell_nodes.num_nodes() - ghost_owners.size();
    int ghost_offset = cell_nodes.offsets()[num_local_cells];
    external_vertices = std::vector<std::int64_t>(
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
    cell_group_offsets
        = {0, std::int32_t(cell_nodes.num_nodes() - ghost_owners.size()),
           cell_nodes.num_nodes()};
  }

  std::vector<mesh::CellType> cell_types = {cmap.cell_shape()};
  mesh::Topology topology = mesh::create_topology(
      comm, cell_nodes, original_cell_index, ghost_owners, cell_types,
      cell_group_offsets, external_vertices);
  int tdim = topology.dim();

  mesh::Geometry geometry
      = mesh::create_geometry(comm, topology, {cmap}, cell_nodes, x, xshape[1]);

  mesh::Mesh<T> mesh(comm,
                     std::make_shared<mesh::Topology>(std::move(topology)),
                     std::move(geometry));

  auto t = mesh.topology();
  CHECK(t->index_map(tdim)->size_global() == 2 * N * N);
  CHECK(t->index_map(tdim)->size_local() > 0);
  CHECK(t->index_map(0)->size_global() == (N + 1) * (N + 1));
  CHECK(t->index_map(0)->size_local() > 0);
  CHECK((int)mesh.geometry().x().size() / 3
        == t->index_map(0)->size_local() + t->index_map(0)->num_ghosts());

  MPI_Group_free(&comm_group);
  MPI_Group_free(&new_group);
  if (subset_comm != MPI_COMM_NULL)
    MPI_Comm_free(&subset_comm);
}
} // namespace

/// Create a mesh on even ranks and distribute to all ranks in mpi_comm
TEST_CASE("Create box", "[create_box]")
{
#ifdef HAS_PTSCOTCH
  CHECK_NOTHROW(test_create_box(mesh::create_cell_partitioner(
      mesh::GhostMode::none, graph::scotch::partitioner())));
#endif
#ifdef HAS_PARMETIS
  CHECK_NOTHROW(test_create_box(mesh::create_cell_partitioner(
      mesh::GhostMode::none, graph::parmetis::partitioner())));
#endif
  // #ifdef HAS_KAHIP
  //   CHECK_NOTHROW(test_create_box(mesh::create_cell_partitioner(
  //       mesh::GhostMode::none, graph::kahip::partitioner(1, 1, 0.03,
  //       false))));
  // #endif
}

TEST_CASE("Distributed Mesh", "[distributed_mesh]")
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
    create_mesh_file(MPI_COMM_SELF);
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef HAS_PTSCOTCH
  CHECK_NOTHROW(test_distributed_mesh(mesh::create_cell_partitioner(
      mesh::GhostMode::none, graph::scotch::partitioner())));
#endif
#ifdef HAS_PARMETIS
  CHECK_NOTHROW(test_distributed_mesh(mesh::create_cell_partitioner(
      mesh::GhostMode::none, graph::parmetis::partitioner())));
#endif
#ifdef HAS_KAHIP
  CHECK_NOTHROW(test_distributed_mesh(mesh::create_cell_partitioner(
      mesh::GhostMode::none, graph::kahip::partitioner(1, 1, 0.03, false))));
#endif
}
