// Copyright (C) 2006-2020 Anders Logg, Chris Richardson, Jorgen S.
// Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Geometry.h"
#include "Partitioning.h"
#include "Topology.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
Eigen::ArrayXd cell_h(const mesh::Mesh& mesh)
{
  const int dim = mesh.topology().dim();
  auto map = mesh.topology().index_map(dim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  if (num_cells == 0)
    throw std::runtime_error("Cannot compute h min/max. No cells.");

  Eigen::ArrayXi cells(num_cells);
  std::iota(cells.data(), cells.data() + cells.size(), 0);
  return mesh::h(mesh, cells, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd cell_r(const mesh::Mesh& mesh)
{
  const int dim = mesh.topology().dim();
  auto map = mesh.topology().index_map(dim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  if (num_cells == 0)
    throw std::runtime_error("Cannnot compute inradius min/max. No cells.");

  Eigen::ArrayXi cells(num_cells);
  std::iota(cells.data(), cells.data() + cells.size(), 0);
  return mesh::inradius(mesh, cells);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Mesh mesh::create(MPI_Comm comm,
                  const graph::AdjacencyList<std::int64_t>& cells,
                  const fem::ElementDofLayout& layout,
                  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>& x,
                  mesh::GhostMode ghost_mode)
{
  // TODO: This step can be skipped for 'P1' elements
  //
  // Extract topology data, e.g. just the vertices. For P1 geometry this
  // should just be the identity operator. For other elements the
  // filtered lists may have 'gaps', i.e. the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_topology
      = mesh::extract_topology(layout, cells);

  // Compute the destination rank for cells on this process via graph
  // partitioning
  const int size = dolfinx::MPI::size(comm);
  const graph::AdjacencyList<std::int32_t> dest = Partitioning::partition_cells(
      comm, size, layout.cell_type(), cells_topology, ghost_mode);

  // Distribute cells to destination rank
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::Partitioning::distribute(comm, cells, dest);

  Topology topology = mesh::create_topology(
      comm, mesh::extract_topology(layout, cell_nodes), original_cell_index,
      ghost_owners, layout, ghost_mode);

  // FIXME: Figure out how to check which entities are required
  // Initialise facet for P2
  // Create local entities
  if (topology.dim() > 1)
  {
    // Create edges
    auto [cell_entity, entity_vertex, index_map]
        = mesh::TopologyComputation::compute_entities(comm, topology, 1);
    if (cell_entity)
      topology.set_connectivity(cell_entity, topology.dim(), 1);
    if (entity_vertex)
      topology.set_connectivity(entity_vertex, 1, 0);
    if (index_map)
      topology.set_index_map(1, index_map);

    // Create facets
    auto [cell_facet, facet_vertex, index_map1]
        = mesh::TopologyComputation::compute_entities(comm, topology,
                                                      topology.dim() - 1);
    if (cell_facet)
      topology.set_connectivity(cell_facet, topology.dim(), topology.dim() - 1);
    if (facet_vertex)
      topology.set_connectivity(facet_vertex, topology.dim() - 1, 0);
    if (index_map1)
      topology.set_index_map(topology.dim() - 1, index_map1);
  }

  Geometry geometry
      = mesh::create_geometry(comm, topology, layout, cell_nodes, x);

  return Mesh(comm, std::move(topology), std::move(geometry));
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Mesh::Mesh(
    MPI_Comm comm, mesh::CellType type,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x,
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::int64_t>&, const GhostMode ghost_mode, std::int32_t)
    : Mesh(mesh::create(comm, graph::AdjacencyList<std::int64_t>(cells),
                        fem::geometry_layout(type, cells.cols()), x,
                        ghost_mode))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Topology& Mesh::topology() { return _topology; }
//-----------------------------------------------------------------------------
const Topology& Mesh::topology() const { return _topology; }
//-----------------------------------------------------------------------------
Topology& Mesh::topology_mutable() const { return _topology; }
//-----------------------------------------------------------------------------
Geometry& Mesh::geometry() { return _geometry; }
//-----------------------------------------------------------------------------
const Geometry& Mesh::geometry() const { return _geometry; }
//-----------------------------------------------------------------------------
std::int32_t Mesh::create_entities(int dim) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of a
  // mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.
  return _topology.create_entities(dim);
}
//-----------------------------------------------------------------------------
void Mesh::create_connectivity(int d0, int d1) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of a
  // mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  // TODO: This should go away
  _topology.create_connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
void Mesh::create_entity_permutations() const
{
  // TODO: This should go away
  _topology.create_entity_permutations();
}
//-----------------------------------------------------------------------------
void Mesh::create_connectivity_all() const
{
  // TODO: This should go away
  _topology.create_connectivity_all();
}
//-----------------------------------------------------------------------------
double Mesh::hmin() const { return cell_h(*this).minCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::hmax() const { return cell_h(*this).maxCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::rmin() const { return cell_r(*this).minCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::rmax() const { return cell_r(*this).maxCoeff(); }
//-----------------------------------------------------------------------------
std::size_t Mesh::hash() const
{
  // Get local hashes
  const std::size_t kt_local = _topology.hash();
  const std::size_t kg_local = _geometry.hash();

  // Compute global hash
  const std::size_t kt = common::hash_global(_mpi_comm.comm(), kt_local);
  const std::size_t kg = common::hash_global(_mpi_comm.comm(), kg_local);

  // Compute hash based on the Cantor pairing function
  return (kt + kg) * (kt + kg + 1) / 2 + kg;
}
//-----------------------------------------------------------------------------
MPI_Comm Mesh::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
