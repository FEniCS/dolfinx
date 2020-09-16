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
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMapBuilder.h>
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
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>& x,
                       mesh::GhostMode ghost_mode)
{
  if (ghost_mode == mesh::GhostMode::shared_vertex)
    throw std::runtime_error("Ghost mode via vertex currently disabled.");

  // TODO: This step can be skipped for 'P1' elements
  //
  // Extract topology data, e.g. just the vertices. For P1 geometry this
  // should just be the identity operator. For other elements the
  // filtered lists may have 'gaps', i.e. the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_topology
      = mesh::extract_topology(element.cell_shape(), element.dof_layout(),
                               cells);

  // Compute the destination rank for cells on this process via graph
  // partitioning. Always get the ghost cells via facet, though these may be
  // discarded later.
  const int size = dolfinx::MPI::size(comm);
  const graph::AdjacencyList<std::int32_t> dest
      = Partitioning::partition_cells(comm, size, element.cell_shape(),
                                      cells_topology, GhostMode::shared_facet);

  // Distribute cells to destination rank
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::Partitioning::distribute(comm, cells, dest);

  // Create cells and vertices with the ghosting requested. Input topology
  // includes cells shared via facet, but output will remove these, if not
  // required by ghost_mode.
  Topology topology = mesh::create_topology(
      comm,
      mesh::extract_topology(element.cell_shape(), element.dof_layout(),
                             cell_nodes),
      original_cell_index, ghost_owners, element.cell_shape(), ghost_mode);

  // Create connectivity required to compute the Geometry (extra
  // connectivities for higher-order geometries)
  const int tdim = topology.dim();
  for (int e = 1; e < tdim; ++e)
  {
    if (element.dof_layout().num_entity_dofs(e) > 0)
    {
      auto [cell_entity, entity_vertex, index_map]
          = mesh::TopologyComputation::compute_entities(comm, topology, e);
      if (cell_entity)
        topology.set_connectivity(cell_entity, tdim, e);
      if (entity_vertex)
        topology.set_connectivity(entity_vertex, e, 0);
      if (index_map)
        topology.set_index_map(e, index_map);
    }
  }

  int n_cells_local = topology.index_map(tdim)->size_local()
                      + topology.index_map(tdim)->num_ghosts();

  // Remove ghost cells from geometry data, if not required.
  const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>& off1
      = cell_nodes.offsets().head(n_cells_local + 1);
  const Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>& data1
      = cell_nodes.array().head(off1[n_cells_local]);
  graph::AdjacencyList<std::int64_t> cell_nodes_2(data1, off1);

  Geometry geometry
      = mesh::create_geometry(comm, topology, element, cell_nodes_2, x);

  return Mesh(comm, std::move(topology), std::move(geometry));
}
//-----------------------------------------------------------------------------

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
