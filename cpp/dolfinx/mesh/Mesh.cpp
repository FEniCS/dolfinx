// Copyright (C) 2006-2020 Anders Logg, Chris Richardson, Jorgen S.
// Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Geometry.h"
#include "Topology.h"
#include "topologycomputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       const common::array2d<double>& x,
                       mesh::GhostMode ghost_mode)
{
  return create_mesh(
      comm, cells, element, x, ghost_mode,
      static_cast<graph::AdjacencyList<std::int32_t> (*)(
          MPI_Comm, int, const mesh::CellType,
          const graph::AdjacencyList<std::int64_t>&, mesh::GhostMode)>(
          &mesh::partition_cells_graph));
}
//-----------------------------------------------------------------------------
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       const common::array2d<double>& x,
                       mesh::GhostMode ghost_mode,
                       const mesh::CellPartitionFunction& cell_partitioner)
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
  // partitioning. Always get the ghost cells via facet, though these
  // may be discarded later.
  const int size = dolfinx::MPI::size(comm);
  const graph::AdjacencyList<std::int32_t> dest
      = cell_partitioner(comm, size, element.cell_shape(), cells_topology,
                         GhostMode::shared_facet);

  // Distribute cells to destination rank
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::build::distribute(comm, cells, dest);

  // Create cells and vertices with the ghosting requested. Input
  // topology includes cells shared via facet, but output will remove
  // these, if not required by ghost_mode.
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
          = mesh::compute_entities(comm, topology, e);
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
  std::vector<std::int32_t> off1(
      cell_nodes.offsets().begin(),
      std::next(cell_nodes.offsets().begin(), n_cells_local + 1));
  std::vector<std::int64_t> data1(
      cell_nodes.array().begin(),
      std::next(cell_nodes.array().begin(), off1[n_cells_local]));
  graph::AdjacencyList<std::int64_t> cell_nodes_2(std::move(data1),
                                                  std::move(off1));
  if (element.needs_permutation_data())
    topology.create_entity_permutations();
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
MPI_Comm Mesh::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
