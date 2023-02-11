// Copyright (C) 2006-2020 Anders Logg, Chris Richardson, Jorgen S.
// Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Geometry.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include "topologycomputation.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <memory>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
/// Re-order an adjacency list
template <typename T>
graph::AdjacencyList<T> reorder_list(const graph::AdjacencyList<T>& list,
                                     std::span<const std::int32_t> nodemap)
{
  // Copy existing data to keep ghost values (not reordered)
  std::vector<T> data(list.array());
  std::vector<std::int32_t> offsets(list.offsets().size());

  // Compute new offsets (owned and ghost)
  offsets[0] = 0;
  for (std::size_t n = 0; n < nodemap.size(); ++n)
    offsets[nodemap[n] + 1] = list.num_links(n);
  for (std::size_t n = nodemap.size(); n < (std::size_t)list.num_nodes(); ++n)
    offsets[n + 1] = list.num_links(n);
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  graph::AdjacencyList<T> list_new(std::move(data), std::move(offsets));

  for (std::size_t n = 0; n < nodemap.size(); ++n)
  {
    auto links_old = list.links(n);
    auto links_new = list_new.links(nodemap[n]);
    assert(links_old.size() == links_new.size());
    std::copy(links_old.begin(), links_old.end(), links_new.begin());
  }

  return list_new;
}
} // namespace

//-----------------------------------------------------------------------------
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       std::span<const double> x,
                       std::array<std::size_t, 2> xshape,
                       mesh::GhostMode ghost_mode)
{
  return create_mesh(comm, cells, element, x, xshape,
                     create_cell_partitioner(ghost_mode));
}
//-----------------------------------------------------------------------------
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       std::span<const double> x,
                       std::array<std::size_t, 2> xshape,
                       const mesh::CellPartitionFunction& cell_partitioner)
{
  const fem::ElementDofLayout dof_layout = element.create_dof_layout();

  // Function top build geometry. Used to scope memory operations.
  auto build_topology = [](auto comm, auto& element, auto& dof_layout,
                           auto& cells, auto& cell_partitioner)
  {
    // -- Partition topology

    // Note: the function extract_topology (returns an
    // AdjacencyList<std::int64_t>) extract topology data, e.g. just the
    // vertices. For P1 geometry this should just be the identity
    // operator. For other elements the filtered lists may have 'gaps',
    // i.e. the indices might not be contiguous. We don't create an
    // object before calling cell_partitioner to ensure that memory is
    // freed immediately.
    //
    // Note: extract_topology could be skipped for 'P1' elements since
    // it is just the identity

    // Compute the destination rank for cells on this process via graph
    // partitioning.
    const int size = dolfinx::MPI::size(comm);
    const int tdim = cell_dim(element.cell_shape());
    const graph::AdjacencyList<std::int32_t> dest = cell_partitioner(
        comm, size, tdim,
        extract_topology(element.cell_shape(), dof_layout, cells));

    // -- Distribute cells (topology, includes higher-order 'nodes')

    // Distribute cells to destination rank
    auto [cell_nodes, src, original_cell_index0, ghost_owners]
        = graph::build::distribute(comm, cells, dest);

    // Release memory (src is not used)
    decltype(src)().swap(src);

    // -- Extra cell topology

    // Extract cell 'topology', i.e. extract the vertices for each cell
    // and discard any 'higher-order' nodes

    graph::AdjacencyList<std::int64_t> cells_extracted
        = extract_topology(element.cell_shape(), dof_layout, cell_nodes);

    // -- Re-order cells

    // Build local dual graph for owned cells to apply re-ordering to
    const std::int32_t num_owned_cells
        = cells_extracted.num_nodes() - ghost_owners.size();

    auto [graph, unmatched_facets, max_v, facet_attached_cells]
        = build_local_dual_graph(
            std::span<const std::int64_t>(
                cells_extracted.array().data(),
                cells_extracted.offsets()[num_owned_cells]),
            std::span<const std::int32_t>(cells_extracted.offsets().data(),
                                          num_owned_cells + 1),
            tdim);

    const std::vector<int> remap = graph::reorder_gps(graph);

    // Create re-ordered cell lists (leaves ghosts unchanged)
    std::vector<std::int64_t> original_cell_index(original_cell_index0.size());
    for (std::size_t i = 0; i < remap.size(); ++i)
      original_cell_index[remap[i]] = original_cell_index0[i];
    std::copy_n(std::next(original_cell_index0.cbegin(), num_owned_cells),
                ghost_owners.size(),
                std::next(original_cell_index.begin(), num_owned_cells));
    cells_extracted = reorder_list(cells_extracted, remap);
    cell_nodes = reorder_list(cell_nodes, remap);

    // -- Create Topology

    // Boundary vertices are marked as unknown
    std::vector<std::int64_t> boundary_vertices(unmatched_facets);
    std::sort(boundary_vertices.begin(), boundary_vertices.end());
    boundary_vertices.erase(
        std::unique(boundary_vertices.begin(), boundary_vertices.end()),
        boundary_vertices.end());

    // Remove -1 if it occurs in boundary vertices (may occur in mixed topology)
    if (boundary_vertices.size() > 0 and boundary_vertices[0] == -1)
      boundary_vertices.erase(boundary_vertices.begin());

    // Create cells and vertices with the ghosting requested. Input
    // topology includes cells shared via facet, but ghosts will be
    // removed later if not required by ghost_mode.
    return std::pair{create_topology(comm, cells_extracted, original_cell_index,
                                     ghost_owners, element.cell_shape(),
                                     boundary_vertices),
                     std::move(cell_nodes)};
  };

  auto [topology, cell_nodes]
      = build_topology(comm, element, dof_layout, cells, cell_partitioner);

  // Create connectivity required to compute the Geometry (extra
  // connectivities for higher-order geometries)
  int tdim = topology.dim();
  for (int e = 1; e < tdim; ++e)
  {
    if (dof_layout.num_entity_dofs(e) > 0)
      topology.create_entities(e);
  }

  if (element.needs_dof_permutations())
    topology.create_entity_permutations();

  Geometry geometry
      = create_geometry(comm, topology, element, cell_nodes, x, xshape[1]);
  return Mesh(comm, std::move(topology), std::move(geometry));
}
//-----------------------------------------------------------------------------
std::tuple<Mesh, std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
mesh::create_submesh(const Mesh& mesh, int dim,
                     std::span<const std::int32_t> entities)
{
  // Create sub-topology
  auto [topology, subentity_to_entity, subvertex_to_vertex]
      = mesh::create_subtopology(mesh.topology(), dim, entities);

  const int tdim = mesh.topology().dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(tdim, dim);

  // Create sub-geometry
  auto [geometry, subx_to_x_dofmap] = mesh::create_subgeometry(
      mesh.topology(), mesh.geometry(), dim, subentity_to_entity);

  return {Mesh(mesh.comm(), std::move(topology), std::move(geometry)),
          std::move(subentity_to_entity), std::move(subvertex_to_vertex),
          std::move(subx_to_x_dofmap)};
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
MPI_Comm Mesh::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
