// Copyright (C) 2006-2020 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include "topologycomputation.h"
#include "utils.h"
#include <algorithm>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <memory>
#include <string>
#include <utility>

namespace dolfinx::mesh
{

/// A Mesh consists of a set of connected and numbered mesh topological
/// entities, and geometry data
template <typename T>
class Mesh
{
public:
  /// Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <std::convertible_to<Topology> U, std::convertible_to<Geometry<T>> V>
  Mesh(MPI_Comm comm, U&& topology, V&& geometry)
      : _topology(std::forward<U>(topology)),
        _geometry(std::forward<V>(geometry)), _comm(comm)
  {
    // Do nothing
  }

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh) = default;

  /// Move constructor
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh) = default;

  /// Destructor
  ~Mesh() = default;

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  // TODO: Is there any use for this? In many situations one has to get the
  // topology of a const Mesh, which is done by Mesh::topology_mutable. Note
  // that the python interface (calls Mesh::topology()) may still rely on it.
  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology() { return _topology; }

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const { return _topology; }

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  Topology& topology_mutable() const { return _topology; }

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry<T>& geometry() { return _geometry; }

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry<T>& geometry() const { return _geometry; }

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm comm() const { return _comm.comm(); }

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology:
  // TODO: This is mutable because of the current memory management within
  // mesh::Topology. It allows to obtain a non-const Topology from a
  // const mesh (via Mesh::topology_mutable()).
  //
  mutable Topology _topology;

  // Mesh geometry
  Geometry<T> _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};

/// Create a mesh using a provided mesh partitioning function
// template <typename T>
// Mesh<T> create_mesh(MPI_Comm comm,
//                     const graph::AdjacencyList<std::int64_t>& cells,
//                     const fem::CoordinateElement& element, std::span<const T>
//                     x, std::array<std::size_t, 2> xshape, const
//                     CellPartitionFunction& cell_partitioner)
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
            const fem::CoordinateElement& element, const U& x,
            std::array<std::size_t, 2> xshape,
            const CellPartitionFunction& cell_partitioner)
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

    /// Re-order an adjacency list
    auto reorder_list
        = [](const auto& list, std::span<const std::int32_t> nodemap)
    {
      using X =
          typename std::remove_reference_t<decltype(list.array())>::value_type;

      // Copy existing data to keep ghost values (not reordered)
      std::vector<X> data(list.array());
      std::vector<std::int32_t> offsets(list.offsets().size());

      // Compute new offsets (owned and ghost)
      offsets[0] = 0;
      for (std::size_t n = 0; n < nodemap.size(); ++n)
        offsets[nodemap[n] + 1] = list.num_links(n);
      for (std::size_t n = nodemap.size(); n < (std::size_t)list.num_nodes();
           ++n)
        offsets[n + 1] = list.num_links(n);
      std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
      graph::AdjacencyList<X> list_new(std::move(data), std::move(offsets));

      for (std::size_t n = 0; n < nodemap.size(); ++n)
      {
        auto links_old = list.links(n);
        auto links_new = list_new.links(nodemap[n]);
        assert(links_old.size() == links_new.size());
        std::copy(links_old.begin(), links_old.end(), links_new.begin());
      }

      return list_new;
    };

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
  return Mesh<double>(comm, std::move(topology), std::move(geometry));
}

/// @brief Create a mesh using the default partitioner.
///
/// This function takes mesh input data that is distributed across
/// processes and creates a mesh::Mesh, with the mesh cell distribution
/// determined by the default cell partitioner. The default partitioner
/// is based a graph partitioning.
///
/// @param[in] comm The MPI communicator to build the mesh on
/// @param[in] cells The cells on the this MPI rank. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices). For lowest order cells this will be just the cell
/// vertices. For higher-order cells, other cells 'nodes' will be
/// included.
/// @param[in] element The coordinate element that describes the
/// geometric mapping for cells
/// @param[in] x The coordinates of mesh nodes
/// @param[in] xshape The shape of `x`. It should be `(num_points, gdim)`.
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A distributed Mesh.
// template <typename T>
// Mesh<T> create_mesh(MPI_Comm comm,
//                     const graph::AdjacencyList<std::int64_t>& cells,
//                     const fem::CoordinateElement& element, std::span<const T>
//                     x, std::array<std::size_t, 2> xshape, GhostMode
//                     ghost_mode)
template <typename U>
Mesh<typename std::remove_reference_t<typename U::value_type>>
create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
            const fem::CoordinateElement& element, const U& x,
            std::array<std::size_t, 2> xshape, GhostMode ghost_mode)
{
  return create_mesh(comm, cells, element, x, xshape,
                     create_cell_partitioner(ghost_mode));
}

/// Create a new mesh consisting of a subset of entities in a mesh.
/// @param[in] mesh The mesh
/// @param[in] dim Entity dimension
/// @param[in] entities List of entity indices in `mesh` to include in
/// the new mesh
/// @return The new mesh, and maps from the new mesh entities, vertices,
/// and geometry to the input mesh entities, vertices, and geometry.
template <typename T>
std::tuple<Mesh<T>, std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
create_submesh(const Mesh<T>& mesh, int dim,
               std::span<const std::int32_t> entities)
{
  // Create sub-topology
  mesh.topology_mutable().create_connectivity(dim, 0);
  auto [topology, subentity_to_entity, subvertex_to_vertex]
      = mesh::create_subtopology(mesh.topology(), dim, entities);

  // Create sub-geometry
  const int tdim = mesh.topology().dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(tdim, dim);
  auto [geometry, subx_to_x_dofmap] = mesh::create_subgeometry(
      mesh.topology(), mesh.geometry(), dim, subentity_to_entity);

  return {Mesh<T>(mesh.comm(), std::move(topology), std::move(geometry)),
          std::move(subentity_to_entity), std::move(subvertex_to_vertex),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
