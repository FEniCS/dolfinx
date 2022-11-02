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
graph::AdjacencyList<T>
reorder_list(const graph::AdjacencyList<T>& list,
             const std::span<const std::int32_t>& nodemap)
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
                     const std::span<const std::int32_t>& entities)
{
  // -- Submesh topology
  const Topology& topology = mesh.topology();

  // Get the entities in the submesh that are owned by this process
  auto mesh_map = topology.index_map(dim);
  assert(mesh_map);

  std::vector<std::int32_t> submesh_owned;
  std::copy_if(
      entities.begin(), entities.end(), std::back_inserter(submesh_owned),
      [size = mesh_map->size_local()](std::int32_t e) { return e < size; });

  // Create submesh entity index map
  // TODO Call common::get_owned_indices here? Do we want to
  // support `entities` possibly having a ghost on one process that is
  // not in `entities` on the owning process?
  // TODO: Should entities still be ghosted in the submesh even if they
  // are not in the `entities` list? If this is not desirable,
  // create_submap needs to be changed

  // Create a map from the (local) entities in the submesh to the
  // (local) entities in the mesh, and create the submesh entity index
  // map.
  std::vector<int32_t> submesh_to_mesh_map(submesh_owned.begin(),
                                           submesh_owned.end());
  std::shared_ptr<common::IndexMap> submesh_map;
  {
    std::pair<common::IndexMap, std::vector<int32_t>> map_data
        = mesh_map->create_submap(submesh_owned);
    submesh_map = std::make_shared<common::IndexMap>(std::move(map_data.first));

    // Add ghost entities to the entity map
    submesh_to_mesh_map.reserve(submesh_map->size_local()
                                + submesh_map->num_ghosts());
    std::transform(
        map_data.second.begin(), map_data.second.end(),
        std::back_inserter(submesh_to_mesh_map),
        [size_local = mesh_map->size_local()](std::int32_t entity_index)
        { return size_local + entity_index; });
  }

  // Get the vertices in the submesh. Use submesh_to_mesh_map
  // (instead of `entities`) to ensure vertices for ghost entities are
  // included
  std::vector<std::int32_t> submesh_vertices
      = compute_incident_entities(mesh, submesh_to_mesh_map, dim, 0);

  // Get the vertices in the submesh owned by this process
  auto mesh_index_map0 = topology.index_map(0);
  assert(mesh_index_map0);
  std::vector<int32_t> submesh_owned_vertices
      = common::compute_owned_indices(submesh_vertices, *mesh_index_map0);

  // Create submesh vertex index map
  std::shared_ptr<common::IndexMap> submesh_map0;
  std::vector<int32_t> submesh_to_mesh_map0;
  {
    std::pair<common::IndexMap, std::vector<int32_t>> map_data
        = mesh_index_map0->create_submap(submesh_owned_vertices);
    submesh_map0
        = std::make_shared<common::IndexMap>(std::move(map_data.first));

    // Create a map from the (local) vertices in the submesh to the
    // (local) vertices in the mesh
    submesh_to_mesh_map0.assign(submesh_owned_vertices.begin(),
                                submesh_owned_vertices.end());
    submesh_to_mesh_map0.reserve(submesh_map0->size_local()
                                 + submesh_map0->num_ghosts());

    // Add ghost vertices to the map
    std::transform(
        map_data.second.begin(), map_data.second.end(),
        std::back_inserter(submesh_to_mesh_map0),
        [size_local = mesh_index_map0->size_local()](std::int32_t vertex_index)
        { return size_local + vertex_index; });
  }

  // Submesh vertex to vertex connectivity (identity)
  auto submesh_v_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      submesh_map0->size_local() + submesh_map0->num_ghosts());

  // Submesh entity to vertex connectivity
  const CellType entity_type = cell_entity_type(topology.cell_type(), dim, 0);
  const int num_vertices_per_entity = cell_num_entities(entity_type, 0);
  auto mesh_e_to_v = topology.connectivity(dim, 0);
  std::vector<std::int32_t> submesh_e_to_v_vec;
  submesh_e_to_v_vec.reserve(submesh_to_mesh_map.size()
                             * num_vertices_per_entity);
  std::vector<std::int32_t> submesh_e_to_v_offsets(1, 0);
  submesh_e_to_v_offsets.reserve(submesh_to_mesh_map.size() + 1);

  // Create mesh to submesh vertex map (i.e. the inverse of
  // submesh_to_mesh_map0)
  // NOTE: Depending on the submesh, this may be densely or sparsely
  // populated. Is a different data structure more appropriate?
  std::vector<std::int32_t> mesh_to_submesh_map0(
      mesh_index_map0->size_local() + mesh_index_map0->num_ghosts(), -1);
  for (std::size_t i = 0; i < submesh_to_mesh_map0.size(); ++i)
    mesh_to_submesh_map0[submesh_to_mesh_map0[i]] = i;

  for (std::int32_t e : submesh_to_mesh_map)
  {
    std::span<const std::int32_t> vertices = mesh_e_to_v->links(e);
    for (std::int32_t v : vertices)
    {
      std::int32_t v_submesh = mesh_to_submesh_map0[v];
      assert(v_submesh != -1);
      submesh_e_to_v_vec.push_back(v_submesh);
    }
    submesh_e_to_v_offsets.push_back(submesh_e_to_v_vec.size());
  }
  auto submesh_e_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(submesh_e_to_v_vec), std::move(submesh_e_to_v_offsets));

  // Create submesh topology
  Topology submesh_topology(mesh.comm(), entity_type);
  submesh_topology.set_index_map(0, submesh_map0);
  submesh_topology.set_index_map(dim, submesh_map);
  submesh_topology.set_connectivity(submesh_v_to_v, 0, 0);
  submesh_topology.set_connectivity(submesh_e_to_v, dim, 0);

  // -- Submesh geometry
  const Geometry& geometry = mesh.geometry();

  // Get the geometry dofs in the submesh based on the entities in
  // submesh
  const fem::ElementDofLayout layout = geometry.cmap().create_dof_layout();
  // NOTE: Unclear what this return for prisms
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);

  std::vector<std::int32_t> geometry_indices(num_entity_dofs
                                             * submesh_to_mesh_map.size());
  {
    const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
    const int tdim = topology.dim();
    mesh.topology_mutable().create_entities(dim);
    mesh.topology_mutable().create_connectivity(dim, tdim);
    mesh.topology_mutable().create_connectivity(tdim, dim);

    // Fetch connectivities required to get entity dofs
    const std::vector<std::vector<std::vector<int>>>& closure_dofs
        = layout.entity_closure_dofs_all();
    std::shared_ptr<const graph::AdjacencyList<int>> e_to_c
        = topology.connectivity(dim, tdim);
    assert(e_to_c);
    std::shared_ptr<const graph::AdjacencyList<int>> c_to_e
        = topology.connectivity(tdim, dim);
    assert(c_to_e);
    for (std::size_t i = 0; i < submesh_to_mesh_map.size(); ++i)
    {
      const std::int32_t idx = submesh_to_mesh_map[i];
      assert(!e_to_c->links(idx).empty());
      // Always pick the last cell to be consistent with the e_to_v connectivity
      const std::int32_t cell = e_to_c->links(idx).back();
      auto cell_entities = c_to_e->links(cell);
      auto it = std::find(cell_entities.begin(), cell_entities.end(), idx);
      assert(it != cell_entities.end());
      const auto local_entity = std::distance(cell_entities.begin(), it);
      const std::vector<std::int32_t>& entity_dofs
          = closure_dofs[dim][local_entity];

      auto xc = xdofs.links(cell);
      for (std::size_t j = 0; j < num_entity_dofs; ++j)
        geometry_indices[i * num_entity_dofs + j] = xc[entity_dofs[j]];
    }
  }

  std::vector<std::int32_t> submesh_x_dofs = geometry_indices;
  std::sort(submesh_x_dofs.begin(), submesh_x_dofs.end());
  submesh_x_dofs.erase(
      std::unique(submesh_x_dofs.begin(), submesh_x_dofs.end()),
      submesh_x_dofs.end());

  // Get the geometry dofs in the submesh owned by this process
  auto mesh_geometry_dof_index_map = geometry.index_map();
  assert(mesh_geometry_dof_index_map);
  auto submesh_owned_x_dofs = common::compute_owned_indices(
      submesh_x_dofs, *mesh_geometry_dof_index_map);

  // Create submesh geometry index map
  std::vector<int32_t> submesh_to_mesh_x_dof_map(submesh_owned_x_dofs.begin(),
                                                 submesh_owned_x_dofs.end());
  std::shared_ptr<common::IndexMap> submesh_x_dof_index_map;
  {
    std::pair<common::IndexMap, std::vector<int32_t>>
        submesh_x_dof_index_map_pair
        = mesh_geometry_dof_index_map->create_submap(submesh_owned_x_dofs);

    submesh_x_dof_index_map = std::make_shared<common::IndexMap>(
        std::move(submesh_x_dof_index_map_pair.first));

    // Create a map from the (local) geometry dofs in the submesh to the
    // (local) geometry dofs in the mesh.
    submesh_to_mesh_x_dof_map.reserve(submesh_x_dof_index_map->size_local()
                                      + submesh_x_dof_index_map->num_ghosts());
    std::transform(submesh_x_dof_index_map_pair.second.begin(),
                   submesh_x_dof_index_map_pair.second.end(),
                   std::back_inserter(submesh_to_mesh_x_dof_map),
                   [size = mesh_geometry_dof_index_map->size_local()](
                       auto x_dof_index) { return size + x_dof_index; });
  }

  // Create submesh geometry coordinates
  std::span<const double> mesh_x = mesh.geometry().x();
  const int submesh_num_x_dofs = submesh_to_mesh_x_dof_map.size();
  std::vector<double> submesh_x(3 * submesh_num_x_dofs);
  for (int i = 0; i < submesh_num_x_dofs; ++i)
  {
    std::copy_n(std::next(mesh_x.begin(), 3 * submesh_to_mesh_x_dof_map[i]), 3,
                std::next(submesh_x.begin(), 3 * i));
  }

  // Create mesh to submesh geometry map
  std::vector<std::int32_t> mesh_to_submesh_x_dof_map(
      mesh_geometry_dof_index_map->size_local()
          + mesh_geometry_dof_index_map->num_ghosts(),
      -1);
  for (std::size_t i = 0; i < submesh_to_mesh_x_dof_map.size(); ++i)
    mesh_to_submesh_x_dof_map[submesh_to_mesh_x_dof_map[i]] = i;

  // Create submesh geometry dofmap
  std::vector<std::int32_t> entity_x_dofs;
  std::vector<std::int32_t> submesh_x_dofmap_vec;
  submesh_x_dofmap_vec.reserve(geometry_indices.size());
  std::vector<std::int32_t> submesh_x_dofmap_offsets(1, 0);
  submesh_x_dofmap_offsets.reserve(submesh_to_mesh_map.size() + 1);
  for (std::size_t i = 0; i < submesh_to_mesh_map.size(); ++i)
  {
    // Get the mesh geometry dofs for ith entity in entities
    auto it = std::next(geometry_indices.begin(), i * num_entity_dofs);
    entity_x_dofs.assign(it, std::next(it, num_entity_dofs));

    // For each mesh dof of the entity, get the submesh dof
    for (std::int32_t x_dof : entity_x_dofs)
    {
      std::int32_t x_dof_submesh = mesh_to_submesh_x_dof_map[x_dof];
      assert(x_dof_submesh != -1);
      submesh_x_dofmap_vec.push_back(x_dof_submesh);
    }
    submesh_x_dofmap_offsets.push_back(submesh_x_dofmap_vec.size());
  }
  graph::AdjacencyList<std::int32_t> submesh_x_dofmap(
      std::move(submesh_x_dofmap_vec), std::move(submesh_x_dofmap_offsets));

  // Create submesh coordinate element
  CellType submesh_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  fem::CoordinateElement submesh_coord_ele(
      submesh_coord_cell, geometry.cmap().degree(), geometry.cmap().variant());

  // Submesh geometry input_global_indices
  // TODO Check this
  const std::vector<std::int64_t>& mesh_igi = geometry.input_global_indices();
  std::vector<std::int64_t> submesh_igi;
  submesh_igi.reserve(submesh_to_mesh_x_dof_map.size());
  std::transform(submesh_to_mesh_x_dof_map.begin(),
                 submesh_to_mesh_x_dof_map.end(),
                 std::back_inserter(submesh_igi),
                 [&mesh_igi](std::int32_t submesh_x_dof)
                 { return mesh_igi[submesh_x_dof]; });

  // Create geometry
  Geometry submesh_geometry(
      submesh_x_dof_index_map, std::move(submesh_x_dofmap), submesh_coord_ele,
      std::move(submesh_x), geometry.dim(), std::move(submesh_igi));

  return {Mesh(mesh.comm(), std::move(submesh_topology),
               std::move(submesh_geometry)),
          std::move(submesh_to_mesh_map), std::move(submesh_to_mesh_map0),
          std::move(submesh_to_mesh_x_dof_map)};
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
