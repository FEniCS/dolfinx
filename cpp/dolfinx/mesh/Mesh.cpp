// Copyright (C) 2006-2020 Anders Logg, Chris Richardson, Jorgen S.
// Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
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
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <xtensor/xsort.hpp>

#include "graphbuild.h"

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
/// Re-order an adjacency list
template <typename T>
graph::AdjacencyList<T>
reorder_list(const graph::AdjacencyList<T>& list,
             const xtl::span<const std::int32_t>& nodemap)
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
                       const xt::xtensor<double, 2>& x,
                       mesh::GhostMode ghost_mode)
{
  return create_mesh(comm, cells, element, x, ghost_mode,
                     create_cell_partitioner());
}
//-----------------------------------------------------------------------------
Mesh mesh::create_mesh(MPI_Comm comm,
                       const graph::AdjacencyList<std::int64_t>& cells,
                       const fem::CoordinateElement& element,
                       const xt::xtensor<double, 2>& x,
                       mesh::GhostMode ghost_mode,
                       const mesh::CellPartitionFunction& cell_partitioner)
{
  if (ghost_mode == mesh::GhostMode::shared_vertex)
    throw std::runtime_error("Ghost mode via vertex currently disabled.");

  const fem::ElementDofLayout dof_layout = element.create_dof_layout();

  // TODO: This step can be skipped for 'P1' elements
  //
  // Extract topology data, e.g. just the vertices. For P1 geometry this
  // should just be the identity operator. For other elements the
  // filtered lists may have 'gaps', i.e. the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_topology
      = mesh::extract_topology(element.cell_shape(), dof_layout, cells);

  // Compute the destination rank for cells on this process via graph
  // partitioning. Always get the ghost cells via facet, though these
  // may be discarded later.
  const int size = dolfinx::MPI::size(comm);
  const int tdim = mesh::cell_dim(element.cell_shape());
  const graph::AdjacencyList<std::int32_t> dest = cell_partitioner(
      comm, size, tdim, cells_topology, GhostMode::shared_facet);

  // Distribute cells to destination rank
  const auto [cell_nodes0, src, original_cell_index0, ghost_owners]
      = graph::build::distribute(comm, cells, dest);

  // Extract cell 'topology', i.e. the vertices for each cell
  const graph::AdjacencyList<std::int64_t> cells_extracted0
      = mesh::extract_topology(element.cell_shape(), dof_layout, cell_nodes0);

  // Build local dual graph for owned cells to apply re-ordering to
  const std::int32_t num_owned_cells
      = cells_extracted0.num_nodes() - ghost_owners.size();
  const auto [g, m] = mesh::build_local_dual_graph(
      xtl::span<const std::int64_t>(
          cells_extracted0.array().data(),
          cells_extracted0.offsets()[num_owned_cells]),
      xtl::span<const std::int32_t>(cells_extracted0.offsets().data(),
                                    num_owned_cells + 1),
      tdim);

  // Compute re-ordering of local dual graph
  std::vector<int> remap = graph::reorder_gps(g);

  // Create re-ordered cell lists
  std::vector<std::int64_t> original_cell_index(original_cell_index0);
  for (std::size_t i = 0; i < remap.size(); ++i)
    original_cell_index[remap[i]] = original_cell_index0[i];
  const graph::AdjacencyList<std::int64_t> cells_extracted
      = reorder_list(cells_extracted0, remap);
  const graph::AdjacencyList<std::int64_t> cell_nodes
      = reorder_list(cell_nodes0, remap);

  // Create cells and vertices with the ghosting requested. Input
  // topology includes cells shared via facet, but ghosts will be
  // removed later if not required by ghost_mode.
  Topology topology
      = mesh::create_topology(comm, cells_extracted, original_cell_index,
                              ghost_owners, element.cell_shape(), ghost_mode);

  // Create connectivity required to compute the Geometry (extra
  // connectivities for higher-order geometries)
  for (int e = 1; e < tdim; ++e)
  {
    if (dof_layout.num_entity_dofs(e) > 0)
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

  const int n_cells_local = topology.index_map(tdim)->size_local()
                            + topology.index_map(tdim)->num_ghosts();

  // Remove ghost cells from geometry data, if not required
  std::vector<std::int32_t> off1(
      cell_nodes.offsets().begin(),
      std::next(cell_nodes.offsets().begin(), n_cells_local + 1));
  std::vector<std::int64_t> data1(
      cell_nodes.array().begin(),
      std::next(cell_nodes.array().begin(), off1[n_cells_local]));
  graph::AdjacencyList<std::int64_t> cell_nodes1(std::move(data1),
                                                 std::move(off1));
  if (element.needs_dof_permutations())
    topology.create_entity_permutations();

  return Mesh(comm, std::move(topology),
              mesh::create_geometry(comm, topology, element, cell_nodes1, x));
}
//-----------------------------------------------------------------------------
std::tuple<Mesh, std::vector<std::int32_t>, std::vector<std::int32_t>>
mesh::create_submesh(const Mesh& mesh, int dim,
                     const xtl::span<const std::int32_t>& entities)
{
  // Submesh topology
  // Get the verticies in the submesh
  std::vector<std::int32_t> submesh_vertices
      = mesh::compute_incident_entities(mesh, entities, dim, 0);

  // Get the vertices in the submesh owned by this process
  auto mesh_vertex_index_map = mesh.topology().index_map(0);
  assert(mesh_vertex_index_map);
  std::vector<int32_t> submesh_owned_vertices
      = dolfinx::common::compute_owned_indices(submesh_vertices,
                                               *mesh_vertex_index_map);

  // Create submesh vertex index map
  std::pair<common::IndexMap, std::vector<int32_t>>
      submesh_vertex_index_map_pair
      = mesh_vertex_index_map->create_submap(submesh_owned_vertices);
  auto submesh_vertex_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_vertex_index_map_pair.first));

  // Create a map from the (local) vertices in the submesh to the (local)
  // vertices in the mesh.
  std::vector<int32_t> submesh_to_mesh_vertex_map(
      submesh_owned_vertices.begin(), submesh_owned_vertices.end());
  submesh_to_mesh_vertex_map.reserve(submesh_vertex_index_map->size_local()
                                     + submesh_vertex_index_map->num_ghosts());
  // Add ghost vertices to the map
  std::transform(submesh_vertex_index_map_pair.second.begin(),
                 submesh_vertex_index_map_pair.second.end(),
                 std::back_inserter(submesh_to_mesh_vertex_map),
                 [mesh_vertex_index_map](std::int32_t vertex_index) {
                   return mesh_vertex_index_map->size_local() + vertex_index;
                 });

  // Get the entities in the submesh that are owned by this process
  auto mesh_entity_index_map = mesh.topology().index_map(dim);
  assert(mesh_entity_index_map);
  std::vector<std::int32_t> submesh_owned_entities;
  std::copy_if(entities.begin(), entities.end(),
               std::back_inserter(submesh_owned_entities),
               [mesh_entity_index_map](std::int32_t e)
               { return e < mesh_entity_index_map->size_local(); });

  // Create submesh entity index map
  // TODO Call dolfinx::common::get_owned_indices here? Do we want to
  // support `entities` possibly haveing a ghost on one process that is not
  // in `entities` on the owning process?
  std::pair<common::IndexMap, std::vector<int32_t>>
      submesh_entity_index_map_pair
      = mesh_entity_index_map->create_submap(submesh_owned_entities);
  auto submesh_entity_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_entity_index_map_pair.first));

  // Submesh vertex to vertex connectivity (identity)
  auto submesh_v_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      submesh_vertex_index_map->size_local()
      + submesh_vertex_index_map->num_ghosts());

  // Submesh entity to vertex connectivity
  const CellType entity_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim, 0);
  const int num_vertices_per_entity = mesh::cell_num_entities(entity_type, 0);
  auto mesh_e_to_v = mesh.topology().connectivity(dim, 0);
  std::vector<std::int32_t> submesh_e_to_v_vec;
  submesh_e_to_v_vec.reserve(entities.size() * num_vertices_per_entity);
  std::vector<std::int32_t> submesh_e_to_v_offsets(1, 0);
  submesh_e_to_v_offsets.reserve(entities.size() + 1);
  for (std::int32_t e : entities)
  {
    xtl::span<const std::int32_t> vertices = mesh_e_to_v->links(e);
    for (std::int32_t v : vertices)
    {
      auto it = std::find(submesh_to_mesh_vertex_map.begin(),
                          submesh_to_mesh_vertex_map.end(), v);
      assert(it != submesh_to_mesh_vertex_map.end());
      submesh_e_to_v_vec.push_back(
          std::distance(submesh_to_mesh_vertex_map.begin(), it));
    }
    submesh_e_to_v_offsets.push_back(submesh_e_to_v_vec.size());
  }
  auto submesh_e_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(submesh_e_to_v_vec), std::move(submesh_e_to_v_offsets));

  // Create submesh topology
  mesh::Topology submesh_topology(mesh.comm(), entity_type);
  submesh_topology.set_index_map(0, submesh_vertex_index_map);
  submesh_topology.set_index_map(dim, submesh_entity_index_map);
  submesh_topology.set_connectivity(submesh_v_to_v, 0, 0);
  submesh_topology.set_connectivity(submesh_e_to_v, dim, 0);

  // Submesh geometry
  // Get the geometry dofs in the submesh based on the entities in
  // submesh
  xt::xtensor<std::int32_t, 2> e_to_g
      = mesh::entities_to_geometry(mesh, dim, entities, false);
  // FIXME Find better way to do this
  xt::xarray<int32_t> submesh_x_dofs_xt = xt::unique(e_to_g);
  std::vector<int32_t> submesh_x_dofs(submesh_x_dofs_xt.begin(),
                                      submesh_x_dofs_xt.end());

  auto mesh_geometry_dof_index_map = mesh.geometry().index_map();
  assert(mesh_geometry_dof_index_map);

  // Get the geometry dofs in the submesh owned by this process
  auto submesh_owned_x_dofs = dolfinx::common::compute_owned_indices(
      submesh_x_dofs, *mesh_geometry_dof_index_map);

  // Create submesh geometry index map
  std::pair<common::IndexMap, std::vector<int32_t>> submesh_x_dof_index_map_pair
      = mesh_geometry_dof_index_map->create_submap(submesh_owned_x_dofs);
  auto submesh_x_dof_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_x_dof_index_map_pair.first));

  // Create a map from the (local) geometry dofs in the submesh to the (local)
  // geometry dofs in the mesh.
  std::vector<int32_t> submesh_to_mesh_x_dof_map(submesh_owned_x_dofs.begin(),
                                                 submesh_owned_x_dofs.end());
  submesh_to_mesh_x_dof_map.reserve(submesh_x_dof_index_map->size_local()
                                    + submesh_x_dof_index_map->num_ghosts());
  std::transform(
      submesh_x_dof_index_map_pair.second.begin(),
      submesh_x_dof_index_map_pair.second.end(),
      std::back_inserter(submesh_to_mesh_x_dof_map),
      [mesh_geometry_dof_index_map](std::int32_t x_dof_index)
      { return mesh_geometry_dof_index_map->size_local() + x_dof_index; });

  // Create submesh geometry coordinates
  xtl::span<const double> mesh_x = mesh.geometry().x();
  const int submesh_num_x_dofs = submesh_to_mesh_x_dof_map.size();
  std::vector<double> submesh_x(3 * submesh_num_x_dofs);
  for (int i = 0; i < submesh_num_x_dofs; ++i)
  {
    common::impl::copy_N<3>(
        std::next(mesh_x.begin(), 3 * submesh_to_mesh_x_dof_map[i]),
        std::next(submesh_x.begin(), 3 * i));
  }

  // Crete submesh geometry dofmap
  std::vector<std::int32_t> submesh_x_dofmap_vec;
  submesh_x_dofmap_vec.reserve(e_to_g.shape()[0] * e_to_g.shape()[1]);
  std::vector<std::int32_t> submesh_x_dofmap_offsets(1, 0);
  submesh_x_dofmap_offsets.reserve(e_to_g.shape()[0] + 1);
  for (std::size_t i = 0; i < e_to_g.shape()[0]; ++i)
  {
    // Get the mesh geometry dofs for ith entity in entities
    auto entity_x_dofs = xt::row(e_to_g, i);

    // For each mesh dof of the entity, get the submesh dof
    for (std::int32_t x_dof : entity_x_dofs)
    {
      auto it = std::find(submesh_to_mesh_x_dof_map.begin(),
                          submesh_to_mesh_x_dof_map.end(), x_dof);
      assert(it != submesh_to_mesh_x_dof_map.end());
      submesh_x_dofmap_vec.push_back(
          std::distance(submesh_to_mesh_x_dof_map.begin(), it));
    }
    submesh_x_dofmap_offsets.push_back(submesh_x_dofmap_vec.size());
  }
  graph::AdjacencyList<std::int32_t> submesh_x_dofmap(
      std::move(submesh_x_dofmap_vec), std::move(submesh_x_dofmap_offsets));

  // Create submesh coordinate element
  CellType submesh_coord_cell
      = mesh::cell_entity_type(mesh.geometry().cmap().cell_shape(), dim, 0);
  auto submesh_coord_ele = fem::CoordinateElement(
      submesh_coord_cell, mesh.geometry().cmap().degree());

  // Submesh geometry input_global_indices
  // TODO Check this
  const std::vector<std::int64_t>& mesh_igi
      = mesh.geometry().input_global_indices();
  std::vector<std::int64_t> submesh_igi;
  submesh_igi.reserve(submesh_to_mesh_x_dof_map.size());
  std::transform(submesh_to_mesh_x_dof_map.begin(),
                 submesh_to_mesh_x_dof_map.end(),
                 std::back_inserter(submesh_igi),
                 [&mesh_igi](std::int32_t submesh_x_dof)
                 { return mesh_igi[submesh_x_dof]; });

  // Create geometry
  mesh::Geometry submesh_geometry(
      submesh_x_dof_index_map, std::move(submesh_x_dofmap), submesh_coord_ele,
      std::move(submesh_x), mesh.geometry().dim(), std::move(submesh_igi));

  return {Mesh(mesh.comm(), std::move(submesh_topology),
               std::move(submesh_geometry)),
          std::move(submesh_to_mesh_vertex_map),
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
