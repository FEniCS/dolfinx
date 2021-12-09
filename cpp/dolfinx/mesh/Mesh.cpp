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
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>

#include "graphbuild.h"

#include <string>
#include <iostream>
#include <sstream>

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
  return create_mesh(
      comm, cells, element, x, ghost_mode,
      static_cast<graph::AdjacencyList<std::int32_t> (*)(
          MPI_Comm, int, int, const graph::AdjacencyList<std::int64_t>&,
          mesh::GhostMode)>(&mesh::partition_cells_graph));
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

  std::stringstream ss;

  int rank = dolfinx::MPI::rank(comm);
  ss << "rank = " << rank << "\n";

  ss << "mesh c_to_v = \n";
  for (auto cell = 0; cell < topology.connectivity(tdim, 0)->num_nodes(); ++cell)
  {
    ss << "   cell " << cell << ": ";
    for (auto v : topology.connectivity(tdim, 0)->links(cell))
    {
      ss << v << " ";
    }
    ss << "\n";
  }
  ss << "\n";

  ss << "mesh cell_nodes1 = ";
  for (auto n : cell_nodes1.array())
  {
    ss << n << " ";
  }
  ss << "\n";

  ss << "mesh x = ";
  ss << x << "\n";

  // std::cout << ss.str() << "\n";

  return Mesh(comm, std::move(topology),
              mesh::create_geometry(comm, topology, element, cell_nodes1, x));
}
//-----------------------------------------------------------------------------
Mesh Mesh::sub(int dim, const xtl::span<const std::int32_t>& entities)
{
  // TODO Specify sizes of vectors

  // Get the verticies in the submesh
  std::vector<std::int32_t> submesh_vertices
      = mesh::compute_incident_entities(*this, entities, dim, 0);

  // For some processes, entities will be empty, but they might still own vertices
  // in the submesh that need to be included.
  auto vertex_index_map = _topology.index_map(0);
  std::vector<int32_t> submesh_owned_vertices =
    dolfinx::common::get_owned_indices(comm(), submesh_vertices, vertex_index_map);

  // Create submesh vertex index map
  std::pair<common::IndexMap, std::vector<int32_t>>
      submesh_vertex_index_map_pair
      = vertex_index_map->create_submap(submesh_owned_vertices);
  auto submesh_vertex_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_vertex_index_map_pair.first));

  // Get the entities in the submesh that are owned by this process
  auto entity_index_map = _topology.index_map(dim);
  std::vector<std::int32_t> submesh_owned_entities;  // Local numbering
  for (auto e : entities)
  {
    if (e < entity_index_map->size_local())
    {
      submesh_owned_entities.push_back(e);
    }
  }

  // Create submesh entity index map
  // NOTE Here it is assumed that every ghost entity in entities corresponds
  // to an owned entity in entities on another process. This means that
  // the ghost entities don't need to be communicated back to their owners,
  // although it is simple to enable this functionality.
  std::pair<common::IndexMap, std::vector<int32_t>>
      submesh_entity_index_map_pair =
        entity_index_map->create_submap(submesh_owned_entities);
  auto submesh_entity_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_entity_index_map_pair.first));

  // Submesh vertex to vertex connectivity (identity)
  auto submesh_v_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      submesh_vertex_index_map->size_local()
      + submesh_vertex_index_map->num_ghosts());

  // Submesh entity to vertex connectivity
  auto e_to_v = _topology.connectivity(dim, 0);
  std::vector<std::int32_t> submesh_e_to_v_vec;
  std::vector<std::int32_t> submesh_e_to_v_offsets(1, 0);
  for (std::int32_t e : entities)
  {
    xtl::span<const std::int32_t> vertices = e_to_v->links(e);

    for (std::int32_t v : vertices)
    {
      auto it
          = std::find(submesh_vertices.begin(), submesh_vertices.end(), v);
      assert(it != submesh_vertices.end());
      submesh_e_to_v_vec.push_back(std::distance(submesh_vertices.begin(), it));
    }
    submesh_e_to_v_offsets.push_back(submesh_e_to_v_vec.size());
  }
  auto submesh_e_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      submesh_e_to_v_vec, submesh_e_to_v_offsets);

  // Create submesh topology
  const CellType entity_type
      = mesh::cell_entity_type(_topology.cell_type(), dim, 0);
  mesh::Topology submesh_topology(comm(), entity_type);
  submesh_topology.set_index_map(0, submesh_vertex_index_map);
  submesh_topology.set_index_map(dim, submesh_entity_index_map);
  submesh_topology.set_connectivity(submesh_v_to_v, 0, 0);
  submesh_topology.set_connectivity(submesh_e_to_v, dim, 0);

  // Geometry
  auto e_to_g = mesh::entities_to_geometry(*this, dim, entities, false);
  
  // ss << "e_to_g = \n";
  // ss << e_to_g << "\n";
  
  xt::xarray<int32_t> submesh_x_dofs_xt = xt::unique(e_to_g);
  // FIXME Find better way
  std::vector<int32_t> submesh_x_dofs(submesh_x_dofs_xt.begin(),
                                      submesh_x_dofs_xt.end());
  
  std::shared_ptr<const dolfinx::common::IndexMap> geometry_dof_index_map
      = this->geometry().index_map();

  auto submesh_owned_x_dofs = dolfinx::common::get_owned_indices(
    comm(), submesh_x_dofs, geometry_dof_index_map);

  // ss << "submesh_owned_x_dofs = ";
  // for (auto x_dof : submesh_owned_x_dofs)
  // {
  //   ss << x_dof << " ";
  // }
  // ss << "\n";

  // Create submesh geometry x_dof index map
  std::pair<common::IndexMap, std::vector<int32_t>>
      submesh_x_dof_index_map_pair
      = geometry_dof_index_map->create_submap(submesh_owned_x_dofs);
  auto submesh_x_dof_index_map = std::make_shared<common::IndexMap>(
      std::move(submesh_x_dof_index_map_pair.first));
  auto submesh_x_dof_index_map_ghost_map =
      std::move(submesh_x_dof_index_map_pair.second);

  // ss << "submesh_x_dof_index_map->global_indices() = ";
  // for (auto i : submesh_x_dof_index_map->global_indices())
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  // ss << "submesh_x_dof_index_map->ghosts() = ";
  // for (auto i : submesh_x_dof_index_map->ghosts())
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  // ss << "submesh_x_dof_index_map_ghost_map = ";
  // for (auto i : submesh_x_dof_index_map_ghost_map)
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  // Ghosts in the (this) mesh that are present in the submap (global index)
  std::vector<std::int64_t> x_dof_ghosts;
  for (int i = 0; i < submesh_x_dof_index_map->ghosts().size(); ++i)
  {
    x_dof_ghosts.push_back(
      geometry_dof_index_map->ghosts()[submesh_x_dof_index_map_ghost_map[i]]);
  }

  // ss << "x_dof_ghosts = ";
  // for (auto x_dof_ghost : x_dof_ghosts)
  // {
  //   ss << x_dof_ghost << " ";
  // }
  // ss << "\n";

  // Get local index of ghost dofs in (this) mesh
  std::vector<std::int32_t> x_dof_ghosts_local;
  auto geometry_dof_index_map_global_indices =
      geometry_dof_index_map->global_indices();
  for (auto x_dof_ghost : x_dof_ghosts)
  {
    auto it = std::find(geometry_dof_index_map_global_indices.begin(),
                        geometry_dof_index_map_global_indices.end(), x_dof_ghost);
    assert(it != geometry_dof_index_map_global_indices.end());
    x_dof_ghosts_local.push_back(
        std::distance(geometry_dof_index_map_global_indices.begin(), it));
  }

  // ss << "x_dof_ghosts_local = ";
  // for (auto i : x_dof_ghosts_local)
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  // Add any owned entities in the submap that might come from other processes
  submesh_x_dofs.insert(submesh_x_dofs.end(),
                        submesh_owned_x_dofs.begin(),
                        submesh_owned_x_dofs.end());

  // Since the submesh must contain the coordinates of all dofs in the index map
  // (owned and ghost), the ghost dofs must also be added, as they might not be
  // present on, for instance, processes with no entities.
  submesh_x_dofs.insert(submesh_x_dofs.end(),
                        x_dof_ghosts_local.begin(),
                        x_dof_ghosts_local.end());

  // Create a sorted list of unique x_dofs. submesh_x_dofs maps the submesh x_dof
  // to the mesh (this) x_dof i.e. x_dof 3 in the submesh is x_dof submesh_x_dofs[3]
  // in the mesh (this)
  std::sort(submesh_x_dofs.begin(), submesh_x_dofs.end());
  submesh_x_dofs.erase(std::unique(submesh_x_dofs.begin(),
                                   submesh_x_dofs.end()),
                                   submesh_x_dofs.end());

  // ss << "submesh_x_dofs = ";
  // for (auto i : submesh_x_dofs)
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  // Crete submesh geometry dofmap
  std::vector<std::int32_t> submesh_x_dofmap_vec;
  std::vector<std::int32_t> submesh_x_dofmap_offsets(1, 0);
  for (std::size_t i = 0; i < e_to_g.shape()[0]; ++i)
  {
    auto entity_x_dofs = xt::row(e_to_g, i);

    // Loop through entity_x_dofs, getting the corresponding x_dof in
    // the submesh
    // TODO Create outside loop and fill to reuse
    std::vector<std::int32_t> submesh_entity_x_dofs;
    for (auto x_dof : entity_x_dofs)
    {
      auto it = std::find(submesh_x_dofs.begin(),
                          submesh_x_dofs.end(), x_dof);
      assert(it != submesh_x_dofs.end());
      auto submesh_entity_x_dof
          = std::distance(submesh_x_dofs.begin(), it);
      submesh_entity_x_dofs.push_back(submesh_entity_x_dof);
    }
    submesh_x_dofmap_vec.insert(submesh_x_dofmap_vec.end(),
                                submesh_entity_x_dofs.begin(),
                                submesh_entity_x_dofs.end());
    submesh_x_dofmap_offsets.push_back(submesh_x_dofmap_vec.size());
  }
  graph::AdjacencyList<std::int32_t> submesh_x_dofmap(
      std::move(submesh_x_dofmap_vec), std::move(submesh_x_dofmap_offsets));
  
  // ss << "submesh_x_dofmap =\n";
  // for (auto cell = 0; cell < submesh_x_dofmap.num_nodes(); ++cell)
  // {
  //   ss << "cell " << cell << ": ";
  //   for (auto dof : submesh_x_dofmap.links(cell))
  //   {
  //     ss << dof << " ";
  //   }
  //   ss << "\n";
  // }
  // ss << "\n";

  // Create submesh geometry coordinates
  const int submesh_num_x_dofs = submesh_x_dofs.size();
  const int gdim = this->geometry().dim();
  xt::xarray<double> submesh_x
      = xt::zeros<double>({submesh_num_x_dofs, gdim});
  const xt::xtensor<double, 2>& x = geometry().x();
  for (int i = 0; i < submesh_num_x_dofs; ++i)
  {
    xt::view(submesh_x, i, xt::all())
        = xt::view(x, submesh_x_dofs[i], xt::range(0, gdim));
  }

  // ss << "submesh_x = \n";
  // ss << submesh_x << "\n";

  // Create submesh coordinate element
  CellType submesh_coord_cell
      = mesh::cell_entity_type(geometry().cmap().cell_shape(), dim, 0);
  // FIXME Currently geometry degree is hardcoded to 1 as there is no way to
  // retrive this from the coordinate element
  auto submesh_coord_ele = fem::CoordinateElement(submesh_coord_cell, 1);

  // ss << "Created coordinate element\n";


  // Submesh geometry input_global_indices
  // TODO Check this. Not 100 % sure what igi does
  const std::vector<std::int64_t>& igi = this->geometry().input_global_indices();
  std::vector<std::int64_t> submesh_igi;

  for (auto submesh_x_dof : submesh_x_dofs)
  {
    submesh_igi.push_back(igi[submesh_x_dof]);
  }

  // ss << "submesh_igi = ";
  // for (auto i : submesh_igi)
  // {
  //   ss << i << " ";
  // }
  // ss << "\n";

  mesh::Geometry submesh_geometry(submesh_x_dof_index_map,
                                  std::move(submesh_x_dofmap),
                                  submesh_coord_ele,
                                  std::move(submesh_x),
                                  std::move(submesh_igi));
  
  // ss << "Geom created\n";

  // std::cout << ss.str() << "\n";

  return Mesh(comm(), std::move(submesh_topology), std::move(submesh_geometry));
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
