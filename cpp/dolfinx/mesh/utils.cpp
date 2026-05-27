// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<std::int64_t>
mesh::extract_topology(CellType cell_type, const fem::ElementDofLayout& layout,
                       std::span<const std::int64_t> cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  const int num_node_per_cell = layout.num_dofs();
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int>& local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology((cells.size() / num_node_per_cell)
                                     * num_vertices_per_cell);
  for (std::size_t c = 0; c < cells.size() / num_node_per_cell; ++c)
  {
    auto p = cells.subspan(c * num_node_per_cell, num_node_per_cell);
    std::span t(topology.data() + c * num_vertices_per_cell,
                num_vertices_per_cell);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      t[j] = p[local_vertices[j]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology,
                                                       int facet_type_idx)
{
  const int tdim = topology.dim();
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error(
        "Facet to cell connectivity has not been computed.");
  }

  // Find all owned facets (not ghost) with only one attached cell
  auto facet_map = topology.index_maps(tdim - 1).at(facet_type_idx);

  std::vector<std::int32_t> facets;
  for (std::int32_t f = 0; f < facet_map->size_local(); ++f)
  {
    if (f_to_c->num_links(f) == 1)
      facets.push_back(f);
  }

  // Remove facets on internal inter-process boundary
  std::vector<std::int32_t> ext_facets;
  std::ranges::set_difference(facets,
                              topology.interprocess_facets(facet_type_idx),
                              std::back_inserter(ext_facets));

  return ext_facets;
}
//------------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology)
{
  if (topology.entity_types(topology.dim() - 1).size() > 1)
  {
    throw std::runtime_error("Multiple facet types in mesh. Call "
                             "exterior_facet_indices with facet type index.");
  }

  return mesh::exterior_facet_indices(topology, 0);
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction mesh::create_cell_partitioner(
    mesh::GhostMode ghost_mode, const graph::partition_fn& partfn,
    std::optional<std::int32_t> max_facet_to_cell_links)
{
  return [partfn, ghost_mode, max_facet_to_cell_links](
             MPI_Comm comm, int nparts, const std::vector<CellType>& cell_types,
             const std::vector<std::span<const std::int64_t>>& cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    spdlog::info("Compute partition of cells across ranks");

    // Compute distributed dual graph (for the cells on this process)
    graph::AdjacencyList dual_graph
        = build_dual_graph(comm, cell_types, cells, max_facet_to_cell_links);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, ghosting);
  };
}
//-----------------------------------------------------------------------------
mesh::CellPartitionFunction mesh::create_cell_partitioner(
    mesh::GhostMode ghost_mode,
    std::optional<std::int32_t> max_facet_to_cell_links)
{
  return create_cell_partitioner(ghost_mode, &graph::partition_graph,
                                 max_facet_to_cell_links);
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::compute_incident_entities(const Topology& topology,
                                std::span<const std::int32_t> entities, int d0,
                                int d1)
{
  auto map0 = topology.index_map(d0);
  if (!map0)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d0)
                             + " have not been created.");
  }

  auto map1 = topology.index_map(d1);
  if (!map1)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d1)
                             + " have not been created.");
  }

  auto e0_to_e1 = topology.connectivity(d0, d1);
  if (!e0_to_e1)
  {
    throw std::runtime_error("Connectivity missing: (" + std::to_string(d0)
                             + ", " + std::to_string(d1) + ")");
  }

  std::vector<std::int32_t> entities1;
  for (std::int32_t entity : entities)
  {
    auto e = e0_to_e1->links(entity);
    entities1.insert(entities1.end(), e.begin(), e.end());
  }

  std::ranges::sort(entities1);
  auto [unique_end, range_end] = std::ranges::unique(entities1);
  entities1.erase(unique_end, range_end);

  return entities1;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mesh::Mesh<T> mesh::interpolate_geometry(
    std::shared_ptr<mesh::Mesh<T>> mesh,
    const fem::CoordinateElement<T>& new_cmap,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  assert(mesh);
  const fem::CoordinateElement<T>& old_cmap = mesh->geometry().cmap();
  if (new_cmap.cell_shape() != old_cmap.cell_shape())
  {
    throw std::runtime_error(
        "Cell shape of new coordinate element must match input mesh.");
  }

  const int gdim = mesh->geometry().dim();

  // Build a vector-valued Lagrange FiniteElement from the coordinate element.
  basix::FiniteElement<T> b_element = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(new_cmap.cell_shape()), new_cmap.degree(),
      new_cmap.variant(), basix::element::dpc_variant::unset, false);
  auto element = std::make_shared<const fem::FiniteElement<T>>(
      b_element, std::vector<std::size_t>{static_cast<std::size_t>(gdim)});

  fem::FunctionSpace<T> V
      = fem::create_functionspace(mesh, element, reorder_fn);

  // Tabulate physical coordinates of the new geometry dofs.
  std::vector<T> x_new = V.tabulate_dof_coordinates(false);

  // Pull the geometry dofmap and index map from V.
  std::shared_ptr<const fem::DofMap> dm = V.dofmap();
  assert(dm);
  std::shared_ptr<const common::IndexMap> new_imap = dm->index_map;
  assert(new_imap);

  auto map_view = dm->map();
  std::vector<std::int32_t> dofmap_flat(
      map_view.data_handle(), map_view.data_handle() + map_view.size());

  // Build input_global_indices as the local-to-global of the new geometry
  // dofs.
  const std::int32_t num_nodes
      = new_imap->size_local() + new_imap->num_ghosts();
  std::vector<std::int32_t> local(num_nodes);
  std::iota(local.begin(), local.end(), 0);
  std::vector<std::int64_t> igi(num_nodes);
  new_imap->local_to_global(local, igi);

  Geometry<T> geometry(
      new_imap, std::vector<std::vector<std::int32_t>>{std::move(dofmap_flat)},
      std::vector<fem::CoordinateElement<T>>{new_cmap}, std::move(x_new), gdim,
      std::move(igi));

  return Mesh<T>(mesh->comm(), mesh->topology(), std::move(geometry));
}
//-----------------------------------------------------------------------------
template mesh::Mesh<float> mesh::interpolate_geometry(
    std::shared_ptr<mesh::Mesh<float>>, const fem::CoordinateElement<float>&,
    const std::function<
        std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>&);
template mesh::Mesh<double> mesh::interpolate_geometry(
    std::shared_ptr<mesh::Mesh<double>>, const fem::CoordinateElement<double>&,
    const std::function<
        std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>&);
//-----------------------------------------------------------------------------
