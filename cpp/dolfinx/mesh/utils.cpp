// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "Mesh.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/partition.h>
#include <stdexcept>
#include <unordered_set>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::extract_topology(const CellType& cell_type,
                       const fem::ElementDofLayout& layout,
                       const graph::AdjacencyList<std::int64_t>& cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int> local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology(cells.num_nodes() * num_vertices_per_cell);
  for (int c = 0; c < cells.num_nodes(); ++c)
  {
    auto p = cells.links(c);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      topology[num_vertices_per_cell * c + j] = p[local_vertices[j]];
  }

  return graph::regular_adjacency_list(std::move(topology),
                                       num_vertices_per_cell);
}
//-----------------------------------------------------------------------------
std::vector<double> mesh::h(const Mesh& mesh,
                            const xtl::span<const std::int32_t>& entities,
                            int dim)
{
  if (dim != mesh.topology().dim())
    throw std::runtime_error("Cell size when dim ne tdim  requires updating.");

  if (mesh.topology().cell_type() == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  // Get number of cell vertices
  const CellType type = cell_entity_type(mesh.topology().cell_type(), dim, 0);
  const int num_vertices = num_cell_vertices(type);

  // Get geometry dofmap and dofs
  const Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();
  auto geom_dofs
      = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector({geometry.x().size() / 3, std::size_t(3)}));
  std::vector<double> h_cells(entities.size(), 0);
  assert(num_vertices <= 8);
  xt::xtensor_fixed<double, xt::xshape<8, 3>> points;
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    // Get the coordinates  of the vertices
    auto dofs = x_dofs.links(entities[e]);

    // The below should work, but misbehaves with the Intel icpx compiler
    // xt::view(points, xt::range(0, num_vertices), xt::all())
    //     = xt::view(geom_dofs, xt::keep(dofs), xt::all());
    auto points_view = xt::view(points, xt::range(0, num_vertices), xt::all());
    points_view.assign(xt::view(geom_dofs, xt::keep(dofs), xt::all()));

    // Get maximum edge length
    for (int i = 0; i < num_vertices; ++i)
    {
      for (int j = i + 1; j < num_vertices; ++j)
      {
        auto p0 = xt::row(points, i);
        auto p1 = xt::row(points, j);
        h_cells[e] = std::max(h_cells[e], xt::norm_l2(p0 - p1)());
      }
    }
  }

  return h_cells;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
mesh::cell_normals(const mesh::Mesh& mesh, int dim,
                   const xtl::span<const std::int32_t>& entities)
{
  if (mesh.topology().cell_type() == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(mesh.topology().cell_type(), dim, 0);

  // Find geometry nodes for topology entities
  auto xg = xt::adapt(
      mesh.geometry().x().data(), mesh.geometry().x().size(),
      xt::no_ownership(),
      std::vector({mesh.geometry().x().size() / 3, std::size_t(3)}));

  // Orient cells if they are tetrahedron
  bool orient = false;
  if (mesh.topology().cell_type() == CellType::tetrahedron)
    orient = true;
  xt::xtensor<std::int32_t, 2> geometry_entities
      = entities_to_geometry(mesh, dim, entities, orient);

  const std::size_t num_entities = entities.size();
  xt::xtensor<double, 2> n({num_entities, 3});
  switch (type)
  {
  case CellType::interval:
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get the two vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);

      // Define normal by rotating tangent counter-clockwise
      auto t = p1 - p0;
      auto ni = xt::row(n, i);
      ni[0] = -t[1];
      ni[1] = t[0];
      ni[2] = 0.0;
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  case CellType::triangle:
  {
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get the three vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);
      auto p2 = xt::row(xg, vertices[2]);

      // Define cell normal via cross product of first two edges
      auto ni = xt::row(n, i);
      ni = math::cross((p1 - p0), (p2 - p0));
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  case CellType::quadrilateral:
  {
    // TODO: check
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get three vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);
      auto p2 = xt::row(xg, vertices[2]);

      // Defined cell normal via cross product of first two edges:
      auto ni = xt::row(n, i);
      ni = math::cross((p1 - p0), (p2 - p0));
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
mesh::compute_midpoints(const Mesh& mesh, int dim,
                        const xtl::span<const std::int32_t>& entities)
{
  auto x = xt::adapt(
      mesh.geometry().x().data(), mesh.geometry().x().size(),
      xt::no_ownership(),
      std::vector({mesh.geometry().x().size() / 3, std::size_t(3)}));

  // Build map from entity -> geometry dof
  // FIXME: This assumes a linear geometry.
  xt::xtensor<std::int32_t, 2> entity_to_geometry
      = entities_to_geometry(mesh, dim, entities, false);

  xt::xtensor<double, 2> x_mid({entities.size(), 3});
  for (std::size_t e = 0; e < entity_to_geometry.shape(0); ++e)
  {
    auto rows = xt::row(entity_to_geometry, e);
    // The below should work, but misbehaves with the Intel icpx compiler
    // xt::row(x_mid, e) = xt::mean(xt::view(x, xt::keep(rows)), 0);
    auto _x = xt::row(x_mid, e);
    _x.assign(xt::mean(xt::view(x, xt::keep(rows)), 0));
  }

  return x_mid;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities(
    const Mesh& mesh, int dim,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker)
{
  const Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable().create_connectivity(dim, 0);

  // Get all vertex 'node' indices
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::int32_t num_vertices = topology.index_map(0)->size_local()
                                    + topology.index_map(0)->num_ghosts();
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    auto vertices = c_to_v->links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_node[vertices[i]] = x_dofs[i];
  }

  // Pack coordinates of vertices
  xtl::span<const double> x_nodes = mesh.geometry().x();
  xt::xtensor<double, 2> x_vertices({3, vertex_to_node.size()});
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
  {
    const int pos = 3 * vertex_to_node[i];
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices(j, i) = x_nodes[pos + j];
  }

  // Run marker function on vertex coordinates
  const xt::xtensor<bool, 1> marked = marker(x_vertices);
  if (marked.shape(0) != x_vertices.shape(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Iterate over entities to build vector of marked entities
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (int e = 0; e < e_to_v->num_nodes(); ++e)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
    for (std::int32_t v : e_to_v->links(e))
    {
      if (!marked[v])
      {
        all_vertices_marked = false;
        break;
      }
    }

    if (all_vertices_marked)
      entities.push_back(e);
  }

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities_boundary(
    const Mesh& mesh, int dim,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker)
{
  const Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute marker for boundary facets
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  const std::vector boundary_facet = compute_boundary_facets(topology);

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, 0);
  mesh.topology_mutable().create_connectivity(0, tdim);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  // Build set of vertices on boundary and set of boundary entities
  auto f_to_v = topology.connectivity(tdim - 1, 0);
  assert(f_to_v);
  auto f_to_e = topology.connectivity(tdim - 1, dim);
  assert(f_to_e);
  std::unordered_set<std::int32_t> boundary_vertices;
  std::unordered_set<std::int32_t> facet_entities;
  for (std::size_t f = 0; f < boundary_facet.size(); ++f)
  {
    if (boundary_facet[f])
    {
      for (auto e : f_to_e->links(f))
        facet_entities.insert(e);

      for (auto v : f_to_v->links(f))
        boundary_vertices.insert(v);
    }
  }

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  xtl::span<const double> x_nodes = mesh.geometry().x();

  // Build vector of boundary vertices
  const std::vector<std::int32_t> vertices(boundary_vertices.begin(),
                                           boundary_vertices.end());

  // Get all vertex 'node' indices
  auto v_to_c = topology.connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  xt::xtensor<double, 2> x_vertices({3, vertices.size()});
  std::vector<std::int32_t> vertex_to_pos(v_to_c->num_nodes(), -1);
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t v = vertices[i];

    // Get first cell and find position
    const int c = v_to_c->links(v)[0];
    auto vertices = c_to_v->links(c);
    auto it = std::find(vertices.begin(), vertices.end(), v);
    assert(it != vertices.end());
    const int local_pos = std::distance(vertices.begin(), it);

    auto dofs = x_dofmap.links(c);
    for (int j = 0; j < 3; ++j)
      x_vertices(j, i) = x_nodes[3 * dofs[local_pos] + j];

    vertex_to_pos[v] = i;
  }

  // Run marker function on the vertex coordinates
  const xt::xtensor<bool, 1> marked = marker(x_vertices);
  if (marked.shape(0) != x_vertices.shape(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Loop over entities and check vertex markers
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (auto e : facet_entities)
  {
    // Assume all vertices on this entity are marked
    bool all_vertices_marked = true;

    // Iterate over entity vertices
    for (auto v : e_to_v->links(e))
    {
      const std::int32_t pos = vertex_to_pos[v];
      if (!marked[pos])
      {
        all_vertices_marked = false;
        break;
      }
    }

    // Mark facet with all vertices marked
    if (all_vertices_marked)
      entities.push_back(e);
  }

  return entities;
}
//-----------------------------------------------------------------------------
xt::xtensor<std::int32_t, 2>
mesh::entities_to_geometry(const Mesh& mesh, int dim,
                           const xtl::span<const std::int32_t>& entity_list,
                           bool orient)
{
  CellType cell_type = mesh.topology().cell_type();
  if (cell_type == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cells");

  const std::size_t num_entity_vertices
      = num_cell_vertices(cell_entity_type(cell_type, dim, 0));
  xt::xtensor<std::int32_t, 2> entity_geometry(
      {entity_list.size(), num_entity_vertices});

  if (orient and (cell_type != CellType::tetrahedron or dim != 2))
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");

  const Geometry& geometry = mesh.geometry();
  auto geom_dofs
      = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector({geometry.x().size() / 3, std::size_t(3)}));
  const Topology& topology = mesh.topology();

  const int tdim = topology.dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(dim, 0);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
  const auto e_to_c = topology.connectivity(dim, tdim);
  assert(e_to_c);
  const auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  const auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
  {
    const std::int32_t idx = entity_list[i];
    const std::int32_t cell = e_to_c->links(idx)[0];
    auto ev = e_to_v->links(idx);
    assert(ev.size() == num_entity_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (std::size_t j = 0; j < num_entity_vertices; ++j)
    {
      int k = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), ev[j]));
      assert(k < (int)cv.size());
      entity_geometry(i, j) = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      xt::xtensor_fixed<double, xt::xshape<3>> midpoint = {0, 0, 0};
      for (std::int32_t j : xc)
        for (int k = 0; k < 3; ++k)
          midpoint[k] += geom_dofs(j, k);
      midpoint /= xc.size();

      // Compute vector triple product of two edges and vector to midpoint
      auto p0 = xt::row(geom_dofs, entity_geometry(i, 0));
      auto p1 = xt::row(geom_dofs, entity_geometry(i, 1));
      auto p2 = xt::row(geom_dofs, entity_geometry(i, 2));

      xt::xtensor_fixed<double, xt::xshape<3, 3>> a;
      xt::row(a, 0) = midpoint - p0;
      xt::row(a, 1) = p1 - p0;
      xt::row(a, 2) = p2 - p0;

      // Midpoint direction should be opposite to normal, hence this
      // should be negative. Switch points if not.
      if (math::det(a) > 0.0)
        std::swap(entity_geometry(i, 1), entity_geometry(i, 2));
    }
  }

  return entity_geometry;
}
//------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Mesh& mesh)
{
  // Note: Possible duplication of mesh::Topology::compute_boundary_facets

  const Topology& topology = mesh.topology();
  std::vector<std::int32_t> surface_facets;

  // Get number of facets owned by this process
  const int tdim = topology.dim();
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(topology.index_map(tdim - 1));

  // Only need to consider shared facets when there are no ghost cells
  std::set<std::int32_t> fwd_shared_facets;
  if (topology.index_map(tdim)->num_ghosts() == 0)
  {
    fwd_shared_facets.insert(
        topology.index_map(tdim - 1)->scatter_fwd_indices().array().begin(),
        topology.index_map(tdim - 1)->scatter_fwd_indices().array().end());
  }

  // Find all owned facets (not ghost) with only one attached cell, which are
  // also not shared forward (ghost on another process)
  const int num_facets = topology.index_map(tdim - 1)->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1
        and fwd_shared_facets.find(f) == fwd_shared_facets.end())
    {
      surface_facets.push_back(f);
    }
  }

  return surface_facets;
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction
mesh::create_cell_partitioner(const graph::partition_fn& partfn)
{
  return
      [partfn](
          MPI_Comm comm, int nparts, int tdim,
          const graph::AdjacencyList<std::int64_t>& cells,
          GhostMode ghost_mode) -> dolfinx::graph::AdjacencyList<std::int32_t>
  {
    LOG(INFO) << "Compute partition of cells across ranks";

    // Compute distributed dual graph (for the cells on this process)
    const auto [dual_graph, num_ghost_edges]
        = build_dual_graph(comm, cells, tdim);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, num_ghost_edges, ghosting);
  };
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::compute_incident_entities(const Mesh& mesh,
                                const xtl::span<const std::int32_t>& entities,
                                int d0, int d1)
{
  auto map0 = mesh.topology().index_map(d0);
  if (!map0)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d0)
                             + " have not been created.");
  }

  auto map1 = mesh.topology().index_map(d1);
  if (!map1)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d1)
                             + " have not been created.");
  }

  auto e0_to_e1 = mesh.topology().connectivity(d0, d1);
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

  std::sort(entities1.begin(), entities1.end());
  entities1.erase(std::unique(entities1.begin(), entities1.end()),
                  entities1.end());
  return entities1;
}
//-----------------------------------------------------------------------------
