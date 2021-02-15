// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "MeshTags.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/partition.h>
#include <stdexcept>
#include <unordered_set>

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

  return graph::build_adjacency_list<std::int64_t>(std::move(topology),
                                                   num_vertices_per_cell);
}
//-----------------------------------------------------------------------------
std::vector<double> mesh::h(const Mesh& mesh,
                            const tcb::span<const std::int32_t>& entities,
                            int dim)
{
  if (dim != mesh.topology().dim())
    throw std::runtime_error("Cell size when dim ne tdim  requires updating.");

  // Get number of cell vertices
  const mesh::CellType type
      = cell_entity_type(mesh.topology().cell_type(), dim);
  const int num_vertices = num_cell_vertices(type);

  // Get geometry dofmap and dofs
  const mesh::Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();
  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      geom_dofs(geometry.x().data(), geometry.x().shape[0], geometry.x().shape[1]);

  std::vector<double> h_cells(entities.size(), 0);
  assert(num_vertices <= 8);
  std::array<Eigen::Vector3d, 8> points;
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    // Get the coordinates  of the vertices
    auto dofs = x_dofs.links(entities[e]);
    for (int i = 0; i < num_vertices; ++i)
      points[i] = geom_dofs.row(dofs[i]);

    // Get maximum edge length
    for (int i = 0; i < num_vertices; ++i)
    {
      for (int j = i + 1; j < num_vertices; ++j)
        h_cells[e] = std::max(h_cells[e], (points[i] - points[j]).norm());
    }
  }

  return h_cells;
}
//-----------------------------------------------------------------------------
common::array2d<double>
mesh::cell_normals(const mesh::Mesh& mesh, int dim,
                   const tcb::span<const std::int32_t>& entities)
{
  const int gdim = mesh.geometry().dim();
  const mesh::CellType type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim);

  // Find geometry nodes for topology entities
  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      geom_dofs(mesh.geometry().x().data(), mesh.geometry().x().shape[0],
                mesh.geometry().x().shape[1]);

  // Orient cells if they are tetrahedron
  bool orient = false;
  if (mesh.topology().cell_type() == mesh::CellType::tetrahedron)
    orient = true;
  common::array2d<std::int32_t> geometry_entities
      = entities_to_geometry(mesh, dim, entities, orient);

  const std::int32_t num_entities = entities.size();
  common::array2d<double> _n(num_entities, 3);

  Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> n(
      _n.data(), _n.shape[0], _n.shape[1]);
  switch (type)
  {
  case (mesh::CellType::interval):
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    for (int i = 0; i < num_entities; ++i)
    {
      // Get the two vertices as points
      auto vertices = geometry_entities.row(i);
      Eigen::Vector3d p0 = geom_dofs.row(vertices[0]);
      Eigen::Vector3d p1 = geom_dofs.row(vertices[1]);
      // Define normal by rotating tangent counter-clockwise
      Eigen::Vector3d t = p1 - p0;
      n.row(i) = Eigen::Vector3d(-t[1], t[0], 0.0).normalized();
    }
    return _n;
  }
  case (mesh::CellType::triangle):
  {
    for (int i = 0; i < num_entities; ++i)
    {
      // Get the three vertices as points
      auto vertices = geometry_entities.row(i);
      const Eigen::Vector3d p0 = geom_dofs.row(vertices[0]);
      const Eigen::Vector3d p1 = geom_dofs.row(vertices[1]);
      const Eigen::Vector3d p2 = geom_dofs.row(vertices[2]);

      // Define cell normal via cross product of first two edges
      n.row(i) = ((p1 - p0).cross(p2 - p0)).normalized();
    }
    return _n;
  }
  case (mesh::CellType::quadrilateral):
  {
    // TODO: check
    for (int i = 0; i < num_entities; ++i)
    {
      // Get three vertices as points
      auto vertices = geometry_entities.row(i);
      const Eigen::Vector3d p0 = geom_dofs.row(vertices[0]);
      const Eigen::Vector3d p1 = geom_dofs.row(vertices[1]);
      const Eigen::Vector3d p2 = geom_dofs.row(vertices[2]);

      // Defined cell normal via cross product of first two edges:
      n.row(i) = ((p1 - p0).cross(p2 - p0)).normalized();
    }
    return _n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }

  return common::array2d<double>(0, 3);
}
//-----------------------------------------------------------------------------
common::array2d<double>
mesh::midpoints(const mesh::Mesh& mesh, int dim,
                const tcb::span<const std::int32_t>& entities)
{
  const mesh::Geometry& geometry = mesh.geometry();

  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      x(geometry.x().data(), geometry.x().shape[0], geometry.x().shape[1]);

  // Build map from entity -> geometry dof
  // FIXME: This assumes a linear geometry.
  common::array2d<std::int32_t> entity_to_geometry
      = entities_to_geometry(mesh, dim, entities, false);

  common::array2d<double> midpoints(entities.size(), 3);
  Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> x_mid(
      midpoints.data(), entities.size(), 3);

  for (std::size_t e = 0; e < entity_to_geometry.shape[0]; ++e)
  {
    auto entity_vertices = entity_to_geometry.row(e);
    x_mid.row(e) = 0.0;
    for (std::size_t v = 0; v < entity_vertices.size(); ++v)
      x_mid.row(e) += x.row(entity_vertices[v]);
    x_mid.row(e) /= entity_vertices.size();
  }

  return midpoints;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities(
    const mesh::Mesh& mesh, int dim,
    const std::function<std::vector<bool>(const common::array2d<double>&)>&
        marker)
{
  const mesh::Topology& topology = mesh.topology();
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
  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      x_nodes(mesh.geometry().x().data(), mesh.geometry().x().shape[0],
              mesh.geometry().x().shape[1]);

  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x_vertices(
      3, vertex_to_node.size());
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
    x_vertices.col(i) = x_nodes.row(vertex_to_node[i]);

  common::array2d<double> verts(x_vertices);

  // Run marker function on vertex coordinates
  const std::vector<bool> marked = marker(verts);
  if ((int)marked.size() != x_vertices.cols())
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
    const mesh::Mesh& mesh, int dim,
    const std::function<std::vector<bool>(const common::array2d<double>&)>&
        marker)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute marker for boundary facets
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  const std::vector boundary_facet = mesh::compute_boundary_facets(topology);

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

  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      x_nodes(mesh.geometry().x().data(), mesh.geometry().x().shape[0],
              mesh.geometry().x().shape[1]);

  // Build vector of boundary vertices
  const std::vector<std::int32_t> vertices(boundary_vertices.begin(),
                                           boundary_vertices.end());

  // Get all vertex 'node' indices
  auto v_to_c = topology.connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x_vertices(
      3, vertices.size());
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
    x_vertices.col(i) = x_nodes.row(dofs[local_pos]);

    vertex_to_pos[v] = i;
  }

  common::array2d<double> verts(x_vertices);

  // Run marker function on the vertex coordinates
  const std::vector<bool> marked = marker(verts);
  if ((int)marked.size() != x_vertices.cols())
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
common::array2d<std::int32_t>
mesh::entities_to_geometry(const mesh::Mesh& mesh, int dim,
                           const tcb::span<const std::int32_t>& entity_list,
                           bool orient)
{
  dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  int num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, dim));
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      entity_geometry(entity_list.size(), num_entity_vertices);

  if (orient
      and (cell_type != dolfinx::mesh::CellType::tetrahedron or dim != 2))
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");

  const mesh::Geometry& geometry = mesh.geometry();

  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      geom_dofs(geometry.x().data(), geometry.x().shape[0], geometry.x().shape[1]);

  const mesh::Topology& topology = mesh.topology();

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
    assert((int)ev.size() == num_entity_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (int j = 0; j < num_entity_vertices; ++j)
    {
      int k = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), ev[j]));
      assert(k < (int)cv.size());
      entity_geometry(i, j) = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      Eigen::Vector3d midpoint(0.0, 0.0, 0.0);
      for (auto j : xc)
        midpoint += geom_dofs.row(j).matrix().transpose();
      midpoint /= xc.size();

      // Compute vector triple product of two edges and vector to midpoint
      Eigen::Vector3d p0 = geom_dofs.row(entity_geometry(i, 0));
      Eigen::Matrix3d a;
      a.row(0) = midpoint - p0;
      a.row(1) = geom_dofs.row(entity_geometry(i, 1)).matrix().transpose() - p0;
      a.row(2) = geom_dofs.row(entity_geometry(i, 2)).matrix().transpose() - p0;

      // Midpoint direction should be opposite to normal, hence this
      // should be negative. Switch points if not.
      if (a.determinant() > 0.0)
        std::swap(entity_geometry(i, 1), entity_geometry(i, 2));
    }
  }
  common::array2d<std::int32_t> ent_geom(entity_geometry);
  return ent_geom;
}
//------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Mesh& mesh)
{
  // Note: Possible duplication of mesh::Topology::compute_boundary_facets

  const mesh::Topology& topology = mesh.topology();
  std::vector<std::int32_t> surface_facets;

  // Get number of facets owned by this process
  const int tdim = topology.dim();
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(topology.index_map(tdim - 1));
  std::set<std::int32_t> fwd_shared_facets;

  // Only need to consider shared facets when there are no ghost cells
  if (topology.index_map(tdim)->num_ghosts() == 0)
  {
    fwd_shared_facets.insert(
        topology.index_map(tdim - 1)->shared_indices().begin(),
        topology.index_map(tdim - 1)->shared_indices().end());
  }

  // Find all owned facets (not ghost) with only one attached cell, which are
  // also not shared forward (ghost on another process)
  const int num_facets = topology.index_map(tdim - 1)->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1
        and fwd_shared_facets.find(f) == fwd_shared_facets.end())
      surface_facets.push_back(f);
  }

  return surface_facets;
}
//------------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> mesh::partition_cells_graph(
    MPI_Comm comm, int n, const mesh::CellType cell_type,
    const graph::AdjacencyList<std::int64_t>& cells, mesh::GhostMode ghost_mode)
{
  return partition_cells_graph(comm, n, cell_type, cells, ghost_mode,
                               &graph::partition_graph);
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> mesh::partition_cells_graph(
    MPI_Comm comm, int n, const mesh::CellType cell_type,
    const graph::AdjacencyList<std::int64_t>& cells, mesh::GhostMode ghost_mode,
    const graph::partition_fn& partfn)
{
  LOG(INFO) << "Compute partition of cells across ranks";

  if (cells.num_nodes() > 0)
  {
    if (cells.num_links(0) != mesh::num_cell_vertices(cell_type))
    {
      throw std::runtime_error(
          "Inconsistent number of cell vertices. Got "
          + std::to_string(cells.num_links(0)) + ", expected "
          + std::to_string(mesh::num_cell_vertices(cell_type)) + ".");
    }
  }

  // Compute distributed dual graph (for the cells on this process)
  const auto [dual_graph, graph_info]
      = mesh::build_dual_graph(comm, cells, cell_type);

  // Extract data from graph_info
  const auto [num_ghost_nodes, num_local_edges] = graph_info;

  // Just flag any kind of ghosting for now
  bool ghosting = (ghost_mode != mesh::GhostMode::none);

  // Compute partition
  return partfn(comm, n, dual_graph, num_ghost_nodes, ghosting);
}
//-----------------------------------------------------------------------------
