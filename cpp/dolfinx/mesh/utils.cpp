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
#include <array>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <stdexcept>
#include <vector>

using namespace dolfinx;
namespace stdex = std::experimental;
using cmdspan3x_t
    = stdex::mdspan<const double,
                    stdex::extents<std::size_t, 3, stdex::dynamic_extent>>;

namespace
{

/// The coordinates for all 'vertices' in the mesh
/// @param[in] mesh The mesh to compute the vertex coordinates for
/// @return The vertex coordinates. The shape is `(3, num_vertices)` and
/// the jth column hold the coordinates of vertex j.
std::pair<std::vector<double>, std::array<std::size_t, 2>>
compute_vertex_coords(const mesh::Mesh& mesh)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Create entities and connectivities
  mesh.topology_mutable().create_connectivity(tdim, 0);

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
  std::span<const double> x_nodes = mesh.geometry().x();
  std::vector<double> x_vertices(3 * vertex_to_node.size(), 0.0);
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
  {
    const int pos = 3 * vertex_to_node[i];
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices[j * vertex_to_node.size() + i] = x_nodes[pos + j];
  }

  return {std::move(x_vertices), {3, vertex_to_node.size()}};
}

/// The coordinates of 'vertices' for for entities of a give dimension
/// that are attached to specified facets.
///
/// @pre The provided facets must be on the boundary of the mesh.
///
/// @param[in] mesh The mesh to compute the vertex coordinates for
/// @param[in] dim The topological dimension of the entities
/// @param[in] facets List of facets on the meh boundary
/// @return (0) The entities attached to the boundary facets, (1) vertex
/// coordinates (shape is `(3, num_vertices)`) and (2) map from vertex
/// in the full mesh to the position (column) in the vertex coordinates
/// array (set to -1 if vertex in full mesh is not in the coordinate
/// array).
std::tuple<std::vector<std::int32_t>, std::vector<double>,
           std::vector<std::int32_t>>
compute_vertex_coords_boundary(const mesh::Mesh& mesh, int dim,
                               std::span<const std::int32_t> facets)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Build set of vertices on boundary and set of boundary entities
  mesh.topology_mutable().create_connectivity(tdim - 1, 0);
  mesh.topology_mutable().create_connectivity(tdim - 1, dim);
  std::vector<std::int32_t> vertices, entities;
  {
    auto f_to_v = topology.connectivity(tdim - 1, 0);
    assert(f_to_v);
    auto f_to_e = topology.connectivity(tdim - 1, dim);
    assert(f_to_e);
    for (auto f : facets)
    {
      auto v = f_to_v->links(f);
      vertices.insert(vertices.end(), v.begin(), v.end());
      auto e = f_to_e->links(f);
      entities.insert(entities.end(), e.begin(), e.end());
    }

    // Build vector of boundary vertices
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()),
                   vertices.end());
    std::sort(entities.begin(), entities.end());
    entities.erase(std::unique(entities.begin(), entities.end()),
                   entities.end());
  }

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  std::span<const double> x_nodes = mesh.geometry().x();

  // Get all vertex 'node' indices
  mesh.topology_mutable().create_connectivity(0, tdim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  auto v_to_c = topology.connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<double> x_vertices(3 * vertices.size(), -1.0);
  std::vector<std::int32_t> vertex_to_pos(v_to_c->num_nodes(), -1);
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t v = vertices[i];

    // Get first cell and find position
    const int c = v_to_c->links(v).front();
    auto cell_vertices = c_to_v->links(c);
    auto it = std::find(cell_vertices.begin(), cell_vertices.end(), v);
    assert(it != cell_vertices.end());
    const int local_pos = std::distance(cell_vertices.begin(), it);

    auto dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices[j * vertices.size() + i] = x_nodes[3 * dofs[local_pos] + j];
    vertex_to_pos[v] = i;
  }

  return {std::move(entities), std::move(x_vertices), std::move(vertex_to_pos)};
}
} // namespace

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
                            std::span<const std::int32_t> entities, int dim)
{
  if (entities.empty())
    return std::vector<double>();
  if (dim == 0)
    return std::vector<double>(entities.size(), 0);

  // Get the geometry dofs for the vertices of each entity
  const std::vector<std::int32_t> vertex_xdofs
      = entities_to_geometry(mesh, dim, entities, false);
  assert(!entities.empty());
  const std::size_t num_vertices = vertex_xdofs.size() / entities.size();

  // Get the  geometry coordinate
  std::span<const double> x = mesh.geometry().x();

  // Function to compute the length of (p0 - p1)
  auto delta_norm = [](const auto& p0, const auto& p1)
  {
    double norm = 0;
    for (std::size_t i = 0; i < 3; ++i)
      norm += (p0[i] - p1[i]) * (p0[i] - p1[i]);
    return std::sqrt(norm);
  };

  // Compute greatest distance between any to vertices
  assert(dim > 0);
  std::vector<double> h(entities.size(), 0);
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    // Get geometry 'dof' for each vertex of entity e
    std::span<const std::int32_t> e_vertices(
        vertex_xdofs.data() + e * num_vertices, num_vertices);

    // Compute maximum distance between any two vertices
    for (std::size_t i = 0; i < e_vertices.size(); ++i)
    {
      std::span<const double, 3> p0(x.data() + 3 * e_vertices[i], 3);
      for (std::size_t j = i + 1; j < e_vertices.size(); ++j)
      {
        std::span<const double, 3> p1(x.data() + 3 * e_vertices[j], 3);
        h[e] = std::max(h[e], delta_norm(p0, p1));
      }
    }
  }

  return h;
}
//-----------------------------------------------------------------------------
std::vector<double> mesh::cell_normals(const mesh::Mesh& mesh, int dim,
                                       std::span<const std::int32_t> entities)
{
  if (entities.empty())
    return std::vector<double>();

  if (mesh.topology().cell_type() == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(mesh.topology().cell_type(), dim, 0);

  // Find geometry nodes for topology entities
  std::span<const double> x = mesh.geometry().x();

  // Orient cells if they are tetrahedron
  bool orient = false;
  if (mesh.topology().cell_type() == CellType::tetrahedron)
    orient = true;

  std::vector<std::int32_t> geometry_entities
      = entities_to_geometry(mesh, dim, entities, orient);

  const std::size_t shape1 = geometry_entities.size() / entities.size();
  std::vector<double> n(entities.size() * 3);
  switch (type)
  {
  case CellType::interval:
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the two vertices as points
      std::array vertices{geometry_entities[i * shape1],
                          geometry_entities[i * shape1 + 1]};
      std::array p
          = {std::span<const double, 3>(x.data() + 3 * vertices[0], 3),
             std::span<const double, 3>(x.data() + 3 * vertices[1], 3)};

      // Define normal by rotating tangent counter-clockwise
      std::array<double, 3> t;
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), t.begin(),
                     [](auto x, auto y) { return x - y; });

      double norm = std::sqrt(t[0] * t[0] + t[1] * t[1]);
      std::span<double, 3> ni(n.data() + 3 * i, 3);
      ni[0] = -t[1] / norm;
      ni[1] = t[0] / norm;
      ni[2] = 0.0;
    }
    return n;
  }
  case CellType::triangle:
  {
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the three vertices as points
      std::array vertices = {geometry_entities[i * shape1 + 0],
                             geometry_entities[i * shape1 + 1],
                             geometry_entities[i * shape1 + 2]};
      std::array p
          = {std::span<const double, 3>(x.data() + 3 * vertices[0], 3),
             std::span<const double, 3>(x.data() + 3 * vertices[1], 3),
             std::span<const double, 3>(x.data() + 3 * vertices[2], 3)};

      // Compute (p1 - p0) and (p2 - p0)
      std::array<double, 3> dp1, dp2;
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), dp1.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p[2].begin(), p[2].end(), p[0].begin(), dp2.begin(),
                     [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<double, 3> ni = math::cross(dp1, dp2);
      double norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::transform(ni.begin(), ni.end(), std::next(n.begin(), 3 * i),
                     [norm](auto x) { return x / norm; });
    }
    return n;
  }
  case CellType::quadrilateral:
  {
    // TODO: check
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
      // Get the three vertices as points
      std::array vertices = {geometry_entities[i * shape1 + 0],
                             geometry_entities[i * shape1 + 1],
                             geometry_entities[i * shape1 + 2]};
      std::array p
          = {std::span<const double, 3>(x.data() + 3 * vertices[0], 3),
             std::span<const double, 3>(x.data() + 3 * vertices[1], 3),
             std::span<const double, 3>(x.data() + 3 * vertices[2], 3)};

      // Compute (p1 - p0) and (p2 - p0)
      std::array<double, 3> dp1, dp2;
      std::transform(p[1].begin(), p[1].end(), p[0].begin(), dp1.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p[2].begin(), p[2].end(), p[0].begin(), dp2.begin(),
                     [](auto x, auto y) { return x - y; });

      // Define cell normal via cross product of first two edges
      std::array<double, 3> ni = math::cross(dp1, dp2);
      double norm = std::sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
      std::transform(ni.begin(), ni.end(), std::next(n.begin(), 3 * i),
                     [norm](auto x) { return x / norm; });
    }
    return n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
}
//-----------------------------------------------------------------------------
std::vector<double>
mesh::compute_midpoints(const Mesh& mesh, int dim,
                        std::span<const std::int32_t> entities)
{
  if (entities.empty())
    return std::vector<double>();

  std::span<const double> x = mesh.geometry().x();

  // Build map from entity -> geometry dof
  // FIXME: This assumes a linear geometry.
  const std::vector<std::int32_t> e_to_g
      = entities_to_geometry(mesh, dim, entities, false);
  std::size_t shape1 = e_to_g.size() / entities.size();

  std::vector<double> x_mid(entities.size() * 3, 0);
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    std::span<double, 3> p(x_mid.data() + 3 * e, 3);
    std::span<const std::int32_t> rows(e_to_g.data() + e * shape1, shape1);
    for (auto row : rows)
    {
      std::span<const double, 3> xg(x.data() + 3 * row, 3);
      std::transform(p.begin(), p.end(), xg.begin(), p.begin(),
                     [size = rows.size()](auto x, auto y)
                     { return x + y / size; });
    }
  }

  return x_mid;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities(
    const Mesh& mesh, int dim,
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<
                std::size_t, 3, std::experimental::dynamic_extent>>)>& marker)
{
  // Run marker function on vertex coordinates
  const auto [xdata, xshape] = compute_vertex_coords(mesh);
  cmdspan3x_t x(xdata.data(), xshape);
  const std::vector<std::int8_t> marked = marker(x);
  if (marked.size() != x.extent(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable().create_connectivity(dim, 0);

  // Iterate over entities of dimension 'dim' to build vector of marked
  // entities
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
    const std::function<std::vector<std::int8_t>(
        std::experimental::mdspan<
            const double,
            std::experimental::extents<
                std::size_t, 3, std::experimental::dynamic_extent>>)>& marker)
{
  const Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute list of boundary facets
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  const std::vector<std::int32_t> boundary_facets
      = exterior_facet_indices(topology);

  // Run marker function on the vertex coordinates
  const auto [facet_entities, xdata, vertex_to_pos]
      = compute_vertex_coords_boundary(mesh, dim, boundary_facets);
  cmdspan3x_t x(xdata.data(), 3, xdata.size() / 3);
  const std::vector<std::int8_t> marked = marker(x);
  if (marked.size() != x.extent(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Loop over entities and check vertex markers
  mesh.topology_mutable().create_entities(dim);
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (auto e : facet_entities)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
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
std::vector<std::int32_t>
mesh::entities_to_geometry(const Mesh& mesh, int dim,
                           std::span<const std::int32_t> entities, bool orient)
{
  CellType cell_type = mesh.topology().cell_type();
  if (cell_type == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cells");
  if (orient and (cell_type != CellType::tetrahedron or dim != 2))
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");

  const Geometry& geometry = mesh.geometry();
  auto x = geometry.x();

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

  const std::size_t num_vertices
      = num_cell_vertices(cell_entity_type(cell_type, dim, 0));
  std::vector<std::int32_t> geometry_idx(entities.size() * num_vertices);
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    const std::int32_t idx = entities[i];
    // Always pick the second cell to be consistent with the e_to_v connectivity
    const std::int32_t cell = e_to_c->links(idx).back();
    auto ev = e_to_v->links(idx);
    assert(ev.size() == num_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (std::size_t j = 0; j < num_vertices; ++j)
    {
      int k = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), ev[j]));
      assert(k < (int)cv.size());
      geometry_idx[i * num_vertices + j] = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      std::array<double, 3> midpoint = {0, 0, 0};
      for (std::int32_t j : xc)
        for (int k = 0; k < 3; ++k)
          midpoint[k] += x[3 * j + k];
      std::transform(midpoint.begin(), midpoint.end(), midpoint.begin(),
                     [size = xc.size()](auto x) { return x / size; });

      // Compute vector triple product of two edges and vector to midpoint
      std::array<double, 3> p0, p1, p2;
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 0]),
                  3, p0.begin());
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 1]),
                  3, p1.begin());
      std::copy_n(std::next(x.begin(), 3 * geometry_idx[i * num_vertices + 2]),
                  3, p2.begin());

      std::array<double, 9> a;
      std::transform(midpoint.begin(), midpoint.end(), p0.begin(), a.begin(),
                     [](auto x, auto y) { return x - y; });
      std::transform(p1.begin(), p1.end(), p0.begin(), std::next(a.begin(), 3),
                     [](auto x, auto y) { return x - y; });
      std::transform(p2.begin(), p2.end(), p0.begin(), std::next(a.begin(), 6),
                     [](auto x, auto y) { return x - y; });

      // Midpoint direction should be opposite to normal, hence this
      // should be negative. Switch points if not.
      if (math::det(a.data(), {3, 3}) > 0.0)
      {
        std::swap(geometry_idx[i * num_vertices + 1],
                  geometry_idx[i * num_vertices + 2]);
      }
    }
  }

  return geometry_idx;
}
//------------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology)
{
  const int tdim = topology.dim();
  auto facet_map = topology.index_map(tdim - 1);
  if (!facet_map)
    throw std::runtime_error("Facets have not been computed.");

  // Find all owned facets (not ghost) with only one attached cell
  const int num_facets = facet_map->size_local();
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(f_to_c);
  std::vector<std::int32_t> facets;
  for (std::int32_t f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1)
      facets.push_back(f);
  }

  // Remove facets on internal inter-process boundary
  const std::vector<std::int32_t>& interprocess_facets
      = topology.interprocess_facets();
  std::vector<std::int32_t> ext_facets;
  std::set_difference(facets.begin(), facets.end(), interprocess_facets.begin(),
                      interprocess_facets.end(),
                      std::back_inserter(ext_facets));
  return ext_facets;
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction
mesh::create_cell_partitioner(mesh::GhostMode ghost_mode,
                              const graph::partition_fn& partfn)
{
  return [partfn, ghost_mode](MPI_Comm comm, int nparts, int tdim,
                              const graph::AdjacencyList<std::int64_t>& cells)
             -> graph::AdjacencyList<std::int32_t>
  {
    LOG(INFO) << "Compute partition of cells across ranks";

    // Compute distributed dual graph (for the cells on this process)
    const graph::AdjacencyList<std::int64_t> dual_graph
        = build_dual_graph(comm, cells, tdim);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, ghosting);
  };
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::compute_incident_entities(
    const Mesh& mesh, std::span<const std::int32_t> entities, int d0, int d1)
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
