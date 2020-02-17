// Copyright (C) 2007-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Ordering.h"
#include "CoordinateDofs.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "cell_types.h"
#include <array>
#include <vector>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
bool increasing(const int n, const std::int32_t* v0, const std::int32_t* v1,
                int num_vertices, const std::int32_t* vertices,
                const std::vector<std::int64_t>& global_indices)
{
  assert(num_vertices > n);
  assert(num_vertices <= 4);
  const int num_non_incident = num_vertices - n;
  assert(num_non_incident <= 3);

  // Array of global vertices
  std::array<std::int64_t, 4> v = {-1};
  for (int i = 0; i < num_vertices; ++i)
    v[i] = global_indices[vertices[i]];
  assert(std::is_sorted(v.begin(), v.begin() + num_vertices));

  // Compute non-incident vertices for first entity
  std::array<std::int64_t, 3> _v0 = {-1};
  for (int i = 0; i < n; ++i)
    _v0[i] = global_indices[v0[i]];
  std::sort(_v0.begin(), _v0.begin() + n);
  std::array<std::int64_t, 3> w0 = {-1};
  auto it = std::set_difference(v.begin(), v.begin() + num_vertices,
                                _v0.begin(), _v0.begin() + n, w0.begin());
  assert(std::distance(w0.begin(), it) == num_non_incident);

  std::array<std::int64_t, 3> _v1 = {-1};
  for (int i = 0; i < n; ++i)
    _v1[i] = global_indices[v1[i]];
  std::sort(_v1.begin(), _v1.begin() + n);
  std::array<std::int64_t, 3> w1 = {-1};
  auto it1 = std::set_difference(v.begin(), v.begin() + num_vertices,
                                 _v1.begin(), _v1.begin() + n, w1.begin());
  assert(std::distance(w1.begin(), it1) == num_non_incident);

  return w0 < w1;
}
//-----------------------------------------------------------------------------
void sort_1_0(graph::AdjacencyList<std::int32_t>& connect_1_0,
              const mesh::MeshEntity& cell,
              const std::vector<std::int64_t>& global_vertex_indices,
              const int num_edges)
{
  // Sort vertices on each edge
  const std::int32_t* cell_edges = cell.entities_ptr(1);
  assert(cell_edges);
  for (int i = 0; i < num_edges; ++i)
  {
    auto edge_vertices = connect_1_0.links(cell_edges[i]);
    std::sort(edge_vertices.data(), edge_vertices.data() + 2,
              [&](auto& a, auto& b) {
                return global_vertex_indices[a] < global_vertex_indices[b];
              });
  }
}
//-----------------------------------------------------------------------------
void sort_2_0(graph::AdjacencyList<std::int32_t>& connect_2_0,
              const mesh::MeshEntity& cell,
              const std::vector<std::int64_t>& global_vertex_indices,
              const int num_faces)
{
  // Sort vertices on each facet
  const std::int32_t* cell_faces = cell.entities_ptr(2);
  assert(cell_faces);
  for (int i = 0; i < num_faces; ++i)
  {
    auto face_vertices = connect_2_0.links(cell_faces[i]);
    std::sort(face_vertices.data(), face_vertices.data() + 3,
              [&](auto& a, auto& b) {
                return global_vertex_indices[a] < global_vertex_indices[b];
              });
  }
}
//-----------------------------------------------------------------------------
void sort_2_1(graph::AdjacencyList<std::int32_t>& connect_2_1,
              const graph::AdjacencyList<std::int32_t>& connect_2_0,
              const graph::AdjacencyList<std::int32_t>& connect_1_0,
              const mesh::MeshEntity& cell, const int num_faces)
{
  // Loop over faces on cell
  const std::int32_t* cell_faces = cell.entities_ptr(2);
  assert(cell_faces);
  for (int i = 0; i < num_faces; ++i)
  {
    // For each face number get the global vertex numbers
    auto face_vertices = connect_2_0.links(cell_faces[i]);

    // For each facet number get the global edge number
    auto cell_edges = connect_2_1.links(cell_faces[i]);

    // Loop over vertices on face
    std::size_t m = 0;
    for (int j = 0; j < 3; ++j)
    {
      // Loop edges on face
      for (int k = m; k < 3; ++k)
      {
        // For each edge number get the global vertex numbers
        auto edge_vertices = connect_1_0.links(cell_edges[k]);

        // Check if the jth vertex of facet i is non-incident on edge k
        if (!std::count(edge_vertices.data(), edge_vertices.data() + 2,
                        face_vertices[j]))
        {
          // Swap face numbers
          std::swap(cell_edges[m], cell_edges[k]);
          m++;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void sort_3_0(graph::AdjacencyList<std::int32_t>& connect_3_0,
              const mesh::MeshEntity& cell,
              const std::vector<std::int64_t>& global_vertex_indices)
{
  auto cell_vertices = connect_3_0.links(cell.index());
  std::sort(cell_vertices.data(), cell_vertices.data() + 4,
            [&](auto& a, auto& b) {
              return global_vertex_indices[a] < global_vertex_indices[b];
            });
}
//-----------------------------------------------------------------------------
void sort_3_1(graph::AdjacencyList<std::int32_t>& connect_3_1,
              const graph::AdjacencyList<std::int32_t>& connect_1_0,
              const mesh::MeshEntity& cell)
{
  // Get cell vertices and edge numbers
  const std::int32_t* cell_vertices = cell.entities_ptr(0);
  assert(cell_vertices);
  auto cell_edges = connect_3_1.links(cell.index());

  // Loop two vertices on cell as a lexicographical tuple
  // (i, j): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
  int m = 0;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = i + 1; j < 4; ++j)
    {
      // Loop edge numbers
      for (int k = m; k < 6; ++k)
      {
        // Get local vertices on edge
        auto edge_vertices = connect_1_0.links(cell_edges[k]);

        // Check if the ith and jth vertex of the cell are
        // non-incident on edge k
        if (!std::count(edge_vertices.data(), edge_vertices.data() + 2,
                        cell_vertices[i])
            and !std::count(edge_vertices.data(), edge_vertices.data() + 2,
                            cell_vertices[j]))
        {
          // Swap edge numbers
          std::swap(cell_edges[m], cell_edges[k]);
          m++;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void sort_3_2(graph::AdjacencyList<std::int32_t>& connect_3_2,
              const graph::AdjacencyList<std::int32_t>& connect_2_0,
              const mesh::MeshEntity& cell)
{
  // Get cell vertices and facet numbers
  const std::int32_t* cell_vertices = cell.entities_ptr(0);
  assert(cell_vertices);
  auto cell_faces = connect_3_2.links(cell.index());

  // Loop vertices on cell
  for (int i = 0; i < 4; ++i)
  {
    // Loop facets on cell
    for (int j = i; j < 4; ++j)
    {
      auto face_vertices = connect_2_0.links(cell_faces[j]);

      // Check if the ith vertex of the cell is non-incident on facet j
      if (!std::count(face_vertices.data(), face_vertices.data() + 3,
                      cell_vertices[i]))
      {
        // Swap facet numbers
        std::swap(cell_faces[i], cell_faces[j]);
        break;
      }
    }
  }
}
//-----------------------------------------------------------------------------
bool ordered_cell_simplex(
    const std::vector<std::int64_t>& global_vertex_indices,
    const mesh::MeshEntity& cell)
{
  // Get mesh topology
  const mesh::Topology& topology = cell.mesh().topology();
  const int tdim = topology.dim();
  const int c = cell.index();

  // Get vertices
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connect_tdim_0
      = topology.connectivity(tdim, 0);
  assert(connect_tdim_0);

  const int num_vertices = connect_tdim_0->num_links(c);
  auto vertices = connect_tdim_0->links(c);

  // Check that vertices are in ascending order
  if (!std::is_sorted(vertices.data(), vertices.data() + num_vertices,
                      [&](auto& a, auto& b) {
                        return global_vertex_indices[a]
                               < global_vertex_indices[b];
                      }))
  {
    return false;
  }
  // Note the comparison below: d + 1 < dim, not d < dim - 1
  // Otherwise, d < dim - 1 will evaluate to true for dim = 0 with std::size_t

  // Check numbering of entities of positive dimension and codimension
  for (int d = 1; d + 1 < tdim; ++d)
  {
    // Check if entities exist, otherwise skip
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connect_d_0
        = topology.connectivity(d, 0);
    if (!connect_d_0)
      continue;

    // Get entities
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connect_tdim_d
        = topology.connectivity(tdim, d);
    assert(connect_tdim_d);
    const int num_entities = connect_tdim_d->num_links(c);
    auto entities = connect_tdim_d->links(c);

    // Iterate over entities
    for (int e = 1; e < num_entities; ++e)
    {
      // Get vertices for first entity
      const int e0 = entities[e - 1];
      const int n0 = connect_d_0->num_links(e0);
      auto v0 = connect_d_0->links(e0);

      // Get vertices for second entity
      const int e1 = entities[e];
      const int n1 = connect_d_0->num_links(e1);
      auto v1 = connect_d_0->links(e1);

      // Check ordering of entities
      assert(n0 == n1);
      if (!increasing(n0, v0.data(), v1.data(), num_vertices, vertices.data(),
                      global_vertex_indices))
      {
        return false;
      }
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
void mesh::Ordering::order_simplex(mesh::Mesh& mesh)
{
  if (!mesh::is_simplex(mesh.topology().cell_type()))
    throw std::runtime_error("Mesh ordering is for simplex cell types only.");

  if (mesh.degree() > 1)
  {
    throw std::runtime_error(
        "Mesh re-ordering not yet working for high-order meshes");
  }

  mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (mesh.num_entities(tdim) == 0)
    return;

  // Skip ordering for dimension 0
  if (tdim == 0)
    return;

  graph::AdjacencyList<std::int32_t>& connect_g
      = mesh.coordinate_dofs().entity_points();

  // Get global vertex numbering
  auto map = mesh.topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_vertex_indices
      = map->global_indices(false);

  const int num_edges = mesh::cell_num_entities(mesh.topology().cell_type(), 1);
  const int num_faces
      = (tdim > 1) ? mesh::cell_num_entities(mesh.topology().cell_type(), 2)
                   : -1;

  std::shared_ptr<graph::AdjacencyList<std::int32_t>> connect_1_0, connect_2_0,
      connect_2_1, connect_3_0, connect_3_1, connect_3_2;
  connect_1_0 = topology.connectivity(1, 0);
  if (tdim > 1)
  {
    connect_2_0 = topology.connectivity(2, 0);
    connect_2_1 = topology.connectivity(2, 1);
  }
  if (tdim > 2)
  {
    connect_3_0 = topology.connectivity(3, 0);
    connect_3_1 = topology.connectivity(3, 1);
    connect_3_2 = topology.connectivity(3, 2);
  }

  // Iterate over all cells
  for (mesh::MeshEntity& cell : mesh::MeshRange(mesh, tdim))
  {
    // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

    // Order 'coordinate' connectivity
    if (tdim == 1)
      sort_1_0(connect_g, cell, global_vertex_indices, num_edges);
    else if (tdim == 2)
      sort_2_0(connect_g, cell, global_vertex_indices, num_faces);
    else if (tdim == 3)
      sort_3_0(connect_g, cell, global_vertex_indices);

    // Sort local vertices on edges in ascending order, connectivity 1-0
    if (connect_1_0)
      sort_1_0(*connect_1_0, cell, global_vertex_indices, num_edges);

    // Sort local vertices on faces in ascending order, connectivity 2-0
    if (connect_2_0)
      sort_2_0(*connect_2_0, cell, global_vertex_indices, num_faces);

    // Sort local edges on local faces after non-incident vertex,
    // connectivity 2-1
    if (connect_2_1)
      sort_2_1(*connect_2_1, *connect_2_0, *connect_1_0, cell, num_faces);

    // Sort local vertices on cell in ascending order, connectivity 3-0
    if (connect_3_0)
      sort_3_0(*connect_3_0, cell, global_vertex_indices);

    // Sort local edges on cell after non-incident vertex tuple,
    // connectivity 3-1
    if (connect_3_1)
      sort_3_1(*connect_3_1, *connect_1_0, cell);

    // Sort local facets on cell after non-incident vertex, connectivity
    // 3-2
    if (connect_3_2)
      sort_3_2(*connect_3_2, *connect_2_0, cell);
  }
}
//-----------------------------------------------------------------------------
bool mesh::Ordering::is_ordered_simplex(const mesh::Mesh& mesh)
{
  if (!mesh::is_simplex(mesh.topology().cell_type()))
  {
    throw std::runtime_error(
        "Mesh ordering check is for simplex cell types only.");
  }

  const int tdim = mesh.topology().dim();
  if (mesh.num_entities(tdim) == 0)
    return true;

  // Get global vertex numbering
  auto map = mesh.topology().index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_vertex_indices
      = map->global_indices(false);

  // Check if all cells are ordered
  for (const mesh::MeshEntity& cell : mesh::MeshRange(mesh, tdim))
  {
    if (!ordered_cell_simplex(global_vertex_indices, cell))
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
