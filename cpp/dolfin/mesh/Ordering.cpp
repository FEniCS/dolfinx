// Copyright (C) 2007-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Ordering.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include <array>
#include <vector>

using namespace dolfin;

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
void sort_1_0(mesh::Connectivity& connect_1_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices,
              const int num_edges)
{
  // Sort vertices on each edge
  const std::int32_t* cell_edges = cell.entities(1);
  assert(cell_edges);
  for (int i = 0; i < num_edges; ++i)
  {
    std::int32_t* edge_vertices = connect_1_0.connections(cell_edges[i]);
    assert(edge_vertices);
    std::sort(edge_vertices, edge_vertices + 2, [&](auto& a, auto& b) {
      return global_vertex_indices[a] < global_vertex_indices[b];
    });
  }
}
//-----------------------------------------------------------------------------
void sort_2_0(mesh::Connectivity& connect_2_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices,
              const int num_faces)
{
  // Sort vertices on each facet
  const std::int32_t* cell_faces = cell.entities(2);
  assert(cell_faces);
  for (int i = 0; i < num_faces; ++i)
  {
    std::int32_t* face_vertices = connect_2_0.connections(cell_faces[i]);
    assert(face_vertices);
    std::sort(face_vertices, face_vertices + 3, [&](auto& a, auto& b) {
      return global_vertex_indices[a] < global_vertex_indices[b];
    });
  }
}
//-----------------------------------------------------------------------------
void sort_2_1(mesh::Connectivity& connect_2_1,
              const mesh::Connectivity& connect_1_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices,
              const int num_faces)
{
  // Loop over faces on cell
  const std::int32_t* cell_faces = cell.entities(2);
  assert(cell_faces);
  for (int i = 0; i < num_faces; ++i)
  {
    // For each face number get the global vertex numbers
    const std::int32_t* face_vertices = connect_2_1.connections(cell_faces[i]);
    assert(face_vertices);

    // For each facet number get the global edge number
    std::int32_t* cell_edges = connect_2_1.connections(cell_faces[i]);
    assert(cell_edges);

    // Loop over vertices on face
    std::size_t m = 0;
    for (int j = 0; j < 3; ++j)
    {
      // Loop edges on face
      for (int k = m; k < 3; ++k)
      {
        // For each edge number get the global vertex numbers
        const std::int32_t* edge_vertices
            = connect_1_0.connections(cell_edges[k]);
        assert(edge_vertices);

        // Check if the jth vertex of facet i is non-incident on edge k
        if (!std::count(edge_vertices, edge_vertices + 2, face_vertices[j]))
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
void sort_3_0(mesh::Connectivity& connect_3_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices)
{
  std::int32_t* cell_vertices = connect_3_0.connections(cell.index());
  assert(cell_vertices);
  std::sort(cell_vertices, cell_vertices + 4, [&](auto& a, auto& b) {
    return global_vertex_indices[a] < global_vertex_indices[b];
  });
}
//-----------------------------------------------------------------------------
void sort_3_1(mesh::Connectivity& connect_3_1,
              const mesh::Connectivity& connect_1_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices)
{
  // Get cell vertices and edge numbers
  const std::int32_t* cell_vertices = cell.entities(0);
  assert(cell_vertices);
  std::int32_t* cell_edges = connect_3_1.connections(cell.index());
  assert(cell_edges);

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
        const std::int32_t* edge_vertices
            = connect_1_0.connections(cell_edges[k]);
        assert(edge_vertices);

        // Check if the ith and jth vertex of the cell are
        // non-incident on edge k
        if (!std::count(edge_vertices, edge_vertices + 2, cell_vertices[i])
            and !std::count(edge_vertices, edge_vertices + 2, cell_vertices[j]))
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
void sort_3_2(mesh::Connectivity& connect_3_2,
              const mesh::Connectivity& connect_2_0, const mesh::Cell& cell,
              const std::vector<std::int64_t>& global_vertex_indices)
{
  // Get cell vertices and facet numbers
  const std::int32_t* cell_vertices = cell.entities(0);
  assert(cell_vertices);
  std::int32_t* cell_faces = connect_3_2.connections(cell.index());
  assert(cell_faces);

  // Loop vertices on cell
  for (int i = 0; i < 4; ++i)
  {
    // Loop facets on cell
    for (int j = i; j < 4; ++j)
    {
      const std::int32_t* face_vertices
          = connect_2_0.connections(cell_faces[j]);
      assert(face_vertices);

      // Check if the ith vertex of the cell is non-incident on facet j
      if (!std::count(face_vertices, face_vertices + 3, cell_vertices[i]))
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
    const mesh::Cell& cell)
{
  // Get mesh topology
  const mesh::Topology& topology = cell.mesh().topology();
  const int tdim = topology.dim();
  const int c = cell.index();

  // Get vertices
  std::shared_ptr<const mesh::Connectivity> connect_tdim_0
      = topology.connectivity(tdim, 0);
  assert(connect_tdim_0);

  const int num_vertices = connect_tdim_0->size(c);
  const std::int32_t* vertices = connect_tdim_0->connections(c);
  assert(vertices);

  // Check that vertices are in ascending order
  if (!std::is_sorted(vertices, vertices + num_vertices, [&](auto& a, auto& b) {
        return global_vertex_indices[a] < global_vertex_indices[b];
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
    std::shared_ptr<const mesh::Connectivity> connect_d_0
        = topology.connectivity(d, 0);
    if (!connect_d_0)
      continue;

    // Get entities
    std::shared_ptr<const mesh::Connectivity> connect_tdim_d
        = topology.connectivity(tdim, d);
    assert(connect_tdim_d);
    const int num_entities = connect_tdim_d->size(c);
    const std::int32_t* entities = connect_tdim_d->connections(c);

    // Iterate over entities
    for (int e = 1; e < num_entities; ++e)
    {
      // Get vertices for first entity
      const int e0 = entities[e - 1];
      const int n0 = connect_d_0->size(e0);
      const std::int32_t* v0 = connect_d_0->connections(e0);

      // Get vertices for second entity
      const int e1 = entities[e];
      const int n1 = connect_d_0->size(e1);
      const std::int32_t* v1 = connect_d_0->connections(e1);

      // Check ordering of entities
      assert(n0 == n1);
      if (!increasing(n0, v0, v1, num_vertices, vertices,
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
  const mesh::CellType& cell_type = mesh.type();
  if (!cell_type.is_simplex())
    throw std::runtime_error("Mesh ordering is for simplex cell types only.");

  const int tdim = mesh.topology().dim();
  if (mesh.num_entities(tdim) == 0)
    return;

  // Skip ordering for dimension 0
  if (tdim == 0)
    return;

  if (mesh.degree() > 1)
  {
    throw std::runtime_error(
        "Mesh re-ordering not yet working for high-order meshes");
  }

  mesh::Topology& topology = mesh.topology();

  // Get global vertex numbering
  if (!mesh.topology().have_global_indices(0))
    throw std::runtime_error("Mesh does not have global vertex indices.");
  const std::vector<std::int64_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  mesh::Connectivity& connect_g = mesh.coordinate_dofs().entity_points();
  const int num_edges = cell_type.num_entities(1);

  // Get connectivities
  std::shared_ptr<mesh::Connectivity> connect_1_0, connect_2_0, connect_3_0,
      connect_2_1, connect_3_1, connect_3_2;
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
  for (mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

    // Order 'coordinate' connectivity
    if (tdim == 1)
      sort_1_0(connect_g, cell, global_vertex_indices, num_edges);
    else if (tdim == 2)
      sort_2_0(connect_g, cell, global_vertex_indices,
               cell_type.num_entities(2));
    else if (tdim == 3)
      sort_3_0(connect_g, cell, global_vertex_indices);

    // Sort local vertices on edges in ascending order, connectivity 1 - 0
    if (connect_1_0)
      sort_1_0(*connect_1_0, cell, global_vertex_indices, num_edges);

    if (tdim < 2)
      break;

    const int num_faces = cell_type.num_entities(2);

    // Sort local vertices on faces in ascending order, connectivity 2 - 0
    if (connect_2_0)
      sort_2_0(*connect_2_0, cell, global_vertex_indices, num_faces);

    // Sort local edges on local faces after non-incident vertex,
    // connectivity 2 - 1
    if (connect_2_1)
    {
      sort_2_1(*connect_2_1, *connect_1_0, cell, global_vertex_indices,
               num_faces);
    }

    if (tdim < 3)
      break;

    // Sort local vertices on cell in ascending order, connectivity 3 - 0
    if (connect_3_0)
      sort_3_0(*connect_3_0, cell, global_vertex_indices);

    // Sort local edges on cell after non-incident vertex tuble,
    // connectivity 3-1
    if (connect_3_1)
      sort_3_1(*connect_3_1, *connect_1_0, cell, global_vertex_indices);

    // Sort local facets on cell after non-incident vertex, connectivity 3
    // - 2
    if (connect_3_2)
      sort_3_2(*connect_3_2, *connect_2_0, cell, global_vertex_indices);
  }
}
//-----------------------------------------------------------------------------
bool mesh::Ordering::is_ordered_simplex(const mesh::Mesh& mesh)
{
  const mesh::CellType& cell_type = mesh.type();
  if (!cell_type.is_simplex())
  {
    throw std::runtime_error(
        "Mesh ordering check is for simplex cell types only.");
  }

  const int tdim = mesh.topology().dim();
  if (mesh.num_entities(tdim) == 0)
    return true;

  // Get global vertex numbering
  if (!mesh.topology().have_global_indices(0))
    throw std::runtime_error("Mesh does not have global vertex indices.");
  const std::vector<std::int64_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  // Check if all cells are ordered
  for (const mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    if (!ordered_cell_simplex(global_vertex_indices, cell))
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
