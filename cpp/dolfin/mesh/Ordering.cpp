// Copyright (C) 2007-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Ordering.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include <vector>

#include <iostream>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
bool increasing(int num_vertices, const std::int32_t* local_vertices,
                const std::vector<std::int64_t>& global_vertex_indices)
{
  // return std::is_sorted(local_vertices, local_vertices + num_vertices,
  //                       [&map = global_vertex_indices](auto& a, auto& b) {
  //                         return map[a] < map[b];
  //                       });

  for (int v = 1; v < num_vertices; v++)
  {
    if (global_vertex_indices[local_vertices[v - 1]]
        >= global_vertex_indices[local_vertices[v]])
    {
      return false;
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
bool increasing(int n0, const std::int32_t* v0, int n1, const std::int32_t* v1,
                int num_vertices, const std::int32_t* local_vertices,
                const std::vector<std::int64_t>& global_vertex_indices)
{
  assert(n0 == n1);
  assert(num_vertices > n0);
  const int num_non_incident = num_vertices - n0;

  // Compute non-incident vertices for first entity
  std::vector<int> w0(num_non_incident);
  int k = 0;
  for (int i = 0; i < num_vertices; i++)
  {
    const int v = local_vertices[i];
    bool incident = false;
    for (int j = 0; j < n0; j++)
    {
      if (v0[j] == v)
      {
        incident = true;
        break;
      }
    }
    if (!incident)
      w0[k++] = v;
  }
  assert(k == num_non_incident);

  // Compute non-incident vertices for second entity
  std::vector<int> w1(num_non_incident);
  k = 0;
  for (int i = 0; i < num_vertices; i++)
  {
    const int v = local_vertices[i];
    bool incident = false;
    for (int j = 0; j < n1; j++)
    {
      if (v1[j] == v)
      {
        incident = true;
        break;
      }
    }

    if (!incident)
      w1[k++] = v;
  }
  assert(k == num_non_incident);

  // Compare lexicographic ordering of w0 and w1
  for (int i = 0; i < num_non_incident; i++)
  {
    if (global_vertex_indices[w0[i]] < global_vertex_indices[w1[i]])
      return true;
    else if (global_vertex_indices[w0[i]] > global_vertex_indices[w1[i]])
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void order_cell_simplex(const std::vector<std::int64_t>& global_vertex_indices,
                        dolfin::mesh::Cell& cell)
{
  const mesh::MeshTopology& topology = cell.mesh().topology();
  const int tdim = topology.dim();

  if (tdim < 1)
    return;

  const mesh::CellType& cell_type = cell.mesh().type();
  const int num_edges = cell_type.num_entities(1);

  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  std::shared_ptr<const mesh::MeshConnectivity> connect_1_0
      = topology.connectivity(1, 0);
  if (connect_1_0 and !connect_1_0->empty())
  {
    // std::cout << "Sorting vertices on edges: " << num_edges << std::endl;
    // assert(!topology(tdim, 1).empty());

    // Sort vertices on each edge
    const std::int32_t* cell_edges = cell.entities(1);
    for (int i = 0; i < num_edges; ++i)
    {
      std::int32_t* edge_vertices
          = const_cast<std::int32_t*>((*connect_1_0)(cell_edges[i]));
      // sort_entities(2, edge_vertices, global_vertex_indices);
      std::sort(edge_vertices, edge_vertices + 2, [&](auto& a, auto& b) {
        return global_vertex_indices[a] < global_vertex_indices[b];
      });
    }
  }

  if (tdim < 2)
    return;

  const int num_faces = cell_type.num_entities(2);

  // Sort local vertices on faces in ascending order, connectivity 2 - 0
  std::shared_ptr<const mesh::MeshConnectivity> connect_2_0
      = topology.connectivity(2, 0);
  if (connect_2_0 and !connect_2_0->empty())
  {
    // std::cout << "Sorting vertices on faces: " << num_faces << std::endl;
    // assert(!topology(3, 2).empty());

    // Sort vertices on each facet
    const std::int32_t* cell_faces = cell.entities(2);
    for (int i = 0; i < num_faces; ++i)
    {
      std::int32_t* face_vertices
          = const_cast<std::int32_t*>((*connect_2_0)(cell_faces[i]));
      std::sort(face_vertices, face_vertices + 3, [&](auto& a, auto& b) {
        return global_vertex_indices[a] < global_vertex_indices[b];
      });
    }
  }

  // Sort local edges on local faces after non-incident vertex,
  // connectivity 2 - 1
  std::shared_ptr<const mesh::MeshConnectivity> connect_2_1
      = topology.connectivity(2, 1);
  if (connect_2_1 and !connect_2_1->empty())
  {
    // dolfin_assert(!topology(3, 2).empty());
    // dolfin_assert(!topology(2, 0).empty());
    // dolfin_assert(!topology(1, 0).empty());

    // Get face numbers
    const std::int32_t* cell_faces = cell.entities(2);

    // Loop over faces on cell
    for (int i = 0; i < num_faces; ++i)
    {
      // For each face number get the global vertex numbers
      const std::int32_t* face_vertices = (*connect_2_0)(cell_faces[i]);

      // For each facet number get the global edge number
      std::int32_t* cell_edges
          = const_cast<std::int32_t*>((*connect_2_1)(cell_faces[i]));

      // Loop over vertices on face
      std::size_t m = 0;
      for (int j = 0; j < 3; ++j)
      {
        // Loop edges on face
        for (int k = m; k < 3; ++k)
        {
          // For each edge number get the global vertex numbers
          const std::int32_t* edge_vertices = (*connect_1_0)(cell_edges[k]);

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

  if (tdim < 3)
    return;

  // Sort local vertices on cell in ascending order, connectivity 3 - 0
  std::shared_ptr<const mesh::MeshConnectivity> connect_3_0
      = topology.connectivity(3, 0);
  if (connect_3_0 and !connect_3_0->empty())
  {
    std::int32_t* cell_vertices = const_cast<std::int32_t*>(cell.entities(0));
    std::sort(cell_vertices, cell_vertices + 4, [&](auto& a, auto& b) {
      return global_vertex_indices[a] < global_vertex_indices[b];
    });
  }

  // Sort local edges on cell after non-incident vertex tuble,
  // connectivity 3-1
  std::shared_ptr<const mesh::MeshConnectivity> connect_3_1
      = topology.connectivity(3, 1);
  if (connect_3_1 and !connect_3_1->empty())
  {
    // assert(!topology(1, 0).empty());

    // Get cell vertices and edge numbers
    const std::int32_t* cell_vertices = cell.entities(0);
    std::int32_t* cell_edges = const_cast<std::int32_t*>(cell.entities(1));

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
          const std::int32_t* edge_vertices = (*connect_1_0)(cell_edges[k]);

          // Check if the ith and jth vertex of the cell are
          // non-incident on edge k
          if (!std::count(edge_vertices, edge_vertices + 2, cell_vertices[i])
              and !std::count(edge_vertices, edge_vertices + 2,
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

  // Sort local facets on cell after non-incident vertex, connectivity 3
  // - 2
  std::shared_ptr<const mesh::MeshConnectivity> connect_3_2
      = topology.connectivity(3, 2);
  if (connect_3_2 and !connect_3_2->empty())
  {
    // std::cout << "Sorting final: " << num_edges << std::endl;

    // assert(!topology(2, 0).empty());

    // Get cell vertices and facet numbers
    const std::int32_t* cell_vertices = cell.entities(0);
    std::int32_t* cell_faces = const_cast<std::int32_t*>(cell.entities(2));

    // Loop vertices on cell
    for (int i = 0; i < 4; ++i)
    {
      // Loop facets on cell
      for (int j = i; j < 4; ++j)
      {
        std::int32_t* face_vertices
            = const_cast<std::int32_t*>((*connect_2_0)(cell_faces[j]));

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
}
//-----------------------------------------------------------------------------
bool ordered_cell_simplex(
    const std::vector<std::int64_t>& global_vertex_indices,
    const dolfin::mesh::Cell& cell)
{
  // Get mesh topology
  const mesh::MeshTopology& topology = cell.mesh().topology();
  const int tdim = topology.dim();
  const int c = cell.index();

  // Get vertices
  std::shared_ptr<const mesh::MeshConnectivity> connect_tdim_0
      = topology.connectivity(tdim, 0);
  assert(connect_tdim_0);

  const int num_vertices = connect_tdim_0->size(c);
  const std::int32_t* vertices = (*connect_tdim_0)(c);
  assert(vertices);

  // Check that vertices are in ascending order
  if (!increasing(num_vertices, vertices, global_vertex_indices))
    return false;

  // Note the comparison below: d + 1 < dim, not d < dim - 1
  // Otherwise, d < dim - 1 will evaluate to true for dim = 0 with std::size_t

  // Check numbering of entities of positive dimension and codimension
  for (int d = 1; d + 1 < tdim; ++d)
  {
    // Check if entities exist, otherwise skip
    std::shared_ptr<const mesh::MeshConnectivity> connect_d_0
        = topology.connectivity(d, 0);
    if (!connect_d_0 or connect_d_0->empty())
      continue;

    // Get entities
    std::shared_ptr<const mesh::MeshConnectivity> connect_tdim_d
        = topology.connectivity(tdim, d);
    assert(connect_tdim_d);
    const int num_entities = connect_tdim_d->size(c);
    const std::int32_t* entities = (*connect_tdim_d)(c);

    // Iterate over entities
    for (int e = 1; e < num_entities; ++e)
    {
      // Get vertices for first entity
      const int e0 = entities[e - 1];
      const int n0 = connect_d_0->size(e0);
      const std::int32_t* v0 = (*connect_d_0)(e0);

      // Get vertices for second entity
      const int e1 = entities[e];
      const int n1 = connect_d_0->size(e1);
      const std::int32_t* v1 = (*connect_d_0)(e1);

      // Check ordering of entities
      if (!increasing(n0, v0, n1, v1, num_vertices, vertices,
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

  // Get global vertex numbering
  if (!mesh.topology().have_global_indices(0))
    throw std::runtime_error("Mesh does not have global vertex indices.");
  const std::vector<std::int64_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  // Iterate over all cells
  for (mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
    order_cell_simplex(global_vertex_indices, cell);
}
//-----------------------------------------------------------------------------
bool mesh::Ordering::is_ordered_simplex(const mesh::Mesh& mesh)
{
  const mesh::CellType& cell_type = mesh.type();
  if (!cell_type.is_simplex())
    throw std::runtime_error(
        "Mesh ordering check is for simplex cell types only.");

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
    if (ordered_cell_simplex(global_vertex_indices, cell))
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
