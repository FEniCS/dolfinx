// Copyright (C) 2007-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshOrdering.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include <vector>

using namespace dolfin;
using namespace dolfin::mesh;

namespace
{
//-----------------------------------------------------------------------------
void sort_entities(
    int num_vertices, std::int32_t* local_vertices,
    const std::vector<std::int64_t>& local_to_global_vertex_indices)
{
  // Two cases here, either sort vertices directly (when running in
  // serial) or sort based on the global indices (when running in
  // parallel)

  // Comparison operator for sorting based on global indices
  class GlobalSort
  {
  public:
    GlobalSort(const std::vector<std::int64_t>& local_to_global_vertex_indices)
        : g(local_to_global_vertex_indices)
    {
    }

    bool operator()(const std::size_t& l, const std::size_t& r)
    {
      return g[l] < g[r];
    }

    const std::vector<std::int64_t>& g;
  };

  // Sort on global vertex indices
  GlobalSort global_sort(local_to_global_vertex_indices);
  std::sort(local_vertices, local_vertices + num_vertices, global_sort);
}
//-----------------------------------------------------------------------------
void order_simplex(
    const std::vector<std::int64_t>& local_to_global_vertex_indices,
    dolfin::mesh::Cell& cell)
{
  const mesh::CellType& cell_type = cell.mesh().type();
  const mesh::MeshTopology& topology = cell.mesh().topology();

  const int num_edges = cell_type.num_entities(1);
  const int num_faces = cell_type.num_entities(2);

  // Sort i - j for i > j: 1 - 0, 2 - 0, 2 - 1, 3 - 0, 3 - 1, 3 - 2

  // const MeshTopology& topology = cell.mesh().topology();
  // const int tdim = topology.dim();

  // Sort local vertices on edges in ascending order, connectivity 1 - 0
  std::shared_ptr<const MeshConnectivity> connect_1_0
      = topology.connectivity(1, 0);
  if (connect_1_0)
  {
    // assert(!topology(tdim, 1).empty());

    // Sort vertices on each edge
    const std::int32_t* cell_edges = cell.entities(1);
    for (int i = 0; i < num_edges; ++i)
    {
      std::int32_t* edge_vertices
          = const_cast<std::int32_t*>((*connect_1_0)(cell_edges[i]));
      sort_entities(2, edge_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local vertices on faces in ascending order, connectivity 2 - 0
  std::shared_ptr<const MeshConnectivity> connect_2_0
      = topology.connectivity(2, 0);
  if (connect_2_0)
  {
    // assert(!topology(3, 2).empty());

    // Sort vertices on each facet
    const std::int32_t* cell_faces = cell.entities(2);
    for (int i = 0; i < num_faces; ++i)
    {
      std::int32_t* face_vertices
          = const_cast<std::int32_t*>((*connect_2_0)(cell_faces[i]));
      sort_entities(3, face_vertices, local_to_global_vertex_indices);
    }
  }

  // Sort local edges on local faces after non-incident vertex,
  // connectivity 2 - 1
  std::shared_ptr<const MeshConnectivity> connect_2_1
      = topology.connectivity(2, 1);
  if (connect_2_1)
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
            std::size_t tmp = cell_edges[m];
            cell_edges[m] = cell_edges[k];
            cell_edges[k] = tmp;
            m++;
            break;
          }
        }
      }
    }
  }

  // Sort local vertices on cell in ascending order, connectivity 3 - 0
  std::shared_ptr<const MeshConnectivity> connect_3_0
      = topology.connectivity(3, 0);
  if (connect_3_0)
  {
    std::int32_t* cell_vertices = const_cast<std::int32_t*>(cell.entities(0));
    sort_entities(4, cell_vertices, local_to_global_vertex_indices);
  }

  // Sort local edges on cell after non-incident vertex tuble,
  // connectivity 3-1
  std::shared_ptr<const MeshConnectivity> connect_3_1
      = topology.connectivity(3, 1);
  if (connect_3_1)
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
            std::int32_t tmp = cell_edges[m];
            cell_edges[m] = cell_edges[k];
            cell_edges[k] = tmp;
            m++;
            break;
          }
        }
      }
    }
  }

  // Sort local facets on cell after non-incident vertex, connectivity 3
  // - 2
  std::shared_ptr<const MeshConnectivity> connect_3_2
      = topology.connectivity(3, 2);
  if (connect_3_2)
  {
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
          std::int32_t tmp = cell_faces[i];
          cell_faces[i] = cell_faces[j];
          cell_faces[j] = tmp;
          break;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  const int tdim = mesh.topology().dim();
  if (mesh.num_entities(tdim) == 0)
    return;

  // Skip ordering for dimension 0
  if (tdim == 0)
    return;

  // Get global vertex numbering
  if (!mesh.topology().have_global_indices(0))
    throw std::runtime_error("Mesh does not have global vertex indices.");
  const std::vector<std::int64_t>& local_to_global_vertex_indices
      = mesh.topology().global_indices(0);

  // Iterate over all cells
  for (mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
    order_simplex(local_to_global_vertex_indices, cell);

  // // // Iterate over all cells and order the mesh entities locally
  // // for (CellIterator cell(mesh, "all"); !cell.end(); ++cell)
  // //   cell->order(local_to_global_vertex_indices);
}
//-----------------------------------------------------------------------------
bool MeshOrdering::ordered(const Mesh& mesh)
{
  const int tdim = mesh.topology().dim();
  if (mesh.num_entities(tdim) == 0)
    return true;

  // // Get global vertex numbering
  // dolfin_assert(mesh.topology().have_global_indices(0));
  // const auto& local_to_global_vertex_indices
  //     = mesh.topology().global_indices(0);

  // // Check if all cells are ordered
  // for (CellIterator cell(mesh, "all"); !cell.end(); ++cell)
  // {
  //   if (!cell->ordered(local_to_global_vertex_indices))
  //     return false;
  // }

  return true;
}
//-----------------------------------------------------------------------------
