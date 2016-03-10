// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-02-07
// Last changed: 2014-02-06

#include <vector>

#include <dolfin/log/log.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/IndexSet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshEditor.h>
#include "RegularCutRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void RegularCutRefinement::refine(Mesh& refined_mesh,
                                  const Mesh& mesh,
                                  const MeshFunction<bool>& cell_markers)
{
  not_working_in_parallel("RegularCutRefinement::refine");

  // Currently only implemented in 2D
  if (mesh.topology().dim() != 2)
  {
    dolfin_error("RegularCutRefinement.cpp",
                 "refine mesh",
                 "Mesh is not two-dimensional: regular-cut mesh refinement is currently only implemented in 2D");
  }

  // Check that mesh is ordered
  if (!mesh.ordered())
  {
    dolfin_error("RegularCutRefinement.cpp",
                 "refine mesh",
                 "Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order()");
  }

  // Initialize edges
  mesh.init(1);

  // Compute refinement markers
  std::vector<int> refinement_markers;
  IndexSet marked_edges(mesh.num_edges());
  compute_markers(refinement_markers, marked_edges, mesh, cell_markers);

  // Refine mesh based on computed markers
  refine_marked(refined_mesh, mesh, refinement_markers, marked_edges);
}
//-----------------------------------------------------------------------------
void
RegularCutRefinement::compute_markers(std::vector<int>& refinement_markers,
                                      IndexSet& marked_edges,
                                      const Mesh& mesh,
                                      const MeshFunction<bool>& cell_markers)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Create edge markers and initialize to false
  const std::size_t edges_per_cell = D + 1;
  std::vector<std::vector<bool>> edge_markers(mesh.num_cells());
  for (std::size_t i = 0; i < mesh.num_cells(); i++)
  {
    edge_markers[i].resize(edges_per_cell);
    for (std::size_t j = 0; j < edges_per_cell; j++)
      edge_markers[i][j] = false;
  }

  // Create index sets for marked cells
  IndexSet cells(mesh.num_cells());
  IndexSet marked_cells(mesh.num_cells());

  // Get bisection data
  const std::vector<std::size_t>* bisection_twins = NULL;
  if (mesh.data().exists("bisection_twins", D))
    bisection_twins = &(mesh.data().array("bisection_twins", D));

  // Iterate until no more cells are marked
  cells.fill();
  while (!cells.empty())
  {
    // Iterate over all cells in list
    for (std::size_t _i = 0; _i < cells.size(); _i++)
    {
      // Get cell index and create cell
      const std::size_t cell_index = cells[_i];
      const Cell cell(mesh, cell_index);

      // Count the number of marked edges
      const std::size_t num_marked = count_markers(edge_markers[cell_index]);

      // Check whether cell has a bisection twin
      std::size_t bisection_twin = cell_index;
      bool is_bisected = false;
      if (bisection_twins)
      {
        bisection_twin = (*bisection_twins)[cell_index];
        is_bisected = bisection_twin != cell_index;
      }

      // Get bisection edge
      std::size_t common_edge = 0;
      std::size_t bisection_edge = 0;
      if (is_bisected)
      {
        common_edge = find_common_edges(cell, mesh, bisection_twin).first;
        bisection_edge = find_bisection_edges(cell, mesh, bisection_twin).first;
      }

      // Decide if cell should be refined
      bool refine = false;
      refine = refine || cell_markers[cell_index];
      if (is_bisected)
        refine = refine || num_marked > 0;
      else
      {
        refine = refine || num_marked > 1;
        refine = refine || too_thin(cell, edge_markers[cell_index]);
      }

      // Skip cell if it should not be marked
      if (!refine)
        continue;

      // Iterate over edges
      for (EdgeIterator edge(cell); !edge.end(); ++edge)
      {
        // Skip edge if it is a bisected edge of a bisected cell
        if (is_bisected && edge.pos() == bisection_edge)
          continue;

        // Mark edge in current cell
        if (!edge_markers[cell_index][edge.pos()])
        {
          edge_markers[cell_index][edge.pos()] = true;
          marked_cells.insert(cell_index);
        }

        // Insert edge into set of marked edges but only if the edge
        // is not the common edge of a bisected cell in which case it
        // will later be removed and no new vertex be inserted...
        if (!is_bisected || edge.pos() != common_edge)
          marked_edges.insert(edge->index());

        // Iterate over cells sharing edge
        for (CellIterator neighbor(*edge); !neighbor.end(); ++neighbor)
        {
          // Get local edge number of edge relative to neighbor
          const std::size_t local_index = neighbor->index(*edge);

          // Mark edge for refinement
          if (!edge_markers[neighbor->index()][local_index])
          {
            edge_markers[neighbor->index()][local_index] = true;
            marked_cells.insert(neighbor->index());
          }
        }
      }
    }

    // Copy marked cells
    cells = marked_cells;
    marked_cells.clear();
  }

  // Extract which cells to refine and indices which edges to bisect
  refinement_markers.resize(mesh.num_cells());
  for (std::size_t i = 0; i < edge_markers.size(); i++)
  {
    // Count the number of marked edges
    const std::size_t num_marked = count_markers(edge_markers[i]);

    // Check if cell has been bisected before
    const bool is_bisected = bisection_twins && (*bisection_twins)[i] != i;

    // No refinement
    if (num_marked == 0)
      refinement_markers[i] = static_cast<int>(marker_type::no_refinement);

    // Mark for bisection
    else if (num_marked == 1 && !is_bisected)
      refinement_markers[i] = extract_edge(edge_markers[i]);

    // Mark for regular refinement
    else if (num_marked == edges_per_cell && !is_bisected)
      refinement_markers[i] = static_cast<int>(marker_type::regular_refinement);

    // Mark for bisection backtracking
    else if (num_marked == 2 && is_bisected)
      refinement_markers[i] = static_cast<int>(marker_type::backtrack_bisection);

    // Mark for bisection backtracking and refinement
    else if (num_marked == edges_per_cell && is_bisected)
      refinement_markers[i] = static_cast<int>(marker_type::backtrack_bisection_refine);

    // Sanity check
    else
    {
      dolfin_error("RegularCutRefinement.cpp",
                   "compute marked edges",
                   "Unexpected number of edges marked");
    }
  }
}
//-----------------------------------------------------------------------------
void RegularCutRefinement::refine_marked(Mesh& refined_mesh,
                                  const Mesh& mesh,
                                  const std::vector<int>& refinement_markers,
                                  const IndexSet& marked_edges)
{
  // Count the number of cells in refined mesh
  std::size_t num_cells = 0;

  // Data structure to hold a cell
  std::vector<std::size_t> cell_data(3);

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const int marker = refinement_markers[cell->index()];
    switch (marker)
    {
    case static_cast<int>(marker_type::no_refinement):
      num_cells += 1;
      break;
    case static_cast<int>(marker_type::regular_refinement):
      num_cells += 4;
      break;
    case static_cast<int>(marker_type::backtrack_bisection):
      num_cells += 2;
      break;
    case static_cast<int>(marker_type::backtrack_bisection_refine):
      num_cells += 3;
      break;
    default:
      num_cells += 2;
    }
  }

  // Initialize mesh editor
  const std::size_t num_vertices = mesh.num_vertices() + marked_edges.size();
  MeshEditor editor;
  editor.open(refined_mesh, mesh.topology().dim(), mesh.geometry().dim());
  editor.init_vertices_global(num_vertices, num_vertices);
  editor.init_cells_global(num_cells, num_cells);

  // Set vertex coordinates
  std::size_t current_vertex = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    editor.add_vertex(current_vertex, vertex->point());
    current_vertex++;
  }
  for (std::size_t i = 0; i < marked_edges.size(); i++)
  {
    Edge edge(mesh, marked_edges[i]);
    editor.add_vertex(current_vertex, edge.midpoint());
    current_vertex++;
  }

  // Get bisection data for old mesh
  const std::size_t D = mesh.topology().dim();
  const std::vector<std::size_t>*  bisection_twins = NULL;
  if (mesh.data().exists("bisection_twins", D))
    bisection_twins = &(mesh.data().array("bisection_twins", D));

  // Markers for bisected cells pointing to their bisection twins in
  // refined mesh
  std::vector<std::size_t>& refined_bisection_twins
    = refined_mesh.data().create_array("bisection_twins", D);
  refined_bisection_twins.resize(num_cells);
  for (std::size_t i = 0; i < num_cells; i++)
    refined_bisection_twins[i] = i;

  // Mapping from old to new unrefined cells (-1 means refined or not
  // yet processed)
  std::vector<int> unrefined_cells(mesh.num_cells());
  std::fill(unrefined_cells.begin(), unrefined_cells.end(), -1);

  // Iterate over all cells and add new cells
  std::size_t current_cell = 0;
  std::vector<std::vector<std::size_t>> cells(4, std::vector<std::size_t>(3));
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get marker
    const int marker = refinement_markers[cell->index()];
    if (marker == static_cast<int>(marker_type::no_refinement))
    {
      // No refinement: just copy cell to new mesh
      std::vector<std::size_t> vertices;
      for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
        vertices.push_back(vertex->index());
      editor.add_cell(current_cell++, vertices);

      // Store mapping to new cell index
      unrefined_cells[cell->index()] = current_cell - 1;

      // Remember unrefined bisection twins
      if (bisection_twins)
      {
        const std::size_t bisection_twin = (*bisection_twins)[cell->index()];
        const int twin_marker = refinement_markers[bisection_twin];
        dolfin_assert(twin_marker == static_cast<int>(marker_type::no_refinement));
        if (unrefined_cells[bisection_twin] >= 0)
        {
          const std::size_t i = current_cell - 1;
          const std::size_t j = unrefined_cells[bisection_twin];
          refined_bisection_twins[i] = j;
          refined_bisection_twins[j] = i;
        }
      }
    }
    else if (marker == static_cast<int>(marker_type::regular_refinement))
    {
      // Regular refinement: divide into sub-simplices
      dolfin_assert(unrefined_cells[cell->index()] == -1);

      // Get vertices and edges
      const unsigned int* v = cell->entities(0);
      const unsigned int* e = cell->entities(1);
      dolfin_assert(v);
      dolfin_assert(e);

      // Get offset for new vertex indices
      const std::size_t offset = mesh.num_vertices();

      // Compute indices for the six new vertices
      const std::size_t v0 = v[0];
      const std::size_t v1 = v[1];
      const std::size_t v2 = v[2];
      const std::size_t e0 = offset + marked_edges.find(e[0]);
      const std::size_t e1 = offset + marked_edges.find(e[1]);
      const std::size_t e2 = offset + marked_edges.find(e[2]);

      // Create four new cells
      cells[0][0] = v0; cells[0][1] = e2; cells[0][2] = e1;
      cells[1][0] = v1; cells[1][1] = e0; cells[1][2] = e2;
      cells[2][0] = v2; cells[2][1] = e1; cells[2][2] = e0;
      cells[3][0] = e0; cells[3][1] = e1; cells[3][2] = e2;

      // Add cells
      std::vector<std::vector<std::size_t>>::const_iterator _cell;
      for (_cell = cells.begin(); _cell != cells.end(); ++_cell)
        editor.add_cell(current_cell++, *_cell);
    }
    else if (marker == static_cast<int>(marker_type::backtrack_bisection)
             || marker ==static_cast<int>(marker_type::backtrack_bisection_refine))
    {
      // Special case: backtrack bisected cells
      dolfin_assert(unrefined_cells[cell->index()] == -1);

      // Get index for bisection twin
      dolfin_assert(bisection_twins);
      const std::size_t bisection_twin = (*bisection_twins)[cell->index()];
      dolfin_assert(bisection_twin != cell->index());

      // Let lowest number twin handle refinement
      if (bisection_twin < cell->index())
        continue;

      // Get marker for twin
      const int twin_marker = refinement_markers[bisection_twin];

      // Find common edge(s) and bisected edge(s)
      const std::pair<std::size_t, std::size_t> common_edges
        = find_common_edges(*cell, mesh, bisection_twin);
      const std::pair<std::size_t, std::size_t> bisection_edges
        = find_bisection_edges(*cell, mesh, bisection_twin);
      const std::pair<std::size_t, std::size_t> bisection_vertices
        = find_bisection_vertices(*cell, mesh, bisection_twin, bisection_edges);

      // Get list of vertices and edges for both cells
      const Cell twin(mesh, bisection_twin);
      const unsigned int* vertices_0 = cell->entities(0);
      const unsigned int* vertices_1 = twin.entities(0);
      const unsigned int* edges_0 = cell->entities(1);
      const unsigned int* edges_1 = twin.entities(1);
      dolfin_assert(vertices_0);
      dolfin_assert(vertices_1);
      dolfin_assert(edges_0);
      dolfin_assert(edges_1);

      // Get offset for new vertex indices
      const std::size_t offset = mesh.num_vertices();

      // Locate vertices such that v_i is the vertex opposite to
      // the edge e_i on the parent triangle
      const std::size_t v0 = vertices_0[common_edges.first];
      const std::size_t v1 = vertices_1[common_edges.second];
      const std::size_t v2 = vertices_0[bisection_edges.first];
      const std::size_t e0 = offset
        + marked_edges.find(edges_1[bisection_vertices.second]);
      const std::size_t e1 = offset
        + marked_edges.find(edges_0[bisection_vertices.first]);
      const std::size_t e2 = vertices_0[bisection_vertices.first];

      // Locate new vertices on bisected edge (if any)
      std::size_t E0 = 0;
      std::size_t E1 = 0;
      if (marker == static_cast<int>(marker_type::backtrack_bisection_refine))
        E0 = offset + marked_edges.find(edges_0[bisection_edges.first]);
      if (twin_marker
          == static_cast<int>(marker_type::backtrack_bisection_refine))
      {
        E1 = offset + marked_edges.find(edges_1[bisection_edges.second]);
      }

      // Add middle two cells (always)
      dolfin_assert(cell_data.size() == 3);
      cell_data[0] = e0; cell_data[1] = e1; cell_data[2] = e2;
      editor.add_cell(current_cell++, cell_data);

      cell_data[0] = v2; cell_data[1] = e1; cell_data[2] = e0;
      editor.add_cell(current_cell++, cell_data);

      // Add one or two remaining cells in current cell (left)
      if (marker == static_cast<int>(marker_type::backtrack_bisection))
      {
        cell_data[0] = v0; cell_data[1] = e2; cell_data[2] = e1;
        editor.add_cell(current_cell++, cell_data);
      }
      else
      {
        // Add the two cells
        cell_data[0] = v0; cell_data[1] = E0; cell_data[2] = e1;
        editor.add_cell(current_cell++, cell_data);

        cell_data[0] = E0; cell_data[1] = e2; cell_data[2] = e1;
        editor.add_cell(current_cell++, cell_data);

        // Set bisection twins
        refined_bisection_twins[current_cell - 2] = current_cell - 1;
        refined_bisection_twins[current_cell - 1] = current_cell - 2;
      }

      // Add one or two remaining cells in twin cell (right)
      if (twin_marker == static_cast<int>(marker_type::backtrack_bisection))
      {
        cell_data[0] = v1; cell_data[1] = e0; cell_data[2] = e2;
        editor.add_cell(current_cell++, cell_data);
      }
      else
      {
        // Add the two cells
        cell_data[0] = v1; cell_data[1] = e0; cell_data[2] = E1;
        editor.add_cell(current_cell++, cell_data);

        cell_data[0] = e0; cell_data[1] = e2; cell_data[2] = E1;
        editor.add_cell(current_cell++, cell_data);

        // Set bisection twins
        refined_bisection_twins[current_cell - 2] = current_cell - 1;
        refined_bisection_twins[current_cell - 1] = current_cell - 2;
      }
    }
    else
    {
      // One edge marked for refinement: do bisection

      dolfin_assert(unrefined_cells[cell->index()] == -1);

      // Get vertices and edges
      const unsigned int* v = cell->entities(0);
      const unsigned int* e = cell->entities(1);
      dolfin_assert(v);
      dolfin_assert(e);

      // Get edge number (equal to marker)
      dolfin_assert(marker >= 0);
      const std::size_t local_edge_index = static_cast<std::size_t>(marker);
      const std::size_t global_edge_index = e[local_edge_index];
      const std::size_t ee = mesh.num_vertices() + marked_edges.find(global_edge_index);

      // Add the two new cells
      if (local_edge_index == 0)
      {
        cell_data[0] = v[0]; cell_data[1] = ee; cell_data[2] = v[1];
        editor.add_cell(current_cell++, cell_data);

        cell_data[0] = v[0]; cell_data[1] = ee; cell_data[2] = v[2];
        editor.add_cell(current_cell++, cell_data);
      }
      else if (local_edge_index == 1)
      {
        cell_data[0] = v[1]; cell_data[1] = ee; cell_data[2] = v[0];
        editor.add_cell(current_cell++, cell_data);

        cell_data[0] = v[1]; cell_data[1] = ee; cell_data[2] = v[2];
        editor.add_cell(current_cell++, cell_data);
      }
      else
      {
        cell_data[0] = v[2]; cell_data[1] = ee; cell_data[2] = v[0];
        editor.add_cell(current_cell++, cell_data);

        cell_data[0] = v[2]; cell_data[1] = ee; cell_data[2] = v[1];
        editor.add_cell(current_cell++, cell_data);
      }

      // Set bisection twins
      refined_bisection_twins[current_cell - 2] = current_cell - 1;
      refined_bisection_twins[current_cell - 1] = current_cell - 2;
    }
  }

  // Close mesh editor
  dolfin_assert(num_cells == current_cell);
  editor.close();
}
//-----------------------------------------------------------------------------
std::size_t RegularCutRefinement::count_markers(const std::vector<bool>& markers)
{
  std::size_t num_markers = 0;
  for (std::size_t i = 0; i < markers.size(); i++)
    if (markers[i])
      num_markers++;
  return num_markers;
}
//-----------------------------------------------------------------------------
std::size_t RegularCutRefinement::extract_edge(const std::vector<bool>& markers)
{
  for (std::size_t i = 0; i < markers.size(); i++)
    if (markers[i])
      return i;

  dolfin_error("RegularCutRefinement.cpp",
               "extract edge",
               "Internal error in algorithm: edge not found");
  return 0;
}
//-----------------------------------------------------------------------------
bool RegularCutRefinement::too_thin(const Cell& cell,
                                    const std::vector<bool>& edge_markers)
{
  const std::size_t num_markers = count_markers(edge_markers);

  // Only care about the case when one edge is marked
  if (num_markers != 1)
    return false;

  // Compute lengths of all edges
  std::vector<double> lengths;
  double L = 0.0;
  for (EdgeIterator edge(cell); !edge.end(); ++edge)
  {
    const double l = edge->length();
    L = std::max(L, l);
    lengths.push_back(l);
  }

  // Get length of marked edge
  const double l = lengths[extract_edge(edge_markers)];

  // Check condition
  const bool too_thin = l < 0.5*L;

  return too_thin;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
RegularCutRefinement::find_common_edges(const Cell& cell,
                                        const Mesh& mesh,
                                        std::size_t bisection_twin)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const unsigned int* e0 = cell.entities(1);
  const unsigned int* e1 = twin.entities(1);
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Iterate over all combinations of edges
  const std::size_t num_edges = cell.num_entities(1);
  for (std::size_t i = 0; i < num_edges; i++)
    for (std::size_t j = 0; j < num_edges; j++)
      if (e0[i] == e1[j])
        return std::make_pair(i, j);

  // Not found
  dolfin_error("RegularCutRefinement.cpp",
               "find common edges between cells",
               "Internal error in algorithm: common edge not found");

  return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
RegularCutRefinement::find_bisection_edges(const Cell& cell,
                                           const Mesh& mesh,
                                           std::size_t bisection_twin)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const unsigned int* e0 = cell.entities(1);
  const unsigned int* e1 = twin.entities(1);
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Iterate over all combinations of edges
  const std::size_t num_edges = cell.num_entities(1);
  for (std::size_t i = 0; i < num_edges; i++)
  {
    // Get list of vertices for edge
    const Edge edge_0(mesh, e0[i]);
    const unsigned int* v0 = edge_0.entities(0);
    dolfin_assert(v0);

    for (std::size_t j = 0; j < num_edges; j++)
    {
      // Don't test against the edge itself
      if (e0[i] == e1[j])
        continue;

      // Get list of vertices for edge
      const Edge edge_1(mesh, e1[j]);
      const unsigned int* v1 = edge_1.entities(0);
      dolfin_assert(v1);

      // Check that we have a common vertex
      if (v0[0] != v1[0] && v0[0] != v1[1] && v0[1] != v1[0] && v0[1] != v1[1])
        continue;

      // Compute normalized dot product between edges
      double dot_product = edge_0.dot(edge_1);
      dot_product /= edge_0.length() * edge_1.length();

      // Bisection edge found if dot product is small
      if (std::abs(std::abs(dot_product) - 1.0) < DOLFIN_EPS_LARGE)
        return std::make_pair(i, j);
    }
  }

  // Not found
  dolfin_error("RegularCutRefinement.cpp",
               "find edge for bisection",
               "Internal error in algorithm; bisection edge not found");

  return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
RegularCutRefinement::find_bisection_vertices(const Cell& cell,
                                              const Mesh& mesh,
                                              std::size_t bisection_twin,
                                              const std::pair<std::size_t, std::size_t>& bisection_edges)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const unsigned int* e0 = cell.entities(1);
  const unsigned int* e1 = twin.entities(1);
  dolfin_assert(e0);
  dolfin_assert(e1);

  // Get vertices of the two edges
  Edge edge_0(mesh, e0[bisection_edges.first]);
  Edge edge_1(mesh, e1[bisection_edges.second]);
  const unsigned int* v0 = edge_0.entities(0);
  const unsigned int* v1 = edge_1.entities(0);
  dolfin_assert(v0);
  dolfin_assert(v1);

  // Find common vertex
  if (v0[0] == v1[0] || v0[0] == v1[1])
  {
    Vertex v(mesh, v0[0]);
    return std::make_pair(cell.index(v), twin.index(v));
  }
  if (v0[1] == v1[0] || v0[1] == v1[1])
  {
    Vertex v(mesh, v0[1]);
    return std::make_pair(cell.index(v), twin.index(v));
  }

  // Not found
  dolfin_error("RegularCutRefinement.cpp",
               "find bisection vertices",
               "Internal error in algorithm: bisection vertex not found");

  return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
