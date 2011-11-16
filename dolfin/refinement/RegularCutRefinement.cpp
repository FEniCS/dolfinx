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
// Last changed: 2011-11-15

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
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
    dolfin_error("RegularCutRefinement.cpp",
                 "refine mesh",
                 "Mesh is not two-dimensional: regular-cut mesh refinement is currently only implemented in 2D");

  // Check that mesh is ordered
  if (!mesh.ordered())
    dolfin_error("RegularCutRefinement.cpp",
                 "refine mesh",
                 "Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order()");

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
void RegularCutRefinement::compute_markers(std::vector<int>& refinement_markers,
                                           IndexSet& marked_edges,
                                           const Mesh& mesh,
                                           const MeshFunction<bool>& cell_markers)
{
  // Create edge markers and initialize to false
  const uint edges_per_cell = mesh.topology().dim() + 1;
  std::vector<std::vector<bool> > edge_markers(mesh.num_cells());
  for (uint i = 0; i < mesh.num_cells(); i++)
  {
    edge_markers[i].resize(edges_per_cell);
    for (uint j = 0; j < edges_per_cell; j++)
      edge_markers[i][j] = false;
  }

  // Create index sets for marked cells
  IndexSet cells(mesh.num_cells());
  IndexSet marked_cells(mesh.num_cells());

  // Get bisection data
  boost::shared_ptr<MeshFunction<unsigned int> > bisection_twins = mesh.data().mesh_function("bisection_twins");

  // Iterate until no more cells are marked
  cells.fill();
  while (cells.size() > 0)
  {
    // Iterate over all cells in list
    for (uint _i = 0; _i < cells.size(); _i++)
    {
      // Get cell index and create cell
      const uint cell_index = cells[_i];
      const Cell cell(mesh, cell_index);

      // Count the number of marked edges
      const uint num_marked = count_markers(edge_markers[cell_index]);

      // Check whether cell has a bisection twin
      uint bisection_twin = cell_index;
      bool is_bisected = false;
      if (bisection_twins)
      {
        bisection_twin = (*bisection_twins)[cell_index];
        is_bisected = bisection_twin != cell_index;
      }

      // Get bisection edge
      uint common_edge = 0;
      uint bisection_edge = 0;
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
          const uint local_index = neighbor->index(*edge);

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
  for (uint i = 0; i < edge_markers.size(); i++)
  {
    // Count the number of marked edges
    const uint num_marked = count_markers(edge_markers[i]);

    // Check if cell has been bisected before
    const bool is_bisected = bisection_twins && (*bisection_twins)[i] != i;

    // No refinement
    if (num_marked == 0)
      refinement_markers[i] = no_refinement;

    // Mark for bisection
    else if (num_marked == 1 && !is_bisected)
      refinement_markers[i] = extract_edge(edge_markers[i]);

    // Mark for regular refinement
    else if (num_marked == edges_per_cell && !is_bisected)
      refinement_markers[i] = regular_refinement;

    // Mark for bisection backtracking
    else if (num_marked == 2 && is_bisected)
      refinement_markers[i] = backtrack_bisection;

    // Mark for bisection backtracking and refinement
    else if (num_marked == edges_per_cell && is_bisected)
      refinement_markers[i] = backtrack_bisection_refine;

    // Sanity check
    else
      dolfin_error("RegularCutRefinement.cpp",
                   "compute marked edges",
                   "Unexpected number of edges marked");
  }
}
//-----------------------------------------------------------------------------
void RegularCutRefinement::refine_marked(Mesh& refined_mesh,
                                         const Mesh& mesh,
                                         const std::vector<int>& refinement_markers,
                                         const IndexSet& marked_edges)
{
  // Count the number of cells in refined mesh
  uint num_cells = 0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const int marker = refinement_markers[cell->index()];
    switch (marker)
    {
    case no_refinement:
      num_cells += 1;
      break;
    case regular_refinement:
      num_cells += 4;
      break;
    case backtrack_bisection:
      num_cells += 2;
      break;
    case backtrack_bisection_refine:
      num_cells += 3;
      break;
    default:
      num_cells += 2;
    }
  }

  // Initialize mesh editor
  const uint num_vertices = mesh.num_vertices() + marked_edges.size();
  MeshEditor editor;
  editor.open(refined_mesh, mesh.topology().dim(), mesh.geometry().dim());
  editor.init_vertices(num_vertices);
  editor.init_cells(num_cells);

  // Set vertex coordinates
  uint current_vertex = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    editor.add_vertex(current_vertex++, vertex->point());
  for (uint i = 0; i < marked_edges.size(); i++)
  {
    Edge edge(mesh, marked_edges[i]);
    editor.add_vertex(current_vertex++, edge.midpoint());
  }

  // Get bisection data for old mesh
  boost::shared_ptr<const MeshFunction<unsigned int> > bisection_twins = mesh.data().mesh_function("bisection_twins");

  // Markers for bisected cells pointing to their bisection twins in refined mesh
  std::vector<uint> refined_bisection_twins(num_cells);
  for (uint i = 0; i < num_cells; i++)
    refined_bisection_twins[i] = i;

  // Mapping from old to new unrefined cells (-1 means refined or not yet processed)
  std::vector<int> unrefined_cells(mesh.num_cells());
  std::fill(unrefined_cells.begin(), unrefined_cells.end(), -1);

  // Iterate over all cells and add new cells
  uint current_cell = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get marker
    const int marker = refinement_markers[cell->index()];

    // No refinement: just copy cell to new mesh
    if (marker == no_refinement)
    {
      std::vector<uint> vertices;
      for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
        vertices.push_back(vertex->index());
      editor.add_cell(current_cell++, vertices);

      // Store mapping to new cell index
      unrefined_cells[cell->index()] = current_cell - 1;

      // Remember unrefined bisection twins
      if (bisection_twins)
      {
        const uint bisection_twin = (*bisection_twins)[cell->index()];
        const int twin_marker = refinement_markers[bisection_twin];
        assert(twin_marker == no_refinement);
        if (unrefined_cells[bisection_twin] >= 0)
        {
          const uint i = current_cell - 1;
          const uint j = unrefined_cells[bisection_twin];
          refined_bisection_twins[i] = j;
          refined_bisection_twins[j] = i;
        }
      }
    }

    // Regular refinement: divide into subsimplicies
    else if (marker == regular_refinement)
    {
      assert(unrefined_cells[cell->index()] == -1);

      // Get vertices and edges
      const uint* v = cell->entities(0);
      const uint* e = cell->entities(1);
      assert(v);
      assert(e);

      // Get offset for new vertex indices
      const uint offset = mesh.num_vertices();

      // Compute indices for the six new vertices
      const uint v0 = v[0];
      const uint v1 = v[1];
      const uint v2 = v[2];
      const uint e0 = offset + marked_edges.find(e[0]);
      const uint e1 = offset + marked_edges.find(e[1]);
      const uint e2 = offset + marked_edges.find(e[2]);

      // Add the four new cells
      editor.add_cell(current_cell++, v0, e2, e1);
      editor.add_cell(current_cell++, v1, e0, e2);
      editor.add_cell(current_cell++, v2, e1, e0);
      editor.add_cell(current_cell++, e0, e1, e2);
    }

    // Special case: backtrack bisected cells
    else if (marker == backtrack_bisection || marker == backtrack_bisection_refine)
    {
      assert(unrefined_cells[cell->index()] == -1);

      // Get index for bisection twin
      assert(bisection_twins);
      const uint bisection_twin = (*bisection_twins)[cell->index()];
      assert(bisection_twin != cell->index());

      // Let lowest number twin handle refinement
      if (bisection_twin < cell->index())
        continue;

      // Get marker for twin
      const int twin_marker = refinement_markers[bisection_twin];

      // Find common edge(s) and bisected edge(s)
      const std::pair<uint, uint> common_edges = find_common_edges(*cell, mesh, bisection_twin);
      const std::pair<uint, uint> bisection_edges = find_bisection_edges(*cell, mesh, bisection_twin);
      const std::pair<uint, uint> bisection_vertices = find_bisection_vertices(*cell, mesh, bisection_twin, bisection_edges);

      // Get list of vertices and edges for both cells
      const Cell twin(mesh, bisection_twin);
      const uint* vertices_0 = cell->entities(0);
      const uint* vertices_1 = twin.entities(0);
      const uint* edges_0 = cell->entities(1);
      const uint* edges_1 = twin.entities(1);
      assert(vertices_0);
      assert(vertices_1);
      assert(edges_0);
      assert(edges_1);

      // Get offset for new vertex indices
      const uint offset = mesh.num_vertices();

      // Locate vertices such that v_i is the vertex opposite to
      // the edge e_i on the parent triangle
      const uint v0 = vertices_0[common_edges.first];
      const uint v1 = vertices_1[common_edges.second];
      const uint v2 = vertices_0[bisection_edges.first];
      const uint e0 = offset + marked_edges.find(edges_1[bisection_vertices.second]);
      const uint e1 = offset + marked_edges.find(edges_0[bisection_vertices.first]);
      const uint e2 = vertices_0[bisection_vertices.first];

      // Locate new vertices on bisected edge (if any)
      uint E0 = 0;
      uint E1 = 0;
      if (marker == backtrack_bisection_refine)
        E0 = offset + marked_edges.find(edges_0[bisection_edges.first]);
      if (twin_marker == backtrack_bisection_refine)
        E1 = offset + marked_edges.find(edges_1[bisection_edges.second]);

      // Add middle two cells (always)
      editor.add_cell(current_cell++, e0, e1, e2);
      editor.add_cell(current_cell++, v2, e1, e0);

      // Add one or two remaining cells in current cell (left)
      if (marker == backtrack_bisection)
      {
        editor.add_cell(current_cell++, v0, e2, e1);
      }
      else
      {
        // Add the two cells
        editor.add_cell(current_cell++, v0, E0, e1);
        editor.add_cell(current_cell++, E0, e2, e1);

        // Set bisection twins
        refined_bisection_twins[current_cell - 2] = current_cell - 1;
        refined_bisection_twins[current_cell - 1] = current_cell - 2;
      }

      // Add one or two remaining cells in twin cell (right)
      if (twin_marker == backtrack_bisection)
      {
        editor.add_cell(current_cell++, v1, e0, e2);
      }
      else
      {
        // Add the two cells
        editor.add_cell(current_cell++, v1, e0, E1);
        editor.add_cell(current_cell++, e0, e2, E1);

        // Set bisection twins
        refined_bisection_twins[current_cell - 2] = current_cell - 1;
        refined_bisection_twins[current_cell - 1] = current_cell - 2;
      }
    }

    // One edge marked for refinement: do bisection
    else
    {
      assert(unrefined_cells[cell->index()] == -1);

      // Get vertices and edges
      const uint* v = cell->entities(0);
      const uint* e = cell->entities(1);
      assert(v);
      assert(e);

      // Get edge number (equal to marker)
      assert(marker >= 0);
      const uint local_edge_index = static_cast<uint>(marker);
      const uint global_edge_index = e[local_edge_index];
      const uint ee = mesh.num_vertices() + marked_edges.find(global_edge_index);

      // Add the two new cells
      if (local_edge_index == 0)
      {
        editor.add_cell(current_cell++, v[0], ee, v[1]);
        editor.add_cell(current_cell++, v[0], ee, v[2]);
      }
      else if (local_edge_index == 1)
      {
        editor.add_cell(current_cell++, v[1], ee, v[0]);
        editor.add_cell(current_cell++, v[1], ee, v[2]);
      }
      else
      {
        editor.add_cell(current_cell++, v[2], ee, v[0]);
        editor.add_cell(current_cell++, v[2], ee, v[1]);
      }

      // Set bisection twins
      refined_bisection_twins[current_cell - 2] = current_cell - 1;
      refined_bisection_twins[current_cell - 1] = current_cell - 2;
    }
  }

  // Close mesh editor
  assert(num_cells == current_cell);
  editor.close();

  // Attach data for bisection twins
  boost::shared_ptr<MeshFunction<unsigned int> > _refined_bisection_twins = refined_mesh.data().create_mesh_function("bisection_twins");
  assert(_refined_bisection_twins);
  _refined_bisection_twins->init(refined_mesh.topology().dim());
  _refined_bisection_twins->set_values(refined_bisection_twins);
}
//-----------------------------------------------------------------------------
dolfin::uint RegularCutRefinement::count_markers(const std::vector<bool>& markers)
{
  uint num_markers = 0;
  for (uint i = 0; i < markers.size(); i++)
    if (markers[i])
      num_markers++;
  return num_markers;
}
//-----------------------------------------------------------------------------
dolfin::uint RegularCutRefinement::extract_edge(const std::vector<bool>& markers)
{
  for (uint i = 0; i < markers.size(); i++)
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
  const uint num_markers = count_markers(edge_markers);

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
std::pair<dolfin::uint, dolfin::uint>
RegularCutRefinement::find_common_edges(const Cell& cell,
                                        const Mesh& mesh,
                                        uint bisection_twin)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const uint* e0 = cell.entities(1);
  const uint* e1 = twin.entities(1);
  assert(e0);
  assert(e1);

  // Iterate over all combinations of edges
  const uint num_edges = cell.num_entities(1);
  for (uint i = 0; i < num_edges; i++)
    for (uint j = 0; j < num_edges; j++)
      if (e0[i] == e1[j])
        return std::make_pair(i, j);

  // Not found
  dolfin_error("RegularCutRefinement.cpp",
               "find common edges between cells",
               "Internal error in algorithm: common edge not found");

  return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint>
RegularCutRefinement::find_bisection_edges(const Cell& cell,
                                           const Mesh& mesh,
                                           uint bisection_twin)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const uint* e0 = cell.entities(1);
  const uint* e1 = twin.entities(1);
  assert(e0);
  assert(e1);

  // Iterate over all combinations of edges
  const uint num_edges = cell.num_entities(1);
  for (uint i = 0; i < num_edges; i++)
  {
    // Get list of vertices for edge
    const Edge edge_0(mesh, e0[i]);
    const uint* v0 = edge_0.entities(0);
    assert(v0);

    for (uint j = 0; j < num_edges; j++)
    {
      // Don't test against the edge itself
      if (e0[i] == e1[j])
        continue;

      // Get list of vertices for edge
      const Edge edge_1(mesh, e1[j]);
      const uint* v1 = edge_1.entities(0);
      assert(v1);

      // Check that we have a common vertex
      if (v0[0] != v1[0] && v0[0] != v1[1] && v0[1] != v1[0] && v0[1] != v1[1])
        continue;

      // Compute normalized dot product between edges
      double dot_product = edge_0.dot(edge_1);
      dot_product /= edge_0.length() * edge_1.length();

      // Bisection edge found if dot product is small
      if (std::abs(std::abs(dot_product) - 1.0) < 100.0 * DOLFIN_EPS)
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
std::pair<dolfin::uint, dolfin::uint>
RegularCutRefinement::find_bisection_vertices(const Cell& cell,
                                              const Mesh& mesh,
                                              uint bisection_twin,
                                              const std::pair<uint, uint>& bisection_edges)
{
  // Get list of edges for both cells
  const Cell twin(mesh, bisection_twin);
  const uint* e0 = cell.entities(1);
  const uint* e1 = twin.entities(1);
  assert(e0);
  assert(e1);

  // Get vertices of the two edges
  Edge edge_0(mesh, e0[bisection_edges.first]);
  Edge edge_1(mesh, e1[bisection_edges.second]);
  const uint* v0 = edge_0.entities(0);
  const uint* v1 = edge_1.entities(0);
  assert(v0);
  assert(v1);

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
