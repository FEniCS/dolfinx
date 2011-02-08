// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-08

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/IndexSet.h>
#include "Mesh.h"
#include "MeshFunction.h"
#include "Cell.h"
#include "Edge.h"
#include "Vertex.h"
#include "MeshEditor.h"
#include "RegularCutRefinement.h"

// FIXME: Temporary while testing
#include "UnitSquare.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void RegularCutRefinement::refine(Mesh& refined_mesh,
                                  const Mesh& mesh,
                                  const MeshFunction<bool>& cell_markers)
{
  // Currently only implemented in 2D
  if (mesh.topology().dim() != 2)
    error("Regular-cut mesh refinement is currently only implemented in 2D.");

  // Initialize edges
  mesh.init(1);

  // Compute refinement markers
  std::vector<int> refinement_markers;
  IndexSet marked_edges(mesh.num_edges());
  compute_markers(refinement_markers, marked_edges, mesh, cell_markers);

  // Refine mesh based on computed markers
  refine_marked(refined_mesh, mesh, refinement_markers, marked_edges);

  //UnitSquare unit_square(3, 3);
  //refined_mesh = unit_square;
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

  // Iterate until no more cells are marked
  cells.fill();
  while (cells.size() > 0)
  {
    // Iterate over all cells in list
    for (uint i = 0; i < cells.size(); i++)
    {
      // Get cell index
      const uint cell_index = cells[i];

      // Mark edges if cell marked for refinement or more than one edge marked
      if (!cell_markers[cell_index] && count_markers(edge_markers[cell_index]) <= 1)
        continue;

      // Iterate over edges
      Cell cell(mesh, cell_index);
      for (EdgeIterator edge(cell); !edge.end(); ++edge)
      {
        // Mark edge in current cell
        if (!edge_markers[cell_index][edge.pos()])
        {
          edge_markers[cell_index][edge.pos()] = true;
          marked_edges.insert(edge->index());
          marked_cells.insert(cell_index);
        }

        // Iterate over cells sharing edge
        for (CellIterator neighbor(*edge); !neighbor.end(); ++neighbor)
        {
          // Get local edge number of edge relative to neighbor
          const uint local_index = neighbor->index(*edge);

          // Mark edge for refinement
          if (!edge_markers[neighbor->index()][local_index])
          {
            edge_markers[neighbor->index()][local_index] = true;
            marked_edges.insert(edge->index());
            marked_cells.insert(neighbor->index());
          }
        }
      }
    }

    cout << "Number of marked cells: " << marked_cells.size() << endl;

    // Copy marked cells
    cells = marked_cells;
    marked_cells.clear();
  }

  // Debug
  for (uint i = 0; i < edge_markers.size(); i++)
  {
    cout << i << ":";
    for (uint j = 0; j < edge_markers[i].size(); j++)
      cout << " " << edge_markers[i][j];
    cout << endl;
  }

  // Extract which cells to refine and indices which edges to bisect
  refinement_markers.resize(mesh.num_cells());
  for (uint i = 0; i < edge_markers.size(); i++)
  {
    const uint num_marked = count_markers(edge_markers[i]);

    // Mark for regular refinement
    if (num_marked == edges_per_cell)
      refinement_markers[i] = regular_refinement;

    // Mark for edge bisection (edge number)
    else if (num_marked == 1)
      refinement_markers[i] = extract_edge(edge_markers[i]);

    // No refinement
    else if (num_marked == 0)
      refinement_markers[i] = no_refinement;

    // Sanity check
    else
      error("Internal error in mesh refinement, unexpected number of marked edges.");
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
      num_cells += 4; // 2D
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




  // New cells
  std::vector<std::vector<uint> > cells;

  // Iterate over all cells and add new cells
  uint current_cell = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get marker
    const int marker = refinement_markers[cell->index()];

    // No refinement: just copy cell to new mesh
    if (marker == no_refinement)
    {
      cout << "No refinement" << endl;
      std::vector<uint> vertices;
      for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
        vertices.push_back(vertex->index());
      editor.add_cell(current_cell++, vertices);
    }

    // Regular refinement: divide into subsimplicies
    else if (marker == regular_refinement)
    {
      cout << "Regular refinement" << endl;

      // FIXME: Move this part to TriangleCell

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

    // One edge marked for refinement: do bisection
    else
    {
      cout << "Refining edge " << endl;

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
    }
  }

  editor.close();

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
  error("Internal error in mesh refinement, unable to extract edge.");
  return 0;
}
//-----------------------------------------------------------------------------
