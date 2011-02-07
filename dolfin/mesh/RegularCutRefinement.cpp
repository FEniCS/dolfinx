// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-07

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/IndexSet.h>
#include "Mesh.h"
#include "MeshFunction.h"
#include "Cell.h"
#include "Edge.h"
#include "RegularCutRefinement.h"

// FIXME: Temporary while testing
#include "UnitSquare.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void RegularCutRefinement::refine(Mesh& refined_mesh,
                                  const Mesh& mesh,
                                  const MeshFunction<bool>& cell_markers)
{
  // Compute refinement markers
  compute_markers(mesh, cell_markers);

  //UnitSquare unit_square(3, 3);
  //refined_mesh = unit_square;
}
//-----------------------------------------------------------------------------
void RegularCutRefinement::compute_markers(const Mesh& mesh,
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

  // Create index set of all cell indices
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
