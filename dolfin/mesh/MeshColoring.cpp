// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-15
// Last changed: 2010-11-18

#ifdef HAS_TRILINOS

#include <boost/foreach.hpp>

#include <dolfin/log/log.h>
#include <dolfin/graph/ZoltanInterface.h>
#include "Cell.h"
#include "Edge.h"
#include "Facet.h"
#include "Mesh.h"
#include "Vertex.h"
#include "MeshColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshColoring::compute_cell_colors(MeshFunction<uint>& colors,
                                       std::string coloring_type)
{
  // Check that coloring type is valid
  if (coloring_type != "vertex" && coloring_type != "edge" && coloring_type != "facet")
    error("Coloring type unkown. Options are \"vertex\", \"edge\" or \"facet\".");

  // Select topological dimension
  if (coloring_type == "vertex")
    compute_cell_colors(colors, 0);
  else if (coloring_type == "edge")
    compute_cell_colors(colors, 1);
  else
    compute_cell_colors(colors, colors.mesh().topology().dim() - 1);
}
//-----------------------------------------------------------------------------
void MeshColoring::compute_cell_colors(MeshFunction<uint>& colors, uint dim)
{
  // Get the mesh
  const Mesh& mesh(colors.mesh());

  // Check mesh function
  if (colors.dim() != mesh.topology().dim())
    error("Wrong dimension (%d) for MeshFunction for computation of cell colors.",
          colors.dim());

  // Create graph
  Graph graph;
  graph.resize(mesh.num_cells());

  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();
    for (MeshEntityIterator entity(*cell, dim); !entity.end(); ++entity)
    {
      for (CellIterator neighbor(*entity); !neighbor.end(); ++neighbor)
        graph[cell_index].insert(neighbor->index());
    }
  }

  // Wrap MeshFunction values
  Array<uint> _colors(colors.size(), colors.values());

  // Create coloring object
  //ZoltanInterface::graph_color(graph);

  // Color cells
  ZoltanInterface::compute_local_vertex_coloring(graph, _colors);
}
//-----------------------------------------------------------------------------

#endif
