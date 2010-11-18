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
#include <dolfin/common/utils.h>
#include <dolfin/graph/ZoltanInterface.h>
#include "Cell.h"
#include "Edge.h"
#include "Facet.h"
#include "Mesh.h"
#include "Vertex.h"
#include "MeshColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>&
MeshColoring::color_cells(Mesh& mesh, std::string coloring_type)
{
  return color_cells(mesh, type_to_dim(coloring_type, mesh));
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>&
MeshColoring::color_cells(Mesh& mesh, uint dim)
{
  info("Coloring mesh.");

  // Get mesh data
  MeshData& data = mesh.data();

  // Clear old coloring data if any
  std::vector<uint>* num_colored_cells = data.array("num colored cells");
  if (num_colored_cells)
  {
    info("Erasing existing mesh coloring data.");
    for (uint c = 0; c < num_colored_cells->size(); c++)
      data.erase_array("colored cells " + to_string(c));
    data.erase_mesh_function("cell colors");
    data.erase_array("num colored cells");
    num_colored_cells = 0;
  }

  // Create mesh function for cell colors
  assert(data.mesh_function("cell colors") == 0);
  MeshFunction<uint>* colors = data.create_mesh_function("cell colors", mesh.topology().dim());
  assert(colors);

  // Compute coloring
  MeshColoring::compute_cell_colors(*colors, dim);

  // Extract cells for each color
  std::vector<std::vector<uint>* > colored_cells;
  for (uint i = 0; i < colors->size(); i++)
  {
    // Get current color
    const uint color = (*colors)[i];

    // Extend list of colors if necessary
    if (color >= colored_cells.size())
    {
      // Append empty lists for all colors up to current color
      for (uint c = colored_cells.size(); c <= color; c++)
      {
        assert(data.array("colored cells", c) == 0);
        colored_cells.push_back(data.create_array("colored cells " + to_string(c)));
      }
    }

    // Add color to list if color has been seen before
    assert(color < colored_cells.size());
    colored_cells[color]->push_back(i);
  }

  // Count the number of cells of each color
  assert(data.array("num colored cells") == 0);
  num_colored_cells = data.create_array("num colored cells");
  assert(num_colored_cells);
  for (uint c = 0; c < colored_cells.size(); c++)
  {
    info("Color %d: %d cells", c, colored_cells[c]->size());
    num_colored_cells->push_back(colored_cells[c]->size());
  }
  info("Mesh has %d colors.", num_colored_cells->size());

  return *colors;
}
//-----------------------------------------------------------------------------
void MeshColoring::compute_cell_colors(MeshFunction<uint>& colors,
                                       std::string coloring_type)
{
  compute_cell_colors(colors, type_to_dim(coloring_type, colors.mesh()));
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
dolfin::uint MeshColoring::type_to_dim(std::string coloring_type,
                                       const Mesh& mesh)
{
  // Check that coloring type is valid
  if (coloring_type != "vertex" && coloring_type != "edge" && coloring_type != "facet")
    error("Coloring type unkown. Options are \"vertex\", \"edge\" or \"facet\".");

  // Select topological dimension
  if (coloring_type == "vertex")
    return 0;
  else if (coloring_type == "edge")
    return 1;
  else
    return mesh.topology().dim() - 1;
}
//-----------------------------------------------------------------------------

#endif
