// Copyright (C) 2010-2011 Garth N. Wells
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
// Modified by Anders Logg 2010-2011
// Modified by Johannes Ring 2011
//
// First added:  2010-11-15
// Last changed: 2011-11-14

#include <map>
#include <memory>
#include <utility>
#include <dolfin/common/Array.h>
#include <dolfin/common/utils.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/GraphColoring.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Edge.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "Vertex.h"
#include "MeshColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const std::vector<std::size_t>& MeshColoring::color_cells(Mesh& mesh,
                                                     std::string coloring_type)
{
  // Define graph type
  std::vector<std::size_t> _coloring_type;
  _coloring_type.push_back(mesh.topology().dim());
  _coloring_type.push_back(type_to_dim(coloring_type, mesh));
  _coloring_type.push_back(mesh.topology().dim());

  return color(mesh, _coloring_type);
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>&
MeshColoring::color(Mesh& mesh, const std::vector<std::size_t>& coloring_type)
{
  // Convenience typedefs
  typedef std::pair<std::vector<std::size_t>,
                    std::vector<std::vector<std::size_t>>> ColorData;

  info("Coloring mesh.");

  // Create empty coloring data
  ColorData _color_data;

  // Clear any old data
  mesh.topology().coloring.erase(coloring_type);

  // Create coloring data
  mesh.topology().coloring.insert(std::make_pair(coloring_type, _color_data));

  // Convenience references to data
  dolfin_assert(mesh.topology().coloring.find(coloring_type) != mesh.topology().coloring.end());
  ColorData& color_data = mesh.topology().coloring.find(coloring_type)->second;

  std::vector<std::size_t>& colors = color_data.first;
  std::vector<std::vector<std::size_t>>& entities_of_color = color_data.second;

  // Initialise mesh function for colors and compute coloring
  const std::size_t colored_entity_dim = coloring_type[0];
  colors.resize(mesh.num_entities(colored_entity_dim));
  const std::size_t num_colors = MeshColoring::compute_colors(mesh, colors,
                                                       coloring_type);

  // Build lists of entities for each color
  entities_of_color.resize(num_colors);
  for (std::size_t i = 0; i < colors.size(); i++)
  {
    const std::size_t color = colors[i];
    dolfin_assert(color < num_colors);
    entities_of_color[color].push_back(i);
  }

  return colors;
}
//-----------------------------------------------------------------------------
std::size_t
MeshColoring::compute_colors(const Mesh& mesh,
                             std::vector<std::size_t>& colors,
                             const std::vector<std::size_t>& coloring_type)
{
  if (coloring_type.front() != coloring_type.back())
  {
    dolfin_error("MeshColoring.cpp",
                 "compute mesh colors",
                 "Mesh coloring does not support dim i - j coloring");
  }

  // Create graph
  Graph graph;
  if (coloring_type.size() == 3)
    graph = GraphBuilder::local_graph(mesh, coloring_type[0], coloring_type[1]);
  else
    graph = GraphBuilder::local_graph(mesh, coloring_type);

  // Color graph
  return GraphColoring::compute_local_vertex_coloring(graph, colors);
}
//-----------------------------------------------------------------------------
CellFunction<std::size_t>
MeshColoring::cell_colors(std::shared_ptr<const Mesh> mesh,
                          std::string coloring_type)
{
  dolfin_assert(mesh);

  // Get graph/coloring type
  const std::size_t dim = MeshColoring::type_to_dim(coloring_type, *mesh);
  std::vector<std::size_t> _coloring_type;
  _coloring_type.push_back(mesh->topology().dim());
  _coloring_type.push_back(dim);
  _coloring_type.push_back(mesh->topology().dim());

  return cell_colors(mesh, _coloring_type);
}
//-----------------------------------------------------------------------------
CellFunction<std::size_t>
MeshColoring::cell_colors(std::shared_ptr<const Mesh> mesh,
                          std::vector<std::size_t> coloring_type)
{
  dolfin_assert(mesh);

  // Get color data
  std::map<std::vector<std::size_t>, std::pair<std::vector<std::size_t>,
           std::vector<std::vector<std::size_t>>>>::const_iterator
    coloring_data;
  coloring_data = mesh->topology().coloring.find(coloring_type);

  // Check that coloring has been computed
  if (coloring_data == mesh->topology().coloring.end())
  {
    dolfin_error("MeshColoring.cpp",
                 "get coloring as MeshFunction",
                 "Requested coloring has not been computed");
  }

  // Colors
  const std::vector<std::size_t>& colors = coloring_data->second.first;

  CellFunction<std::size_t> mf(mesh);
  dolfin_assert(colors.size() == mesh->num_entities(coloring_type[0]));
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
    mf[*cell] = colors[cell->index()];

  return mf;
}
//-----------------------------------------------------------------------------
std::size_t MeshColoring::type_to_dim(std::string coloring_type,
                                       const Mesh& mesh)
{
  // Check that coloring type is valid
  if (coloring_type != "vertex" && coloring_type != "edge" && coloring_type != "facet")
  {
    dolfin_error("MeshColoring.cpp",
                 "compute mesh colors",
                 "Unknown coloring type (\"%s\"). Known options are \"vertex\", \"edge\" and \"facet\"",
                 coloring_type.c_str());
  }

  // Select topological dimension
  if (coloring_type == "vertex")
    return 0;
  else if (coloring_type == "edge")
    return 1;
  else
    return mesh.topology().dim() - 1;
}
//-----------------------------------------------------------------------------
