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
// Modified by Anders Logg, 2010.
// Modified by Johannes Ring, 2011.
//
// First added:  2010-11-15
// Last changed: 2011-05-11

#include <map>
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
#include "MeshFunction.h"
#include "ParallelData.h"
#include "Vertex.h"
#include "MeshColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& MeshColoring::color_cells(Mesh& mesh,
                                                     std::string coloring_type)
{
  // Define graph type
  std::vector<uint> _coloring_type;
  _coloring_type.push_back(mesh.topology().dim());
  _coloring_type.push_back(type_to_dim(coloring_type, mesh));
  _coloring_type.push_back(mesh.topology().dim());

  return color(mesh, _coloring_type);
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& MeshColoring::color(Mesh& mesh,
                                               std::vector<uint> coloring_type)
{
  // Convenience typedefs
  typedef std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > ColorData;

  info("Coloring mesh.");

  // Get mesh data
  ParallelData& data = mesh.parallel_data();

  // Create empty coloring data
  ColorData _color_data;

  // Clear any old data
  data.coloring.erase(coloring_type);

  // Create coloring data
  data.coloring.insert(std::make_pair(coloring_type, _color_data));

  // Convenience references to data
  assert(data.coloring.find(coloring_type) != data.coloring.end());
  ColorData& color_data = data.coloring.find(coloring_type)->second;

  MeshFunction<uint>& colors = color_data.first;
  std::vector<std::vector<uint> >& entities_of_color = color_data.second;

  // Initialise mesh function for colors and compute coloring
  const uint colored_entity_dim = coloring_type[0];
  colors.init(mesh, colored_entity_dim);
  const uint num_colors = MeshColoring::compute_colors(colors, coloring_type);

  // Build lists of entities for each color
  entities_of_color.resize(num_colors);
  for (uint i = 0; i < colors.size(); i++)
  {
    const uint color = colors[i];
    assert(color < num_colors);
    entities_of_color[color].push_back(i);
  }

  return colors;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshColoring::compute_colors(MeshFunction<uint>& colors,
                                          const std::vector<uint> coloring_type)
{
  // Get the mesh
  const Mesh& mesh(colors.mesh());

  // Get number of graph vertices
  const uint colored_vertex_dim = coloring_type[0];
  //const uint num_verticies = mesh.num_entities(colored_vertex_dim);

  if (coloring_type.front() != coloring_type.back())
    error("MeshColoring::compute_colors does not support dim i - j coloring.");

  // Check that mesh functon has right dimension
  if (colors.dim() != colored_vertex_dim)
    error("MeshFunction has wrong dim. MeshColoring::compute_colors does not support dim i - j coloring.");

  // Create graph
  /*
  BoostBidirectionalGraph graph;
  if (coloring_type.size() == 3)
    graph = boost_graph(mesh, coloring_type[0], coloring_type[1]);
  else
    graph = boost_graph(mesh, coloring_type);
  */
  Graph graph;
  if (coloring_type.size() == 3)
    graph = GraphBuilder::local_graph(mesh, coloring_type[0], coloring_type[1]);
  else
    graph = GraphBuilder::local_graph(mesh, coloring_type);

  // Wrap MeshFunction values
  Array<uint> _colors(colors.size(), colors.values());

  // Color graph
  return GraphColoring::compute_local_vertex_coloring(graph, _colors);
}
//-----------------------------------------------------------------------------
dolfin::uint MeshColoring::type_to_dim(std::string coloring_type,
                                       const Mesh& mesh)
{
  // Check that coloring type is valid
  if (coloring_type != "vertex" && coloring_type != "edge" && coloring_type != "facet")
  {
    error("Coloring type '%s' unknown. Options are \"vertex\", \"edge\" or \"facet\".",
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
