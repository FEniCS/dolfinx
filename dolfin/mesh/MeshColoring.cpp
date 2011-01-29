// Copyright (C) 2010-2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-15
// Last changed: 2011-01-29

#include <map>
#include <utility>
#include <boost/unordered_set.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/utils.h>
#include <dolfin/graph/BoostGraphInterface.h>
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
  MeshData& data = mesh.data();

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
  BoostBidirectionalGraph graph;
  if (coloring_type.size() == 3)
    graph = boost_graph(mesh, coloring_type[0], coloring_type[1]);
  else
    graph = boost_graph(mesh, coloring_type);

  // Wrap MeshFunction values
  Array<uint> _colors(colors.size(), colors.values());

  // Color graph
  return BoostGraphInterface::compute_local_vertex_coloring(graph, _colors);
}
//-----------------------------------------------------------------------------
dolfin::uint MeshColoring::type_to_dim(std::string coloring_type,
                                       const Mesh& mesh)
{
  // Check that coloring type is valid
  if (coloring_type != "vertex" && coloring_type != "edge" && coloring_type != "facet")
  {
    error("Coloring type '%s' unkown. Options are \"vertex\", \"edge\" or \"facet\".",
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
const BoostBidirectionalGraph MeshColoring::boost_graph(const Mesh& mesh,
                                        const std::vector<uint>& coloring_type)
{
  // Create graph
  const uint num_verticies = mesh.num_entities(coloring_type[0]);
  BoostBidirectionalGraph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator vertex_entity(mesh, coloring_type[0]); !vertex_entity.end(); ++vertex_entity)
  {
    boost::unordered_set<uint> entity_list0;
    boost::unordered_set<uint> entity_list1;
    entity_list0.insert(vertex_entity->index());

    // Build list of entities, moving between levels
    for (uint level = 1; level < coloring_type.size(); ++level)
    {
      for (boost::unordered_set<uint>::const_iterator entity_index = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
      {
        const MeshEntity entity(mesh, coloring_type[level -1], *entity_index);
        for (MeshEntityIterator neighbor(entity, coloring_type[level]); !neighbor.end(); ++neighbor)
          entity_list1.insert(neighbor->index());
      }
      entity_list0 = entity_list1;
      entity_list1.clear();
    }

    // Add edges to graph
    const uint vertex_entity_index = vertex_entity->index();
    for (boost::unordered_set<uint>::const_iterator entity_index = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
      boost::add_edge(vertex_entity_index, *entity_index, graph);
  }

  return graph;
}
//-----------------------------------------------------------------------------
const BoostBidirectionalGraph MeshColoring::boost_graph(const Mesh& mesh,
                                                        uint dim0, uint dim1)
{
  // Create graph
  const uint num_verticies = mesh.num_entities(dim0);
  BoostBidirectionalGraph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator colored_entity(mesh, dim0); !colored_entity.end(); ++colored_entity)
  {
    const uint colored_entity_index = colored_entity->index();
    for (MeshEntityIterator entity(*colored_entity, dim1); !entity.end(); ++entity)
    {
      for (MeshEntityIterator neighbor(*entity, dim0); !neighbor.end(); ++neighbor)
        boost::add_edge(colored_entity_index, neighbor->index(), graph);
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
