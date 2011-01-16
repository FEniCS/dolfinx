// Copyright (C) 2010-2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-15
// Last changed: 2011-01-16

#include <map>
#include <utility>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/utils.h>
#include <dolfin/graph/BoostGraphInterface.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Edge.h"
#include "Facet.h"
#include "Mesh.h"
#include "Vertex.h"
#include "MeshColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& MeshColoring::color_cells(Mesh& mesh,
                                                     std::string coloring_type)
{
  const boost::tuple<uint, uint, uint> _coloring_type(mesh.topology().dim(),
                                           type_to_dim(coloring_type, mesh), 1);

  return color(mesh, _coloring_type);
}
//-----------------------------------------------------------------------------
const dolfin::MeshFunction<dolfin::uint>& MeshColoring::color(Mesh& mesh,
                                   boost::tuple<uint, uint, uint> coloring_type)
{
  // Convenience typedefs
  //typedef boost::tuple<uint, uint, uint> ColorType;
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
  const uint colored_entity_dim = coloring_type.get<0>();
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
                                  boost::tuple<uint, uint, uint> coloring_type)
{
  // Get the mesh
  const Mesh& mesh(colors.mesh());

  const uint colored_entity_dim = coloring_type.get<0>();
  const uint dim = coloring_type.get<1>();
  const uint distance = coloring_type.get<2>();

  // Check that dimension match
  if (colored_entity_dim != mesh.topology().dim())
  {
    error("Wrong dimension (%d) for MeshFunction for computation of mesh entity colors.",
          colors.dim());
  }

  if (distance != 1)
    error("Only a 1-distance coloring for meshes is presently supported");

  // Get number of graph vertices
  const uint num_verticies = mesh.num_entities(colored_entity_dim);

  // Create graph
  BoostBidirectionalGraph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator colored_entity(mesh, colored_entity_dim); !colored_entity.end(); ++colored_entity)
  {
    const uint colored_entity_index = colored_entity->index();
    for (MeshEntityIterator entity(*colored_entity, dim); !entity.end(); ++entity)
    {
      for (MeshEntityIterator neighbor(*entity, colored_entity_dim); !neighbor.end(); ++neighbor)
        boost::add_edge(colored_entity_index, neighbor->index(), graph);
    }
  }

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
