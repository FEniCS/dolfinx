// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed: 2010-11-16

#ifdef HAS_TRILINOS

#include <boost/foreach.hpp>
#include "dolfin/log/log.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/Edge.h"
#include "dolfin/mesh/Facet.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/Vertex.h"
#include "ZoltanInterface.h"
#include "CellColoring.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CellColoring::CellColoring(const Mesh& mesh, std::string type) : mesh(mesh)
{
  // Check that graph type is valid
  if (type != "vertex" && type != "facet" && type != "edge")
    error("Coloring type unkown. Options are \"vertex\", \"facet\" or \"edge\".");

  // Build graph
  if (type == "vertex")
  {
    build_graph<VertexIterator>(mesh, graph);
  }
  else if (type == "facet")
  {
    // Compute facets and facet - cell connectivity if not already computed
    const uint D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);

    build_graph<FacetIterator>(mesh, graph);
  }
  else if (type == "edge")
  {
    // Compute edges and edges - cell connectivity if not already computed
    const uint D = mesh.topology().dim();
    mesh.init(1);
    mesh.init(1, D);

    build_graph<EdgeIterator>(mesh, graph);
  }
}
//-----------------------------------------------------------------------------
CellFunction<dolfin::uint> CellColoring::compute_local_cell_coloring() const
{
  // Create array to hold colours
  CellFunction<uint> colors(mesh);

  // Wrap MeshFunction values
  Array<uint> _colors(mesh.num_cells(), colors.values());

  // Create coloring object
  //ZoltanInterface::graph_color(graph);

  // Color cells
  ZoltanInterface::compute_local_vertex_coloring(graph, _colors);

  return colors;
}
//-----------------------------------------------------------------------------
template<class T> void CellColoring::build_graph(const Mesh& mesh, Graph& graph)
{
  // Resize graph data
  graph.resize(mesh.num_cells());

  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();
    for (T entity(*cell); !entity.end(); ++entity)
    {
      for (CellIterator ncell(*entity); !ncell.end(); ++ncell)
        graph[cell_index].insert(ncell->index());
    }
  }
}
//-----------------------------------------------------------------------------
#endif
