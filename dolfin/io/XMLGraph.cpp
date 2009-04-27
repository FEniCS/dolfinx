// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-02-12
// Last changed: 2007-03-21

#include <dolfin/log/dolfin_log.h>
#include "XMLGraph.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLGraph::XMLGraph(Graph& graph) : XMLObject(), _graph(graph), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLGraph::~XMLGraph()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLGraph::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:

    if ( xmlStrcasecmp(name, (xmlChar *) "graph") == 0 )
    {
      read_graph(name, attrs);
      state = INSIDE_GRAPH;
    }

    break;

  case INSIDE_GRAPH:

    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      read_vertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "edges") == 0 )
    {
      read_edges(name, attrs);
      state = INSIDE_EDGES;
    }

    break;

  case INSIDE_VERTICES:

    if ( xmlStrcasecmp(name, (xmlChar *) "vertex") == 0 )
      read_vertex(name, attrs);

    break;

  case INSIDE_EDGES:

    if ( xmlStrcasecmp(name, (xmlChar *) "edge") == 0 )
      read_edge(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLGraph::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_GRAPH:
    if ( xmlStrcasecmp(name, (xmlChar *) "graph") == 0 )
    {
      close_graph();
      state = DONE;
    }

    break;

  case INSIDE_VERTICES:

    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
    {
      state = INSIDE_GRAPH;
    }

    break;

  case INSIDE_EDGES:

    if ( xmlStrcasecmp(name, (xmlChar *) "edges") == 0)
    {
      state = INSIDE_GRAPH;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLGraph::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool XMLGraph::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLGraph::read_graph(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parse_string(name, attrs, "type");

  // Open graph for editing
  editor.open(_graph, type);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("read_vertices()");
  uint num_vertices = parseUnsignedInt(name, attrs, "size");
  editor.init_vertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_edges(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("read_edges()");
  uint num_edges = parseUnsignedInt(name, attrs, "size");
  editor.init_edges(num_edges);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  current_vertex = parseUnsignedInt(name, attrs, "index");

  // Read number of incident edges
  uint num_edges = parseUnsignedInt(name, attrs, "num_edges");

  // Vertex weights not yet implemented
  //uint w = parseUnsignedInt(name, attrs, "weight");

  editor.add_vertex(current_vertex, num_edges);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_edge(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");

  //dolfin_debug2("read_edge, v1 = %d, v2 = %d", v1, v2);

  // Edge weights not yet implemented
  //uint w = parseUnsignedInt(name, attrs, "weight");
  editor.add_edge(v1, v2);
}
//-----------------------------------------------------------------------------
void XMLGraph::close_graph()
{
  editor.close();
}
//-----------------------------------------------------------------------------
