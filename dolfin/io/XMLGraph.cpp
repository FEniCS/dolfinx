// Copyright (C) 2009 Ola Skavhaug and Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// This file is a port of Magnus Vikstrom's previous implementation
//
// First added:  2009-03-11
// Last changed: 2000-03-17

#include <dolfin/log/dolfin_log.h>
#include <dolfin/graph/Graph.h>
#include "XMLIndent.h"
#include "XMLFile.h"
#include "XMLGraph.h"

using namespace dolfin;
using dolfin::uint;

//-----------------------------------------------------------------------------
XMLGraph::XMLGraph(Graph& graph, XMLFile& parser)
: XMLHandler(parser), graph(graph), state(OUTSIDE)
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
void XMLGraph::write(const Graph& graph, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Get connections (outgoing edges), offsets and weigts
  const uint* connections = graph.connectivity();
  const uint* offsets = graph.offsets();
  const uint* edge_weights = graph.edge_weights();
  const uint* vertex_weights = graph.vertex_weights();

  // Make sure data is fine
  dolfin_assert(connections);
  dolfin_assert(offsets);
  dolfin_assert(edge_weights);
  dolfin_assert(vertex_weights);

  // Get number of vertices
  const uint num_vertices = graph.num_vertices();

  // Write graph header
  outfile << indent() << "<graph type=\"" << graph.typestr() << "\">" << std::endl;

  // Write vertices header
  ++indent;
  outfile << indent();
  outfile << "<vertices size=\"" << graph.num_vertices() << "\">" << std::endl;

  // Write each vertex
  ++indent;
  for(uint i = 0; i < num_vertices; ++i)
  {
    outfile << indent();
    outfile << "<vertex index=\"" << i << "\" num_edges=\"" << graph.num_edges(i) << "\" weight=\"" << vertex_weights[i] << "\"/>" << std::endl;
  }
  --indent;

  // Write vertices footer
  outfile << indent() << "</vertices>" << std::endl;

  // Write edges header
  outfile << indent();
  outfile << "<edges size=\">" << graph.num_edges() << "\">" << std::endl;

  // Write each edge
  ++indent;
  for(uint i = 0; i < num_vertices; ++i)
  {
    for(uint j=offsets[i]; j<offsets[i] + graph.num_edges(i); ++j)
    {
      outfile << indent();
      outfile << "<edge v1=\"" << i << "\" v2=\"" << connections[j] << "\" weight=\"" << edge_weights[j] << "\"/>" << std::endl;
    }
  }
  --indent;

  // Write edges footer
  outfile << indent() << "</edges>" << std::endl;

  // Write graph footer
  --indent;
  outfile << indent() << "</graph>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLGraph::read_graph(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parse_string(name, attrs, "type");

  // Open graph for editing
  editor.open(graph, type);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("read_vertices()");
  uint num_vertices = parse_uint(name, attrs, "size");
  editor.init_vertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_edges(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("read_edges()");
  uint num_edges = parse_uint(name, attrs, "size");
  editor.init_edges(num_edges);
}
//-----------------------------------------------------------------------------
void XMLGraph::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  editor.add_vertex(parse_uint(name, attrs, "index"), parse_uint(name, attrs, "num_edges"));
}
//-----------------------------------------------------------------------------
void XMLGraph::read_edge(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");

  //dolfin_debug2("read_edge, v1 = %d, v2 = %d", v1, v2);

  // Edge weights not yet implemented
  //uint w = parse_uint(name, attrs, "weight");
  editor.add_edge(v1, v2);
}
//-----------------------------------------------------------------------------
void XMLGraph::close_graph()
{
  editor.close();
}
//-----------------------------------------------------------------------------
