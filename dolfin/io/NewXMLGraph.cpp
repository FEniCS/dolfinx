// Copyright (C) 2009 Ola Skavhaug and Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// This file is a port of Magnus Vikstrom's previous implementation
//
// First added:  2009-03-11
// Last changed: 2000-03-11

#include <dolfin/log/dolfin_log.h>
#include <dolfin/graph/Graph.h>
#include "NewXMLGraph.h"

using namespace dolfin;
using dolfin::uint;

//-----------------------------------------------------------------------------
NewXMLGraph::NewXMLGraph(Graph& graph, NewXMLFile& parser) 
: XMLHandler(parser), graph(graph), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewXMLGraph::~NewXMLGraph()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLGraph::start_element(const xmlChar *name, const xmlChar **attrs)
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
void NewXMLGraph::end_element(const xmlChar *name)
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
void NewXMLGraph::write(const Graph& graph, std::ostream& outfile, uint indentation_level)
{
  uint curr_indent = indentation_level;

  // Get connections (outgoing edges), offsets and weigts
  const uint* connections = graph.connectivity();
  const uint* offsets = graph.offsets();
  const uint* edge_weights = graph.edgeWeights();
  const uint* vertex_weights = graph.vertexWeights();

  // Make sure data is fine
  dolfin_assert(connections);
  dolfin_assert(offsets);
  dolfin_assert(edge_weights);
  dolfin_assert(vertex_weights);

  // Get number of vertices
  const uint num_vertices = graph.numVertices();

  // Write graph header
  outfile << std::setw(curr_indent) << "";
  outfile << "<graph type=\"" << graph.typestr() << "\">" << std::endl;

  // Write vertices header
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "<vertices size=\"" << graph.numVertices() << "\">" << std::endl;

  // Write each vertex
  curr_indent = indentation_level + 4;
  for(uint i = 0; i < num_vertices; ++i)
  {
    outfile << std::setw(curr_indent) << "";
    outfile << "<vertex index=\"" << i << "\" num_edges=\"" << graph.numEdges(i) << "\" weight=\"" << vertex_weights[i] << "\"/>" << std::endl;
  }	  

  // Write vertices footer
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "</vertices>" << std::endl;

  // Write edges header
  outfile << std::setw(curr_indent) << "";
  outfile << "<edges size=\">" << graph.numEdges() << "\">" << std::endl;

  // Write each edge
  curr_indent = indentation_level + 4;
  for(uint i = 0; i < num_vertices; ++i)
  {
    for(uint j=offsets[i]; j<offsets[i] + graph.numEdges(i); ++j)
    {
      outfile << std::setw(curr_indent) << "";
      outfile << "<edge v1=\"" << i << "\" v2=\"" << connections[j] << "\" weight=\"" << vertex_weights[j] << "\"/>" << std::endl;
    }
  }	  

  // Write edges footer
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "</edges>" << std::endl;

  // Write graph footer 
  curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "</graph>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLGraph::read_graph(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parse_string(name, attrs, "type");
  
  // Open graph for editing
  editor.open(graph, type);
}
//-----------------------------------------------------------------------------
void NewXMLGraph::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("readVertices()");
  uint num_vertices = parse_uint(name, attrs, "size");
  editor.initVertices(num_vertices);
}
//-----------------------------------------------------------------------------
void NewXMLGraph::read_edges(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("readEdges()");
  uint num_edges = parse_uint(name, attrs, "size");
  editor.initEdges(num_edges);
}
//-----------------------------------------------------------------------------
void NewXMLGraph::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  editor.addVertex(parse_uint(name, attrs, "index"), parse_uint(name, attrs, "num_edges"));
}
//-----------------------------------------------------------------------------
void NewXMLGraph::read_edge(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");

  //dolfin_debug2("readEdge, v1 = %d, v2 = %d", v1, v2);
  
  // Edge weights not yet implemented
  //uint w = parse_uint(name, attrs, "weight");
  editor.addEdge(v1, v2);
}
//-----------------------------------------------------------------------------
void NewXMLGraph::close_graph()
{
  editor.close();
}
//-----------------------------------------------------------------------------
