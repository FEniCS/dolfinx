// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-02-12
// Last changed: 2007-03-21

#include <dolfin/dolfin_log.h>
#include <dolfin/XMLGraph.h>

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
void XMLGraph::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "graph") == 0 )
    {
      readGraph(name, attrs);
      state = INSIDE_GRAPH;
    }
    
    break;

  case INSIDE_GRAPH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      readVertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "edges") == 0 )
    {
      readEdges(name, attrs);
      state = INSIDE_EDGES;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertex") == 0 )
      readVertex(name, attrs);

    break;
    
  case INSIDE_EDGES:

    if ( xmlStrcasecmp(name, (xmlChar *) "edge") == 0 )
      readEdge(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLGraph::endElement(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_GRAPH:
    if ( xmlStrcasecmp(name, (xmlChar *) "graph") == 0 )
    {
      closeGraph();
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
  cout << "Reading graph from file " << filename << "." << endl;
}
//-----------------------------------------------------------------------------
bool XMLGraph::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLGraph::readGraph(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parseString(name, attrs, "type");
  
  // Open graph for editing
  editor.open(_graph, type);
}
//-----------------------------------------------------------------------------
void XMLGraph::readVertices(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("readVertices()");
  uint num_vertices = parseUnsignedInt(name, attrs, "size");
  editor.initVertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLGraph::readEdges(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_debug("readEdges()");
  uint num_edges = parseUnsignedInt(name, attrs, "size");
  editor.initEdges(num_edges);
}
//-----------------------------------------------------------------------------
void XMLGraph::readVertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  currentVertex = parseUnsignedInt(name, attrs, "index");

  // Read number of incident edges
  uint num_edges = parseUnsignedInt(name, attrs, "num_edges");
  
  // Vertex weights not yet implemented
  //uint w = parseUnsignedInt(name, attrs, "weight");
  
  editor.addVertex(currentVertex, num_edges);
}
//-----------------------------------------------------------------------------
void XMLGraph::readEdge(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");

  dolfin_debug2("readEdge, v1 = %d, v2 = %d", v1, v2);
  
  // Edge weights not yet implemented
  //uint w = parseUnsignedInt(name, attrs, "weight");
  editor.addEdge(v1, v2);
}
//-----------------------------------------------------------------------------
void XMLGraph::closeGraph()
{
  editor.close();
}
//-----------------------------------------------------------------------------
