// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-02-12
// Last changed: 2008-08-11

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/io/File.h>
#include "GraphPartition.h"
#include "Graph.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Graph::Graph() : Variable("graph", "DOLFIN graph"), num_edges(0), 
                 num_arches(0), num_vertices(0), edges(0), vertices(0), 
                 edge_weights(0), vertex_weights(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Graph::Graph(const Graph& graph) : Variable("graph", "DOLFIN graph")
{
  *this = graph;
}
//-----------------------------------------------------------------------------
Graph::Graph(std::string filename) : Variable("graph", "DOLFIN graph")
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Graph::~Graph()
{
  clear();
}
//-----------------------------------------------------------------------------
void Graph::init(uint _num_vertices, uint _num_edges, uint _num_arches)
{
  clear();

  num_vertices = _num_vertices;
  num_edges    = _num_edges;
  num_arches   = _num_arches;
  edges    = new uint[num_arches];
  vertices = new uint[num_vertices + 1];
  vertices[num_vertices] = num_arches;
}
//-----------------------------------------------------------------------------
bool Graph::adjacent(uint u, uint v)
{
  for(uint i=vertices[u]; i<vertices[u+1]; ++i)
  {
	 if(edges[i] == v)
		return true;
  }
  return false;
}
//-----------------------------------------------------------------------------
void Graph::disp()
{
  cout << "Graph type: " << typestr() << endl;
  cout << "Number of vertices = " << num_vertices << endl;
  cout << "Number of edges = " << num_edges << endl;
  cout << "Connectivity" << endl;
  cout << "Vertex: Edges" << endl;
  for(uint i=0; i<num_vertices-1; ++i)
  {
    cout << i << ": ";
    for(uint j=vertices[i]; j<vertices[i+1]; ++j)
    {
      cout << edges[j] << " ";
    }

    cout << endl;
  }
  // last vertex
  cout << num_vertices-1 << ": ";
  for(uint i=vertices[num_vertices-1]; i<num_arches; ++i)
  {
    cout << edges[i] << " ";
  }
  cout << endl;
}
//-----------------------------------------------------------------------------
void Graph::partition(uint num_part, uint* vtx_part)
{
  GraphPartition::partition(*this, num_part, vtx_part);
}
//-----------------------------------------------------------------------------
std::string Graph::typestr()
{
  switch ( _type )
  {
  case directed:
    return "directed";
  case undirected:
    return "undirected";
  default:
    return "";
  }

  return "";
}
//-----------------------------------------------------------------------------
void Graph::clear()
{
  if ( edges )
    delete [] edges;
  if ( vertices )
    delete [] vertices;
  if ( edge_weights )
    delete [] edge_weights;
  if ( vertex_weights )
    delete [] vertex_weights;
}
//-----------------------------------------------------------------------------

