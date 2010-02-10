// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2009.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-02-12
// Last changed: 2009-08-11

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/io/File.h>
#include "GraphBuilder.h"
#include "GraphPartition.h"
#include "Graph.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Graph::Graph()
  : Variable("graph", "DOLFIN graph"),
    _num_edges(0), _num_vertices(0),
    _edges(0), _vertices(0),
    _edge_weights(0), _vertex_weights(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Graph::Graph(Mesh& mesh, Representation rep)
  : Variable("graph", "DOLFIN graph"),
    _num_edges(0), _num_vertices(0),
    _edges(0), _vertices(0),
    _edge_weights(0), _vertex_weights(0)
{
  // Build graph
  //GraphBuilder::build(*this, mesh, rep);
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
void Graph::init(uint num_vertices, uint num_edges)
{
  clear();

  _num_vertices = num_vertices;
  _num_edges    = num_edges;
  _edges        = new uint[num_edges];
  _vertices     = new uint[num_vertices + 1];

  // FIXME: is this needed?
  _vertices[num_vertices] = num_edges;
}
//-----------------------------------------------------------------------------
bool Graph::adjacent(uint u, uint v)
{
  for(uint i = _vertices[u]; i < _vertices[u+1]; ++i)
  {
    if (_edges[i] == v)
      return true;
  }
  return false;
}
//-----------------------------------------------------------------------------
std::string Graph::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "Graph type: " << typestr() << std::endl;
    s << "Number of vertices = " << _num_vertices << std::endl;
    s << "Number of edges = " << _num_edges << std::endl;
    s << "Connectivity" << std::endl;
    s << "Vertex: Edges" << std::endl;

    for (uint i = 0; i < _num_vertices - 1; ++i)
    {
      s << i << ": ";
      for(uint j = _vertices[i]; j < _vertices[i+1]; ++j)
        s << _edges[j] << " ";
      s << std::endl;
    }
    s << _num_vertices - 1 << ": ";
    for (uint i = _vertices[_num_vertices - 1]; i < _num_edges; ++i)
      s << _edges[i] << " ";
    s << std::endl;
  }
  else
  {
    s << "<Graph with " << _num_vertices << " vertices  and " << _num_edges << " edges>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void Graph::partition(uint num_part, uint* vtx_part)
{
  GraphPartition::partition(*this, num_part, vtx_part);
}
//-----------------------------------------------------------------------------
std::string Graph::typestr() const
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
  delete [] _edges;
  delete [] _vertices;
  delete [] _edge_weights;
  delete [] _vertex_weights;
}
//-----------------------------------------------------------------------------
