// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-02-12
// Last changed: 2007-03-19

#include <dolfin/dolfin_log.h>
#include <dolfin.h>
#include <dolfin/File.h>
#include <dolfin/Graph.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Graph::Graph() : Variable("graph", "DOLFIN graph")
{
  vertices = 0;
  edges = 0;
  num_vertices = 0;
  num_edges = 0;
  num_arches = 0;
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
Graph::Graph(Mesh& mesh) : Variable("graph", "Graph")
{
  num_vertices = mesh.numVertices();
  num_edges = mesh.numEdges();
  num_arches = num_edges * 2;

  edges = new uint[num_arches];
  vertices = new uint[num_vertices];

  // Iterate over edges from all vertices
  uint i = 0, j = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    vertices[j++] = i;
    uint* entities = vertex->entities(0);
    for (uint k=0; k<vertex->numEntities(0); k++)
    {
      edges[i++] = entities[k];
    }
  }

}
//-----------------------------------------------------------------------------
Graph::~Graph()
{
  clear();
}
//-----------------------------------------------------------------------------
/*
const Graph& Graph::operator=(const Graph& graph)
{
  data = graph.data;
  rename(graph.name(), graph.label());
  return *this;
}
*/
//-----------------------------------------------------------------------------
unsigned int Graph::numEdges(uint u)
{
  if ( u == num_vertices - 1 )
    return num_arches - vertices[num_vertices - 1];
  else
    return vertices[u+1] - vertices[u];
}
//-----------------------------------------------------------------------------
void Graph::disp()
{
  std::cout << "Graph type: " << typestr() << std::endl;
  std::cout << "Number of vertices = " << num_vertices << std::endl;
  std::cout << "Number of edges = " << num_edges << std::endl;
  std::cout << "Connectivity" << std::endl;
  std::cout << "Vertex: Edges" << std::endl;
  for(uint i=0; i<num_vertices-1; ++i)
  {
    std::cout << i << ": ";
    for(uint j=vertices[i]; j<vertices[i+1]; ++j)
    {
      std::cout << edges[j] << " ";
    }

    std::cout << std::endl;
  }
  // last vertex
  std::cout << num_vertices-1 << ": ";
  for(uint i=vertices[num_vertices-1]; i<num_arches; ++i)
  {
    std::cout << edges[i] << " ";
  }
  std::cout << std::endl;
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
}
