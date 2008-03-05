// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
//
// First added:  2007-02-12
// Last changed: 2007-05-14

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/io/File.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "GraphPartition.h"
#include "Graph.h"

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
  // Default mesh representation
  createDual(mesh);
}
//-----------------------------------------------------------------------------
Graph::Graph(Mesh& mesh, Representation type) : Variable("graph", "Graph")
{
  if ( type == nodal )
    createNodal(mesh);
  else
    createDual(mesh);
}
//-----------------------------------------------------------------------------
Graph::Graph(Mesh& mesh, std::string type) : Variable("graph", "Graph")
{
  if ( type == "nodal" )
    createNodal(mesh);
  else if ( type == "dual" )
    createDual(mesh);
  else
    error("Unknown mesh representation \"%s\".", type.c_str());
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
}
//-----------------------------------------------------------------------------
void Graph::createNodal(Mesh& mesh)
{
  mesh.init();
  num_vertices = mesh.numVertices();
  num_edges = mesh.numEdges();
  num_arches = num_edges * 2;

  edges = new uint[num_arches];
  vertices = new uint[num_vertices + 1];
  vertices[num_vertices] = num_arches;

  // Create nodal graph. Iterate over edges from all vertices
  uint i = 0, j = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    vertices[i++] = j;
    uint* entities = vertex->entities(0);
    for (uint k=0; k<vertex->numEntities(0); k++)
    {
      edges[j++] = entities[k];
    }
    // Replace with this?
    /*
    for (VertexIterator neighbor(vertex); !neighbor.end(); ++neighbor)
    {
      //dolfin_debug1("Edge no %d", j);
		dolfin_debug2("Vertex no %d connected to vertex no %d", vertex->index(), neighbor->index());
      edges[j++] = neighbor->index();
    }
    */
  }
}
//-----------------------------------------------------------------------------
void Graph::createDual(Mesh& mesh)
{
  num_vertices = mesh.numCells();

  // Get number of arches
  uint D = mesh.topology().dim();
  mesh.init(D,D);
  MeshConnectivity& connectivity = mesh.topology()(D,D);
  num_arches = connectivity.size();
  num_edges = num_arches/2;


  // This initialization should be a method
  edges = new uint[num_arches];
  vertices = new uint[num_vertices + 1];
  vertices[num_vertices] = num_arches;

  // Create dual graph. Iterate over neighbors from all cells
  uint i = 0, j = 0;
  for (CellIterator c0(mesh); !c0.end(); ++c0)
  {
    //dolfin_debug1("Cell no %d", i);
    //dolfin_debug1("Cell no %d", c0->index());
    vertices[i++] = j;

    for (CellIterator c1(*c0); !c1.end(); ++c1)
    {
      //dolfin_debug2("Cell no %d connected to cell no %d", c0->index(), c1->index());
      //dolfin_debug2("edges[%d] = %d", j, c1->index());
      edges[j++] = c1->index();

    }
  }
}
//-----------------------------------------------------------------------------
