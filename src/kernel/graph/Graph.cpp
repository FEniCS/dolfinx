// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
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
    dolfin_error1("Unknown mesh representation \"%s\".", type.c_str());
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
//-----------------------------------------------------------------------------
void Graph::createNodal(Mesh& mesh)
{
  mesh.init();
  num_vertices = mesh.numVertices();
  num_edges = mesh.numEdges();
  num_arches = num_edges * 2;

  edges = new uint[num_arches];
  vertices = new uint[num_vertices];

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
      dolfin_debug1("Edge no %d", j);
      edges[j++] = neighbor->index();
    }
    */
  }
}
//-----------------------------------------------------------------------------
void Graph::createDual(Mesh& mesh)
{
  num_vertices = mesh.numCells();
  dolfin_debug1("createDual() with no cells %d", mesh.numCells());

  num_edges = mesh.numEdges();
  dolfin_debug1("createDual() with no edges %d", mesh.numEdges());
  num_arches = num_edges * 2;

  edges = new uint[num_arches];
  vertices = new uint[num_vertices];
  dolfin_debug("finished initing arrays and stuff");

  // Create dual graph. Iterate over neighbors from all cells
  uint i = 0, j = 0;
  //uint D = mesh.topology().dim();
  for (CellIterator c0(mesh); !c0.end(); ++c0)
  {
    dolfin_debug1("Cell no %d", i);
    vertices[i++] = j;
    //uint* facets_0 = c0->entities(D - 1);
    for (CellIterator c1(c0); !c1.end(); ++c1)
    {
      dolfin_debug1("Edge no %d", j);
      edges[j++] = c1->index();

    }


    /*
    for (uint k=0; k<c1->CellIter(mesh.topology().dim()); k++)
    {
      edges[i++] = entities[k];
    }
    */
  }
}
//-----------------------------------------------------------------------------
