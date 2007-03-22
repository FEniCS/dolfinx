// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-02-12
// Last changed: 2007-03-19

#include <dolfin/dolfin_log.h>
#include <dolfin/Graph.h>
#include <dolfin/GraphEditor.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GraphEditor::GraphEditor()
  : next_vertex(0), next_arch(0), graph(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GraphEditor::~GraphEditor()
{
  clear();
}
//-----------------------------------------------------------------------------
void GraphEditor::open(Graph& graph, Graph::Type type)
{
  // Clear old graph data
  graph.clear();
  clear();

  // Save graph and dimension
  this->graph = &graph;

  // Set graph type
  graph._type = type;
}
//-----------------------------------------------------------------------------
void GraphEditor::open(Graph& graph, std::string type)
{
  if ( type == "directed" )
    open(graph, Graph::directed);
  else if ( type == "undirected" )
    open(graph, Graph::undirected);
  else
    dolfin_error1("Unknown graph type \"%s\".", type.c_str());
}
//-----------------------------------------------------------------------------
void GraphEditor::initVertices(uint num_vertices)
{
  // Check if we are currently editing a graph
  if ( !graph )
    dolfin_error("No graph opened, unable to edit.");
  
  // Initialize graph data
  graph->num_vertices = num_vertices;
  graph->vertices = new uint[num_vertices];
  graph->vertex_weights = new uint[num_vertices];

  // Initialize vertex data
  for(uint i=0; i<num_vertices; ++i)
    graph->vertex_weights[i] = 1;
}
//-----------------------------------------------------------------------------
void GraphEditor::initEdges(uint num_edges)
{
  // Check if we are currently editing a graph
  if ( !graph )
    dolfin_error("No graph opened, unable to edit.");
  
  // Initialize graph data
  graph->num_edges = num_edges;
  graph->num_arches = num_edges;

  if ( graph->type() == Graph::undirected )
    graph->num_arches = 2 * num_edges;

  // Check if num_arches matches next_arch
  if ( next_arch != graph->num_arches )
    dolfin_error2("num_arches (%u) mismatch with sum of vertex edges (%u)",
                  graph->num_arches, next_arch);

  graph->edges = new uint[graph->num_arches];
  graph->edge_weights = new uint[graph->num_arches];
  
  // Initialize arrays
  for(uint i=0; i<graph->num_arches; ++i)
  {
    graph->edges[i] = graph->num_vertices;
    graph->edge_weights[i] = 1;
  }
}
//-----------------------------------------------------------------------------
void GraphEditor::addVertex(uint u, uint num_edges)
{
  // Check if we are currently editing a graph
  if ( !graph )
    dolfin_error("No graph opened, unable to edit.");

  // Check value of vertex index
  if ( u >= graph->num_vertices )
    dolfin_error2("Vertex index (%d) out of range [0, %d].",
		  u, graph->num_vertices - 1);

  // Check if vertex added in correct order
  if ( u != next_vertex )
    dolfin_error1("Next vertex that can be added is %d.", next_vertex);
  
  // Set offset and step to next vertex
  dolfin_debug2("addVertex(%d, %d)", u, num_edges);
  graph->vertices[next_vertex++] = next_arch;
  next_arch += num_edges;
}
//-----------------------------------------------------------------------------
void GraphEditor::addEdge(uint u, uint v)
{
  dolfin_debug2("addEdge(%d, %d)", u, v);
  
  // Check value of to vertex index
  if ( v > next_vertex )
    dolfin_error1("Cannot create edge to undefined vertex (%d).", v);

  addArch(u, v);
  if ( graph->type() == Graph::undirected )
    addArch(v, u);
}
//-----------------------------------------------------------------------------
void GraphEditor::addArch(uint u, uint v)
{
  dolfin_debug2("addArch(%d, %d)", u, v);

  // Check if we are currently editing a graph
  if ( !graph )
    dolfin_error("No graph opened, unable to edit.");

  // Check value of from vertex index
  if ( u > next_vertex )
    dolfin_error1("Cannot create edge from undefined vertex (%d).", u);

  // Loop edges not allowed
  if ( u == v )
    dolfin_error1("Cannot create edge from vertex %d to itself.", v);

  // Check that vertex u is correctly specified
  if ( graph->vertices[u] < 0 || graph->vertices[u] > graph->num_arches )
    dolfin_error1("Vertice \"%u\" undefined or incorrectly defined.\n", u);

  uint u_next = graph->vertices[u];

  // if from vertex is last vertex stop at num_edges
  uint stop = (u == graph->num_vertices - 1) ? 
              graph->num_arches : graph->vertices[u+1];

  while(u_next < stop && graph->edges[u_next] != graph->num_vertices)
  {
    u_next++;
  }

  // Check if vertex has room for edge
  if ( u_next == stop || u_next == graph->vertices[u+1] )
    dolfin_error1("Vertex %d does not have room for more edges.", u);

  graph->edges[u_next] = v;
}
//-----------------------------------------------------------------------------
void GraphEditor::close()
{
  for(uint i=0; i<graph->num_vertices; ++i)
  {
    uint stop = graph->vertices[i] + graph->numEdges(i);
    for(uint j=graph->vertices[i]; j<stop; ++j)
    {
      if ( graph->edges[j] == graph->num_vertices )
      {
        dolfin_error1("Cannot close, vertex %u has undefined edges\n", i);
      }
    }
  }
  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void GraphEditor::clear()
{
  next_vertex = 0;
  next_arch = 0;
  graph = 0;
}
//-----------------------------------------------------------------------------
