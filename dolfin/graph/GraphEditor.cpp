// Copyright (C) 2007 Magnus Vikstrom.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Anders Logg, 2009.
//
// First added:  2007-02-12
// Last changed: 2009-04-27

#include <dolfin/log/dolfin_log.h>
#include "Graph.h"
#include "GraphEditor.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GraphEditor::GraphEditor() : next_vertex(0), edge_count(0), graph(0)
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
  // Clear editor data
  clear();

  // Clear graph data
  graph.clear();

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
    error("Unknown graph type \"%s\".", type.c_str());
}
//-----------------------------------------------------------------------------
void GraphEditor::init_vertices(uint num_vertices)
{
  // Check if we are currently editing a graph
  if ( !graph )
    error("No graph opened, unable to edit.");

  // Initialize graph data
  graph->_num_vertices = num_vertices;
  if(!graph->_vertices && !graph->_vertex_weights)
  {
    graph->_vertices = new uint[num_vertices+1];
    graph->_vertex_weights = new uint[num_vertices];
  }
  else
    error("Graph vertex data has already been allocated.");

  // Initialize vertex weights
  for(uint i = 0; i < num_vertices; ++i)
    graph->_vertex_weights[i] = 1;
}
//-----------------------------------------------------------------------------
void GraphEditor::init_edges(uint num_edges)
{
  // Check if we are currently editing a graph
  if ( !graph )
    error("No graph opened, unable to edit.");

  // Check that vertex data has been inialised
  if( !graph->_vertices )
    error("Vertex data has not been initialised.");

  // Initialize graph data
  graph->_num_edges = num_edges;

  // Check if num_edges is consistent with edge_count
  if ( num_edges != edge_count )
    error("num_edges (%u) mismatch with sum of vertex edges (%u)", num_edges, edge_count);

  if(!graph->_edges && !graph->_edge_weights)
  {
    graph->_edges        = new uint[graph->_num_edges];
    graph->_edge_weights = new uint[graph->_num_edges];
  }
  else
    error("Graph edge data has already been allocated.");

  // Initialize arrays
  for(uint i=0; i < graph->_num_edges; ++i)
  {
    graph->_edges[i] = graph->_num_vertices;
    graph->_edge_weights[i] = 1;
  }

  // Update vertex array
  graph->_vertices[graph->num_vertices()] = graph->num_edges();
}
//-----------------------------------------------------------------------------
void GraphEditor::add_vertex(uint u, uint num_edges)
{
  // Check if we are currently editing a graph
  if ( !graph )
    error("No graph opened, unable to edit.");

  // Check value of vertex index
  if ( u >= graph->_num_vertices )
    error("Vertex index (%d) out of range [0, %d].", u, graph->_num_vertices - 1);

  // Check if vertex added in correct order
  if ( u != next_vertex )
    error("Next vertex that can be added is %d.", next_vertex);

  // Set offset and step to next vertex
  graph->_vertices[next_vertex++] = edge_count;

  // Keep track of total number of edges
  edge_count += num_edges;
}
//-----------------------------------------------------------------------------
void GraphEditor::add_edge(uint u, uint v)
{
  // Check value of to vertex index
  if ( v > next_vertex )
    error("Cannot create edge to undefined vertex (%d).", v);

  // Loop edges not allowed
  if ( u == v )
    error("Cannot create edge from vertex %d to itself.", v);

  // Check that vertex u is correctly specified
  if ( graph->_vertices[u] < 0 || graph->_vertices[u] > graph->_num_edges )
    error("Vertex \"%u\" undefined or incorrectly defined.", u);

  uint u_next = graph->_vertices[u];

  // If from vertex is last vertex stop at num_edges
  uint stop = (u == graph->_num_vertices - 1) ? graph->_num_edges : graph->_vertices[u+1];
  while(u_next < stop && graph->_edges[u_next] != graph->_num_vertices)
    u_next++;

  // Check if vertex has room for edge
  if ( u_next == stop || u_next == graph->_vertices[u+1] )
    error("Vertex %d does not have room for more edges.", u);

  graph->_edges[u_next] = v;
}
//-----------------------------------------------------------------------------
void GraphEditor::close()
{
  for(uint i = 0; i < graph->_num_vertices; ++i)
  {
    uint stop = graph->_vertices[i] + graph->num_edges(i);
    for(uint j = graph->_vertices[i]; j < stop; ++j)
    {
      if ( graph->_edges[j] == graph->_num_vertices )
        error("Cannot close, vertex %u has undefined edges", i);
    }
  }
  // Clear data
  clear();
}
//-----------------------------------------------------------------------------
void GraphEditor::clear()
{
  next_vertex = 0;
  edge_count = 0;
  graph = 0;
}
//-----------------------------------------------------------------------------

