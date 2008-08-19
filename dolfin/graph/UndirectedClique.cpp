// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-19
// Last changed: 2007-03-21

#include <dolfin/log/dolfin_log.h>
#include "GraphEditor.h"
#include "Graph.h"
#include "UndirectedClique.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UndirectedClique::UndirectedClique(uint num_vertices) : Graph()
{
/*
  if ( num_vertices < 1 )
    error("Graph must have at least one vertex.");

  rename("graph", "Undirected clique");

  // Open graph for editing
  GraphEditor editor;
  editor.open(*this, Graph::undirected);

  // Create vertices
  editor.initVertices(num_vertices);
  for (uint i = 0; i < num_vertices; ++i)
  {
    editor.addVertex(i, num_vertices - 1);
  }

  // Create edges
  editor.initEdges(((num_vertices - 1) * num_vertices)/2);
  for (uint i = 0; i < num_vertices - 1; ++i)
  {
    for (uint j = i+1; j < num_vertices; ++j)
    {
      editor.addEdge(i, j);
    }
  }

  // Close graph editor
  editor.close();
*/
}
//-----------------------------------------------------------------------------
