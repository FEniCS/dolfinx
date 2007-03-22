// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-19
// Last changed: 2007-03-21

#include <dolfin/GraphEditor.h>
#include <dolfin/Graph.h>
#include <dolfin/DirectedClique.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DirectedClique::DirectedClique(uint num_vertices) : Graph()
{
  if ( num_vertices < 1 )
    dolfin_error("Graph must have at least one vertex.");

  rename("graph", "Directed clique");

  // Open graph for editing
  GraphEditor editor;
  editor.open(*this, Graph::directed);

  // Create vertices
  editor.initVertices(num_vertices);
  for (uint i = 0; i < num_vertices; ++i)
  {
    editor.addVertex(i, num_vertices - 1);
  }

  // Create edges
  editor.initEdges((num_vertices - 1) * num_vertices);
  for (uint i = 0; i < num_vertices; ++i)
  {
    for (uint j = 0; j < i; ++j)
    {
      editor.addEdge(i, j);
    }
    for (uint j = i+1; j < num_vertices; ++j)
    {
      editor.addEdge(i, j);
    }
  }

  // Close graph editor
  editor.close();
}
//-----------------------------------------------------------------------------
