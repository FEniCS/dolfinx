// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/Edge.h>
#include <dolfin/GenericCell.h>
#include <dolfin/GridInit.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void GridInit::init(Grid& grid)
{
  // Write a message
  dolfin_start("Computing grid connectivity:");
  cout << "Found " << grid.noNodes() << " nodes." << endl;
  cout << "Found " << grid.noCells() << " cells." << endl;
  
  // Reset all previous connections
  clear(grid);

  // Init edges (needs to be done before connectivity)
  initEdges(grid);

  // Init connectivity data
  initConnectivity(grid);

  // Init faces (needs to be done after connectivity)
  initFaces(grid);

  // Task is done
  dolfin_end();
}
//-----------------------------------------------------------------------------
void GridInit::clear(Grid& grid)
{  
  // Clear connectivity for nodes
  for (NodeIterator n(grid); !n.end(); ++n) {
    n->nn.clear();
    n->nc.clear();
    n->ne.clear();
  }
  
  // Clear connectivity for cells (c-n is not touched)
  for (CellIterator c(grid); !c.end(); ++c) {
    c->c->cc.clear();
    c->c->ce.clear();
    c->c->cf.clear();
  }
}
//-----------------------------------------------------------------------------
void GridInit::initEdges(Grid& grid)
{
  // Go through all cells and create edges. Each edge checks with its
  // cell neighbors to see if an edge has already been added.
  
  for (CellIterator c(grid); !c.end(); ++c)
    c->createEdges();
  
  // Write a message
  cout << "Created " << grid.noEdges() << " edges." << endl;
}
//-----------------------------------------------------------------------------
void GridInit::initConnectivity(Grid& grid)
{
  // Compute neighbor information
  //
  //   n-c (cell neighbors of node)
  //   c-c (cell neighbors of cell)
  //   n-e (edge neighbors of node)
  //   n-n (node neighbors of node)
  //
  // It is important that these are computed in the correct order.
  // Gives a simple and efficient algorithm that is O(n).
  
  // Compute n-c connections
  initNodeCell(grid);
  
  // Compute c-c connections
  initCellCell(grid);
  
  // Compute n-e connections
  initNodeEdge(grid);
  
  // Compute n-n connections
  initNodeNode(grid);
}
//-----------------------------------------------------------------------------
void GridInit::initFaces(Grid& grid)
{
  // Go through all cells and add new faces. This algorithm is O(n^2),
  // similarly to initEdges().

  // Go through all cells for a grid of tetrahedrons
  for (CellIterator c(grid); !c.end(); ++c)
    c->createFaces();

  // Write a message
  cout << "Created " << grid.noFaces() << " faces." << endl;
}
//-----------------------------------------------------------------------------
void GridInit::initNodeCell(Grid& grid)
{
  // Go through all cells and add eachcell as a neighbour to all its nodes.
  // This is done in three steps:
  //
  //   1. Count the number of cell neighbors for each node
  //   2. Allocate memory for each node
  //   3. Add the cell neighbors
  
  // Count the number of cells the node appears in
  for (CellIterator c(grid); !c.end(); ++c)
    for (NodeIterator n(c); !n.end(); ++n)
      n->nc.setsize(n->nc.size()+1);
  
  // Allocate memory for the cell lists
  for (NodeIterator n(grid); !n.end(); ++n)
    n->nc.init();
  
  // Add the cells to the cell lists
  for (CellIterator c(grid); !c.end(); ++c)
    for (NodeIterator n(c); !n.end(); ++n)
      n->nc.add(c);  
}
//-----------------------------------------------------------------------------
void GridInit::initCellCell(Grid& grid)
{
  // Go through all cells and count the cell neighbors.

  for (CellIterator c1(grid); !c1.end(); ++c1) {

    // Allocate for the maximum number of cell neighbors
    c1->c->cc.init(c1->noBoundaries() + 1);
    c1->c->cc.reset();
    
    // Add all unique cell neighbors
    for (NodeIterator n(c1); !n.end(); ++n)
      for (CellIterator c2(n); !c2.end(); ++c2)
	if ( c1->neighbor(*c2) )
	  if ( !c1->c->cc.contains(c2) )
	    c1->c->cc.add(c2);
    
    // Reallocate
    c1->c->cc.resize();
    
  }
}
//-----------------------------------------------------------------------------
void GridInit::initNodeEdge(Grid& grid)
{
  // Go through all edges and add each edge as a neighbour to all its nodes.
  // This is done in three steps:
  //
  //   1. Count the number of edge neighbors for each node
  //   2. Allocate memory for each node
  //   3. Add the edge neighbors

  // Count the number of edges the node appears in
  for (EdgeIterator e(grid); !e.end(); ++e) {
    Node* n0 = e->node(0);
    Node* n1 = e->node(1);
    n0->ne.setsize(n0->ne.size() + 1);
    n1->ne.setsize(n1->ne.size() + 1);
  }
  
  // Allocate memory for the edge lists
  for (NodeIterator n(grid); !n.end(); ++n)
    n->ne.init();
  
  // Add the edges to the edge lists
  for (EdgeIterator e(grid); !e.end(); ++e) {
    Node* n0 = e->node(0);
    Node* n1 = e->node(1);
    n0->ne.add(e);
    n1->ne.add(e);
  }
}
//-----------------------------------------------------------------------------
void GridInit::initNodeNode(Grid& grid)
{
  // Go through all nodes and compute the node neighbors of each node
  // from the edge neighbors.

  for (NodeIterator n(grid); !n.end(); ++n) {

    // Allocate the list of nodes
    n->nn.init(1 + n->ne.size());

    // First add the node itself
    n->nn(0) = n;

    // Then add the other nodes
    for (int i = 0; i < n->ne.size(); i++) {
      Edge* e = n->ne(i);
      if ( e->node(0) != n )
	n->nn(i+1) = n->node(0);
      else
	n->nn(i+1) = n->node(1);
    }

  }
}
//-----------------------------------------------------------------------------
