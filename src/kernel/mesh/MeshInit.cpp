// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/Node.h>
#include <dolfin/Edge.h>
#include <dolfin/GenericCell.h>
#include <dolfin/MeshInit.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshInit::init(Mesh& mesh)
{
  // Write a message
  dolfin_start("Computing mesh connectivity:");
  cout << "Found " << mesh.noNodes() << " nodes." << endl;
  cout << "Found " << mesh.noCells() << " cells." << endl;
  
  // Reset all previous connections
  clear(mesh);

  // Compute connectivity
  initConnectivity(mesh);

  // Renumber all objects
  renumber(mesh);

  dolfin_end();
}
//-----------------------------------------------------------------------------
void MeshInit::renumber(Mesh& mesh)
{
  // Renumber nodes
  int i = 0;
  for (NodeIterator n(mesh); !n.end(); ++n)
    n->setID(i++, mesh);

  // Renumber cells
  i = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
    c->setID(i++, mesh);

  // Renumber edges
  i = 0;
  for (EdgeIterator e(mesh); !e.end(); ++e)
    e->setID(i++, mesh);

  // Renumber faces
  i = 0;
  for (FaceIterator f(mesh); !f.end(); ++f)
    f->setID(i++, mesh);
}
//-----------------------------------------------------------------------------
void MeshInit::clear(Mesh& mesh)
{  
  // Clear edges
  mesh.gd->edges.clear();

  // Clear faces
  mesh.gd->faces.clear();

  // Clear boundary data for mesh
  mesh.bd->clear();

  // Clear connectivity for nodes
  for (NodeIterator n(mesh); !n.end(); ++n) {
    n->nn.clear();
    n->nc.clear();
    n->ne.clear();
  }
  
  // Clear connectivity for cells (c-n is not touched)
  for (CellIterator c(mesh); !c.end(); ++c) {
    c->c->cc.clear();
    c->c->ce.clear();
    c->c->cf.clear();
  }
}
//-----------------------------------------------------------------------------
void MeshInit::initConnectivity(Mesh& mesh)
{
  // The data needs to be computed in the correct order, see MeshData.h.

  // Compute n-c connections [1]
  initNodeCell(mesh);
  
  // Compute c-c connections [2]
  initCellCell(mesh);

  // Compute edges [3]
  initEdges(mesh);
  
  // Compute n-e connections [4]
  initNodeEdge(mesh);
  
  // Compute n-n connections [5]
  initNodeNode(mesh);

  // Compute faces [6]
  initFaces(mesh);
}
//-----------------------------------------------------------------------------
void MeshInit::initEdges(Mesh& mesh)
{
  // Go through all cells and create edges. Each edge checks with its
  // cell neighbors to see if an edge has already been added.

  cout << "Creating edges" << endl;
  
  for (CellIterator c(mesh); !c.end(); ++c)
    c->createEdges();
  
  // Write a message
  cout << "Created " << mesh.noEdges() << " edges." << endl;
}
//-----------------------------------------------------------------------------
void MeshInit::initFaces(Mesh& mesh)
{
  // Go through all cells and add new faces

  // Go through all cells for a mesh of tetrahedrons
  for (CellIterator c(mesh); !c.end(); ++c)
    c->createFaces();

  // Write a message
  cout << "Created " << mesh.noFaces() << " faces." << endl;
}
//-----------------------------------------------------------------------------
void MeshInit::initNodeCell(Mesh& mesh)
{
  // Go through all cells and add eachcell as a neighbour to all its nodes.
  // This is done in three steps:
  //
  //   1. Count the number of cell neighbors for each node
  //   2. Allocate memory for each node
  //   3. Add the cell neighbors
  
  // Count the number of cells the node appears in
  for (CellIterator c(mesh); !c.end(); ++c)
    for (NodeIterator n(c); !n.end(); ++n)
      n->nc.setsize(n->nc.size()+1);
  
  // Allocate memory for the cell lists
  for (NodeIterator n(mesh); !n.end(); ++n)
    n->nc.init();

  // Add the cells to the cell lists
  for (CellIterator c(mesh); !c.end(); ++c)
    for (NodeIterator n(c); !n.end(); ++n)
      n->nc.add(c);
}
//-----------------------------------------------------------------------------
void MeshInit::initCellCell(Mesh& mesh)
{
  // Go through all cells and count the cell neighbors.
  // This is done in four steps:
  //
  //   1. Count the number of cell neighbors for each cell
  //   2. Allocate memory for each cell (overestimate)
  //   3. Add the cell neighbors
  //   4. Reallocate

  for (CellIterator c1(mesh); !c1.end(); ++c1) {

    // Count the number of cell neighbors (overestimate)
    for (NodeIterator n(c1); !n.end(); ++n)
      for (CellIterator c2(n); !c2.end(); ++c2)
	if ( c1->neighbor(*c2) )
	  c1->c->cc.setsize(c1->c->cc.size()+1);

    // Allocate memory
    c1->c->cc.init();
    
    // Add all *unique* cell neighbors
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
void MeshInit::initNodeEdge(Mesh& mesh)
{
  // Go through all edges and add each edge as a neighbour to all its nodes.
  // This is done in three steps:
  //
  //   1. Count the number of edge neighbors for each node
  //   2. Allocate memory for each node
  //   3. Add the edge neighbors

  // Count the number of edges the node appears in
  for (EdgeIterator e(mesh); !e.end(); ++e) {
    Node& n0 = e->node(0);
    Node& n1 = e->node(1);
    n0.ne.setsize(n0.ne.size() + 1);
    n1.ne.setsize(n1.ne.size() + 1);
  }
  
  // Allocate memory for the edge lists
  for (NodeIterator n(mesh); !n.end(); ++n)
    n->ne.init();
  
  // Add the edges to the edge lists
  for (EdgeIterator e(mesh); !e.end(); ++e) {
    Node& n0 = e->node(0);
    Node& n1 = e->node(1);
    n0.ne.add(e);
    n1.ne.add(e);
  }
}
//-----------------------------------------------------------------------------
void MeshInit::initNodeNode(Mesh& mesh)
{
  // Go through all nodes and compute the node neighbors of each node
  // from the edge neighbors.

  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    // Allocate the list of nodes
    n->nn.init(1 + n->ne.size());

    // First add the node itself
    n->nn(0) = n;
    
    // Then add the other nodes
    for (int i = 0; i < n->ne.size(); i++)
    {
      Edge* e = n->ne(i);

      if ( &e->node(0) != n )
	n->nn(i+1) = &e->node(0);
      else
	n->nn(i+1) = &e->node(1);
    }
  }
}
//-----------------------------------------------------------------------------
