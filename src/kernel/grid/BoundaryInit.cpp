// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
#include <dolfin/BoundaryInit.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryInit::init(Grid& grid)
{
  // It is important that the computation of boundary data is done in
  // the correct order: First compute faces, then use this information
  // to compute edges, and finally compute the nodes from the edges.

  // Write a message
  dolfin_start("Computing boundary:");

  clear(grid);

  initFaces(grid);
  initEdges(grid);
  initNodes(grid);
  
  dolfin_end();
}
//-----------------------------------------------------------------------------
void BoundaryInit::clear(Grid& grid)
{
  grid.bd->clear();
}
//----------------------------------------------------------------------------- 
void BoundaryInit::initFaces(Grid& grid)
{
  switch ( grid.type() ) {
  case Grid::triangles:
    initFacesTri(grid);
    break;
  case Grid::tetrahedrons:
    initFacesTet(grid);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdges(Grid& grid)
{
  switch ( grid.type() ) {
  case Grid::triangles:
    initEdgesTri(grid);
    break;
  case Grid::tetrahedrons:
    initEdgesTet(grid);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
void BoundaryInit::initNodes(Grid& grid)
{
  // Compute nodes from edges. A node is on the boundary if it belongs to
  // an edge which is on the boundary. We need an extra list so that we don't
  // add a node more than once.

  Array<bool> marker(grid.noNodes());
  marker = false;

  // Mark all nodes which are on the boundary
  for (List<Edge*>::Iterator e(grid.bd->edges); !e.end(); ++e) {
    marker((*e)->node(0)->id()) = true;
    marker((*e)->node(1)->id()) = true;
  }

  // Add all nodes on the boundary
  for (NodeIterator n(grid); !n.end(); ++n)
    if ( marker(n->id()) )
      grid.bd->add(n);
  
  // Write a message
  cout << "Found " << grid.bd->noNodes() << " nodes on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initFacesTri(Grid& grid)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryInit::initFacesTet(Grid& grid)
{
  // Go through all faces and for each face check if it is on the
  // boundary. A face is on the boundary if it is contained in only
  // one cell. A list is used to count the number of cell neighbors
  // for all faces. Warning: may not work if some faces have been
  // removed

  Array<int> cellcount(grid.noFaces());
  cellcount = 0;

  // Count the number of cell neighbors for each face
  for (CellIterator c(grid); !c.end(); ++c) 
    for (FaceIterator f(c); !f.end(); ++f)
      cellcount(f->id()) += 1;
  
  // Add faces with only one cell neighbor to the boundary
  for (FaceIterator f(grid); !f.end(); ++f) {
    cout << "Face " << f->id() << " has " << cellcount(f->id()) << " cell neighbors" << endl;

    if ( cellcount(f->id()) == 1 )
      grid.bd->add(f);
    else if ( cellcount(f->id()) != 2 )
      dolfin_error1("Inconsistent grid. Found face with %d cell neighbors.", cellcount(f->id()));
  }
  
  // Check that we found a boundary
  if ( grid.bd->noFaces() == 0 )
    dolfin_error("Found no faces on the boundary.");
  
  // Write a message
  cout << "Found " << grid.bd->noFaces() << " faces on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdgesTri(Grid& grid)
{
  // Go through all edges and for each edge check if it is on the
  // boundary. This is similar to what is done in initFacesTet() for
  // tetrahedrons. An edge is on the boundary if it is contained in
  // only one cell. A list is used to count the number of cell
  // neighbors for all edges.  Warning: may not work if some edges
  // have been removed

  Array<int> cellcount(grid.noEdges());
  cellcount = 0;

  // Count the number of cell neighbors for each edge
  for (CellIterator c(grid); !c.end(); ++c)
    for (EdgeIterator e(c); !e.end(); ++e)
      cellcount(e->id()) += 1;
  
  // Add edges with only one cell neighbor to the boundary
  for (EdgeIterator e(grid); !e.end(); ++e) {
    if ( cellcount(e->id()) == 1 )
      grid.bd->add(e);
    else if ( cellcount(e->id()) != 2 )
      dolfin_error1("Inconsistent grid. Found edge with %d cell neighbors.", cellcount(e->id()));
  }
  
  // Check that we found a boundary
  if ( grid.bd->noEdges() == 0 )
    dolfin_error("Found no edges on the boundary.");
  
  // Write a message
  cout << "Found " << grid.bd->noEdges() << " edges on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdgesTet(Grid& grid)
{
  // Compute edges from faces. An edge is on the boundary if it belongs to
  // a face which is on the boundary. We need an extra list so that we don't
  // add an edge more than once.

  Array<bool> marker(grid.noEdges());
  marker = false;

  // Mark all edges which are on the boundary
  for (List<Face*>::Iterator f(grid.bd->faces); !f.end(); ++f)
    for (EdgeIterator e(**f); !e.end(); ++e)
      marker(e->id()) = true;

  // Add all edges on the boundary
  for (EdgeIterator e(grid); !e.end(); ++e)
    if ( marker(e->id()) )
      grid.bd->add(e);

  // Write a message
  cout << "Found " << grid.bd->noEdges() << " edges on the boundary." << endl;
}
//-----------------------------------------------------------------------------
