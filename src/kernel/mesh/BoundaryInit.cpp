// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryInit.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryInit::init(Mesh& mesh)
{
  // It is important that the computation of boundary data is done in
  // the correct order: First compute faces, then use this information
  // to compute edges, and finally compute the nodes from the edges.

  // Write a message
  dolfin_start("Computing boundary:");

  clear(mesh);

  initFaces(mesh);
  initEdges(mesh);
  initNodes(mesh);
  
  dolfin_end();
}
//-----------------------------------------------------------------------------
void BoundaryInit::clear(Mesh& mesh)
{
  mesh.bd->clear();
}
//----------------------------------------------------------------------------- 
void BoundaryInit::initFaces(Mesh& mesh)
{
  switch ( mesh.type() ) {
  case Mesh::triangles:
    initFacesTri(mesh);
    break;
  case Mesh::tetrahedrons:
    initFacesTet(mesh);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdges(Mesh& mesh)
{
  switch ( mesh.type() ) {
  case Mesh::triangles:
    initEdgesTri(mesh);
    break;
  case Mesh::tetrahedrons:
    initEdgesTet(mesh);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
void BoundaryInit::initNodes(Mesh& mesh)
{
  // Compute nodes from edges. A node is on the boundary if it belongs to
  // an edge which is on the boundary. We need an extra list so that we don't
  // add a node more than once.

  Array<bool> marker(mesh.noNodes());
  marker = false;

  // Mark all nodes which are on the boundary
  for (List<Edge*>::Iterator e(mesh.bd->edges); !e.end(); ++e) {
    marker((*e)->node(0).id()) = true;
    marker((*e)->node(1).id()) = true;
  }

  // Add all nodes on the boundary
  for (NodeIterator n(mesh); !n.end(); ++n)
    if ( marker(n->id()) )
      mesh.bd->add(*n);
  
  // Write a message
  cout << "Found " << mesh.bd->noNodes() << " nodes on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initFacesTri(Mesh& mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryInit::initFacesTet(Mesh& mesh)
{
  // Go through all faces and for each face check if it is on the
  // boundary. A face is on the boundary if it is contained in only
  // one cell. A list is used to count the number of cell neighbors
  // for all faces. Warning: may not work if some faces have been
  // removed

  Array<int> cellcount(mesh.noFaces());
  cellcount = 0;

  // Count the number of cell neighbors for each face
  for (CellIterator c(mesh); !c.end(); ++c) 
    for (FaceIterator f(c); !f.end(); ++f)
      cellcount(f->id()) += 1;
  
  // Add faces with only one cell neighbor to the boundary
  for (FaceIterator f(mesh); !f.end(); ++f) {
    if ( cellcount(f->id()) == 1 )
      mesh.bd->add(*f);
    else if ( cellcount(f->id()) != 2 )
      dolfin_error1("Inconsistent mesh. Found face with %d cell neighbors.", cellcount(f->id()));
  }
  
  // Check that we found a boundary
  if ( mesh.bd->noFaces() == 0 )
    dolfin_error("Found no faces on the boundary.");
  
  // Write a message
  cout << "Found " << mesh.bd->noFaces() << " faces on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdgesTri(Mesh& mesh)
{
  // Go through all edges and for each edge check if it is on the
  // boundary. This is similar to what is done in initFacesTet() for
  // tetrahedrons. An edge is on the boundary if it is contained in
  // only one cell. A list is used to count the number of cell
  // neighbors for all edges.  Warning: may not work if some edges
  // have been removed

  Array<int> cellcount(mesh.noEdges());
  cellcount = 0;

  // Count the number of cell neighbors for each edge
  for (CellIterator c(mesh); !c.end(); ++c)
    for (EdgeIterator e(c); !e.end(); ++e)
      cellcount(e->id()) += 1;
  
  // Add edges with only one cell neighbor to the boundary
  for (EdgeIterator e(mesh); !e.end(); ++e) {
    if ( cellcount(e->id()) == 1 )
      mesh.bd->add(*e);
    else if ( cellcount(e->id()) != 2 )
      dolfin_error1("Inconsistent mesh. Found edge with %d cell neighbors.", cellcount(e->id()));
  }
  
  // Check that we found a boundary
  if ( mesh.bd->noEdges() == 0 )
    dolfin_error("Found no edges on the boundary.");
  
  // Write a message
  cout << "Found " << mesh.bd->noEdges() << " edges on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdgesTet(Mesh& mesh)
{
  // Compute edges from faces. An edge is on the boundary if it belongs to
  // a face which is on the boundary. We need an extra list so that we don't
  // add an edge more than once.

  Array<bool> marker(mesh.noEdges());
  marker = false;

  // Mark all edges which are on the boundary
  for (List<Face*>::Iterator f(mesh.bd->faces); !f.end(); ++f)
    for (EdgeIterator e(**f); !e.end(); ++e)
      marker(e->id()) = true;

  // Add all edges on the boundary
  for (EdgeIterator e(mesh); !e.end(); ++e)
    if ( marker(e->id()) )
      mesh.bd->add(*e);

  // Write a message
  cout << "Found " << mesh.bd->noEdges() << " edges on the boundary." << endl;
}
//-----------------------------------------------------------------------------
