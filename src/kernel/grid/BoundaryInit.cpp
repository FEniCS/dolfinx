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
  grid.bd.clear();
}
//----------------------------------------------------------------------------- 
void BoundaryInit::initFaces(Grid& grid)
{
   // Faces are not computed for a triangular grid.
  if ( grid.type() == Grid::triangles )
    return;
 
  // Go through all faces and for each face check if it is on the boundary.
  // A face is on the boundary if it is contained in only one cell.
  // A list is used to countthe number of cell neighbors for all faces.
  // Warning: may not work if some faces have been removed

  Array<int> cellcount(grid.noFaces());
  cellcount = 0;

  // Count the number of cell neighbors for each face
  for (CellIterator c(grid); !c.end(); ++c)
    for (FaceIterator f(c); !f.end(); ++f)
      cellcount(f->id()) += 1;
  
  // Add faces with only one cell neighbor to the boundary
  for (FaceIterator f(grid); !f.end(); ++f) {
    if ( cellcount(f->id()) == 1 )
      grid.bd.add(f);
    else if ( cellcount(f->id()) != 2 )
      dolfin_error1("Inconsistent grid. Found face with %d cell neighbors.", cellcount(f->id()));
  }

  // Check that we found a boundary
  if ( grid.bd.noFaces() == 0 )
    dolfin_error("Found no faces on the boundary.");

  // Write a message
  cout << "Found " << grid.bd.noFaces() << " faces on the boundary." << endl;
}
//-----------------------------------------------------------------------------
void BoundaryInit::initEdges(Grid& grid)
{

  

}
//-----------------------------------------------------------------------------
void BoundaryInit::initNodes(Grid& grid)
{


}
//-----------------------------------------------------------------------------
