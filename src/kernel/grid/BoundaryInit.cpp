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

  clear(grid);

  initFaces(grid);
  initEdges(grid);
  initNodes(grid);
}
//-----------------------------------------------------------------------------
void BoundaryInit::clear(Grid& grid)
{
  grid.bd.clear();
}
//-----------------------------------------------------------------------------
void BoundaryInit::initFaces(Grid& grid)
{
  // Go through all faces and for each face check if it is on the boundary.
  // A face is on the boundary if it is contained in only one cell.
  
  // Skip this for a triangular grid
  if ( grid.type() == Grid::triangles )
    return;
  
  // A list is used to countthe number of cell neighbors for all faces
  // Warning: may not work if some faces have been removed
  Array<int> cellcount(grid.noFaces());
  cellcount = 0;

  // Count the number of cell neighbors for each face
  for (CellIterator c(grid); !c.end(); ++c)
    for (FaceIterator f(c); !f.end(); ++f)
      cellcount(f->id()) += 1;
  
  // Add faces with only one cell neighbor to the boundary
  for (FaceIterator f(grid); !f.end(); ++f)
    cout << "number of cells: " << cellcount(f->id()) << endl;
    
  

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
