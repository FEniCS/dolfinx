// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

void refine2D(int refinements);
void refine3D(int refinements);

int main()
{
  int refinements = 3;

  // Refine 2D grid
  refine2D(refinements);

  // Refine 3D grid
  refine3D(refinements);
  
  return 0;
}

void refine2D(int refinements)
{
  dolfin::cout << "Refining 2D grid" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load grid
  Grid grid("grid2D.xml.gz");
  
  // Save first grid
  File file("grids2D.m");
  file << grid;

  // Refine a couple of times
  for (int i = 0; i < refinements; i++) {
    

    for (CellIterator cell(grid); !cell.end(); ++cell) {
      
      // Mark cells close to y = x
      for (NodeIterator node(cell); !node.end(); ++node)
	if ( fabs(node->coord().x - node->coord().y) < DOLFIN_EPS )
	  cell->mark();
      
      // Mark cells at the corners
      if ( cell->midpoint().dist(0.0, 0.0) < 0.25 ||
	   cell->midpoint().dist(1.0, 0.0) < 0.25 ||
	   cell->midpoint().dist(1.0, 1.0) < 0.25 ||
	   cell->midpoint().dist(0.0, 1.0) < 0.25 )
	cell->mark();
      
    }

    // Refine grid
    grid.refine();

    // Save all grids in the grid hierarcy
    GridHierarchy grids(grid);
    for (GridIterator g(grids); !g.end(); ++g)
      file << *g;

  }

  dolfin::cout << dolfin::endl;
}

void refine3D(int refinements)
{
  dolfin::cout << "Refining 3D grid" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load grid
  Grid grid("grid3D.xml.gz");
  
  // Save original grid in OpenDX format
  File unref("grid3D_unrefined.dx");
  unref << grid;

  // Refine a couple of times
  for (int i = 0; i < refinements; i++) {
    
    // Mark cells close to (0,0,0)
    for (CellIterator cell(grid); !cell.end(); ++cell)
      if ( cell->midpoint().dist(0.0, 0.0, 0.0) < 0.3 )
	cell->mark();
    
    // Refine grid
    grid.refine();

  }

  // Save refined grid in OpenDX format
  File ref("grid3D_refined.dx");
  ref << grid;
}
