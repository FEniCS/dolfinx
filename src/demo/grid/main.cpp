// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

void refine2D(int refinements);
void refine3D(int refinements);

int main()
{
  dolfin_set("output", "plain text");

  int refinements = 3;

  // Refine 2D grid
  refine2D(refinements);

  // Refine 3D grid
  /* refine3D(refinements); */
  
  return 0;
}

void refine2D(int refinements)
{
  dolfin::cout << "Refining 2D grid" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load grid
  Grid grid("grid2D.xml.gz");
  
  // Save original grid in OpenDX format
  File unref("grid2D_unrefined.m");
  unref << grid;

  // Refine a couple of times
  for (int i = 0; i < refinements; i++) {
    
    // Mark nodes for refinement
    for (CellIterator cell(grid); !cell.end(); ++cell)
      if ( cell->midpoint().dist(0.0, 0.0) < 0.5 )
	cell->mark();
    
    // Refine grid
    grid.refine();

  }

  // Save refined grid in Matlab format
  File ref("grid2D_refined.m");
  ref << grid;

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
    
    // Mark nodes for refinement
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
