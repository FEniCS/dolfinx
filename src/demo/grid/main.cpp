// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("output", "plain text");

  Grid grid;
  File in("grid.xml.gz");
  File out("grid_refined.dx");

  // Read grid from file
  in >> grid;
  
  dolfin::cout << "no nodes = " << grid.noNodes() << dolfin::endl;
  dolfin::cout << "no cells = " << grid.noCells() << dolfin::endl;

  // Mark nodes for refinement
  for (CellIterator cell(grid); !cell.end(); ++cell)
    if ( cell->midpoint().dist(0.0, 0.0, 0.0) < 0.3 )
      cell->mark();
  
  // Refine grid
  grid.refine();
  
  dolfin::cout << "no nodes = " << grid.noNodes() << dolfin::endl;
  dolfin::cout << "no cells = " << grid.noCells() << dolfin::endl;

  // Save refined grid to file
  out << grid;

  return 0;
}
