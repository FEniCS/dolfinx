// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("output", "plain text");

  // Load grid
  Grid grid("grid.xml.gz");
  
  // Save original grid in OpenDX format
  File dx_unref("grid_unrefined.dx");
  dx_unref << grid;

  // Mark nodes for refinement
  for (CellIterator cell(grid); !cell.end(); ++cell)
    if ( cell->midpoint().dist(0.0, 0.0, 0.0) < 0.3 )
      cell->mark();
  
  // Refine grid
  grid.refine();

  // Save refined grid in OpenDX format
  File dx_ref("grid_refined.dx");
  dx_ref << grid;

  return 0;
}
