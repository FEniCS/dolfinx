// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __REFINE_GRID_H
#define __REFINE_GRID_H

namespace dolfin {

  class Grid;
  
  class RefineGrid {
  public:
	 
	 RefineGrid(Grid &grid_) : grid(grid_) {}
	 
	 void refine();
	 
  private:

	 Grid& grid;

  };

}

#endif
