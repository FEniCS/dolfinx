// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// InitGrid implements the algorithm for computing the neighbour
// information (connections) in a grid.
//
// The trick is to compute the connections in the correct order:
//
// 1. All neighbor cells of a node: n-c
// 2. All neighbor cells of a cell: c-c (including the cell itself)
// 3. All neighbor nodes of a node: n-n (including the node itself)
// 4. All neighbor edges of a node: n-e 

#ifndef __INIT_GRID_H
#define __INIT_GRID_H

namespace dolfin{

  class Grid;

  class InitGrid {
  public:

	 InitGrid(Grid& grid_) : grid(grid_) {}

	 void init();
	 
  private:

	 void clear();

	 void initNeighbors();
	 void initBoundary();
		 
	 void initNodeCell();
	 void initCellCell();
	 void initNodeNode();
	 void initNodeEdge();
 
	 Grid& grid;
	 
  };

}

#endif
