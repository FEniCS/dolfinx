// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __REFINE_GRID_H
#define __REFINE_GRID_H

namespace dolfin {

  class Grid;
  class Edge;
  class Cell;
  
  class RefineGrid {
  public:
	 
	 RefineGrid(Grid &grid_) : grid(grid_) {}
	 
	 void globalRegularRefinement();

	 void evaluateMarks(int grid_level);
	 void closeGrid(int grid_level);
	 List<Cell *> closeCell(Cell *parent);

	 void refineGrid(int grid_level);
	 void unrefineGrid(int grid_level);

	 void regularRefinement(Cell* parent);

	 void regularRefinementTetrahedron(Cell* parent);
	 void regularRefinementTriangle(Cell* parent);
	 
	 void localIrregularRefinement(Cell *parent);

	 void irrRef1(Cell *parent);
	 void irrRef2(Cell *parent);
	 void irrRef3(Cell *parent);
	 void irrRef4(Cell *parent);

	 void refine();
	 
  private:

	 Grid& grid;

	 bool _create_edges;
  };

}

#endif
