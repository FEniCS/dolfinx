// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - Maybe not so many methods should be public?
//   - Why _create_edges and not create_edges?
//   - Return List<Cell *> from closeCell() won't work?
//   - Rename refineGrid(int level) and unrefineGrid(int level) to refine(int level) and unrefine(int level)?
//   - Should there be an unrefine()?

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

	 void globalRefinement();

	 void evaluateMarks(int grid_level);
	 void closeGrid(int grid_level);
	 List<Cell *> closeCell(Cell *parent);

	 void refineGrid(int grid_level);
	 void unrefineGrid(int grid_level);

	 void regularRefinement(Cell* parent);

	 void regularRefinementTetrahedron(Cell* parent);
	 void regularRefinementTriangle(Cell* parent);
	 
	 void localIrregularRefinement(Cell *parent);

	 void irregularRefinementBy1(Cell *parent);
	 void irregularRefinementBy2(Cell *parent);
	 void irregularRefinementBy3(Cell *parent);
	 void irregularRefinementBy4(Cell *parent);

	 void refine();
	 
  private:

	 Grid& grid;

	 bool _create_edges;
  };

}

#endif
