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
	 
	 void GlobalRegularRefinement();

	 void RegularRefinement(Cell* parent);

	 void RegularRefinementTetrahedron(Cell* parent);
	 void RegularRefinementTriangle(Cell* parent);
	 
	 void LocalIrregularRefinement(Cell *parent);

	 void IrrRef1(Cell *parent);
	 void IrrRef2(Cell *parent);
	 void IrrRef3(Cell *parent);
	 void IrrRef4(Cell *parent);

	 void refine();
	 
  private:

	 void IrrRef1(Cell* parent, ShortList<Edge*> marked_edges);
	 
	 Grid& grid;

  };

}

#endif
