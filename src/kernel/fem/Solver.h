// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_HH
#define __SOLVER_HH

namespace dolfin {

  class Grid;
  
  class Solver {
  public:
	 
	 Solver(Grid *grid);
	 ~Solver();
	 
	 /// Problem description
	 virtual const char *Description() = 0;
	 
	 /// Solve problem
	 virtual void solve() = 0;
	 
  protected:
	 
	 Grid *grid;
	 
	 int space_dimension;
	 int no_nodes;
	 
  };

}

#endif
