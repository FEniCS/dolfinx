// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_H
#define __SOLVER_H

#include <dolfin.h>

namespace dolfin {

  class Solver {
  public:
	 
	 Solver(Grid &grid_) : grid(grid_) {}
	 
	 /// Problem description
	 virtual const char* description() = 0;
	 
	 /// Solve problem
	 virtual void solve() = 0;
	 
  protected:
	 
	 Grid &grid;
	 
  };

}

#endif
