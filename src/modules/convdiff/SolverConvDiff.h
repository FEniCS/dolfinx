// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_CONVDIFF_H
#define __SOLVER_CONVDIFF_H

#include "Solver.h"

namespace dolfin{
  
  class SolverConvDiff : public Solver {
  public:
	 
	 SolverConvDiff(Grid *grid) : Solver(grid) {}
	 ~SolverConvDiff();
	 
	 const char *Description();
	 void solve();
	 
  };

}

#endif
