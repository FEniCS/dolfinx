// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_CONVDIFF_HH
#define __SOLVER_CONVDIFF_HH

#include "Solver.hh"

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
