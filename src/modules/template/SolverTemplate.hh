// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_TEMPLATE_HH
#define __SOLVER_TEMPLATE_HH

#include <Solver.hh>

namespace dolfin {
  
  class SolverTemplate : public Solver {
  public:
	 
	 SolverTemplate(Grid *grid) : Solver(grid) {}
	 ~SolverTemplate(){}
	 
	 const char *Description();
	 void solve();
    
  };

}
  
#endif
