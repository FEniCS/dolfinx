// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_TEMPLATE_H
#define __SOLVER_TEMPLATE_H

#include <Solver.h>

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
