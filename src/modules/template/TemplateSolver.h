// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#ifndef __TEMPLATE_SOLVER_H
#define __TEMPLATE_SOLVER_H

#include <Solver.h>

namespace dolfin {
  
  class TemplateSolver : public Solver {
  public:
	 
	 TemplateSolver(Grid &grid) : Solver(grid) {}	 
	 
	 const char *description();
	 void solve();
    
  };

}
  
#endif
