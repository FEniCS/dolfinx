// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HEAT_SOLVER_H
#define __HEAT_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class HeatSolver : public Solver {
  public:
    
    HeatSolver(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
