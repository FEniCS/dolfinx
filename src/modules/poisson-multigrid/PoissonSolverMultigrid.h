// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_SOLVER_MULTIGRID_H
#define __POISSON_SOLVER_MULTIGRID_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class PoissonSolverMultigrid : public Solver {
  public:
    
    PoissonSolverMultigrid(Mesh& mesh);
    
    const char* description();
    void solve();
    
  };

}

#endif
