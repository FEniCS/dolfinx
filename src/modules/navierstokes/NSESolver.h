// Copyright (C) 2004 Johan Hoffman.
// Licensed under the GNU GPL Version 2.

#ifndef __NSE_SOLVER_H
#define __NSE_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class NSESolver : public Solver {
  public:
    
    NSESolver(Mesh& mesh);
    
    const char* description();
    void Newsolve();
    void solve();

  };
  
}

#endif
