// Copyright (C) 2004 Johan Hoffman.
// Licensed under the GNU GPL Version 2.

#ifndef __NSE_SOLVER_H
#define __NSE_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class NSE : public Solver {
  public:
    
    NSE(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
