// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CONV_DIFF_SOLVER_H
#define __CONV_DIFF_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class ConvDiffSolver : public Solver {
  public:
    
    ConvDiffSolver(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
