// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_SOLVER_H
#define __ELASTICITY_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class ElasticitySolver : public Solver {
  public:
    
    ElasticitySolver(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
