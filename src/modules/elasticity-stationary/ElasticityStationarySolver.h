// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_STATIONARY_SOLVER_H
#define __ELASTICITY_STATIONARY_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class ElasticityStationarySolver : public Solver {
  public:
    
    ElasticityStationarySolver(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
