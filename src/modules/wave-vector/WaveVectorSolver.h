// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __WAVEVECTOR_SOLVER_H
#define __WAVEVECTOR_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class WaveVectorSolver : public Solver {
  public:
    
    WaveVectorSolver(Mesh& mesh);
    
    const char* description();
    void solve();

  };
  
}

#endif
