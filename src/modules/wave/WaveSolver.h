// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __WAVE_SOLVER_H
#define __WAVE_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  class WaveSolver : public Solver {
  public:
    
    WaveSolver(Mesh& grid);
    
    const char* description();
    void solve();

  };
  
}

#endif
