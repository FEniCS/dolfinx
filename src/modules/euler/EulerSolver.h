// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.

#ifndef __EULER_SOLVER_H
#define __EULER_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin {

  class EulerSolver : public Solver 
  {
  
  public:

    EulerSolver(Mesh& mesh);

    const char* description();

    void solve();

  };

}

#endif
