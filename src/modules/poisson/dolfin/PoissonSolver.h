// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Johan Jansson, 2005.
// Modified by Anders Logg, 2005.

#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <dolfin/NewSolver.h>

namespace dolfin
{

  class PoissonSolver : public NewSolver
  {
  public:
    
    // Create Poisson solver
    PoissonSolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
    
    // Solve Poisson's equation
    void solve();

    // Solve Poisson's equation (static version)
    static void solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    NewFunction& f;
    NewBoundaryCondition& bc;

  };

}

#endif
