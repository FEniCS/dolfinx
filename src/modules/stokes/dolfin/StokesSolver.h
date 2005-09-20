// Copyright (C) 2005 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-09-20

#ifndef __STOKES_SOLVER_H
#define __STOKES_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{

  class StokesSolver : public Solver
  {
  public:
    
    // Create Stokes solver
    StokesSolver(Mesh& mesh, Function& f, BoundaryCondition& bc);
    
    // Solve Stokes equations
    void solve();

    // Solve Stokes equations (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc;

  };

}

#endif
