// Copyright (C) 2005 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2006-05-07

#ifndef __STOKES_SOLVER_H
#define __STOKES_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{

  /// This class implements a solver for Stokes equations, using
  /// a mixed formulation (Taylor-Hood elements).
  ///
  /// FIXME: Make dimension-independent (currently 2D)

  class StokesSolver : public Solver
  {
  public:
    
    // Create Stokes solver
    StokesSolver(Mesh& mesh, Function& f, BoundaryCondition& bc);
    
    // Solve Stokes equations
    void solve();

    // Solve Stokes equations (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc);
  
    // Temporary for testing
    void checkError(Mesh& mesh, Function& u);

  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc;

  };

}

#endif
