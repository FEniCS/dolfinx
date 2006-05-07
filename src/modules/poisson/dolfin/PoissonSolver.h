// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2005.
//
// First added:  2002
// Last changed: 2006-05-07

#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Solver.h>

namespace dolfin
{

  /// This class implements a solver for Poisson's equation.
  ///
  /// FIXME: Make dimension-independent (currently 2D)

  class PoissonSolver : public Solver
  {
  public:
    
    /// Create Poisson solver
    PoissonSolver(Mesh& mesh, Function& f, BoundaryCondition& bc);
    
    /// Solve Poisson's equation
    void solve();

    /// Solve Poisson's equation (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc;

  };

}

#endif

#endif
