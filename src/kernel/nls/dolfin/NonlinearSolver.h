// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#ifndef __NONLINEAR_SOLVER_H
#define __NONLINEAR_SOLVER_H

#include <petscsnes.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>

namespace dolfin
{

  /// This class defines the interface of nonlinear solvers for
  /// equations of the form F(x) = 0.
  
  class NonlinearSolver
  {
  public:

    /// Constructor
    NonlinearSolver();

    /// Destructor
    virtual ~NonlinearSolver();
  
    /// Initialize nonlinear solver
    void init(uint M, uint N);

    /// Solve nonlinear problem F(u) = 0
    void solve(Vector& x);

  private:

    // PETSc nonlinear solver pointer
    SNES snes;

    // Size of old system (need to reinitialize when changing)
    uint M;
    uint N;

  };

}

#endif
