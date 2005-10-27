// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#ifndef __NONLINEAR_SOLVER_H
#define __NONLINEAR_SOLVER_H

#include <petscsnes.h>

#include <dolfin/NonlinearFunctional.h>
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

    /// Initialise nonlinear solver for a given nonlinear function
    NonlinearSolver(NonlinearFunctional& nlfunction);

    /// Destructor
    ~NonlinearSolver();
  
    /// Solve nonlinear problem F(u) = 0
    void solve(Vector& x, NonlinearFunctional& nlfunction);

    /// Form RHS vector F(u)
    static int FormRHS(SNES snes, Vec x, Vec f, void* ptr);

    /// Form Jacobian (stiffness matrix) F'(u) = dF(u)/du
    static int FormJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* ptr);

  private:

    /// Form RHS vector F(u) and  Jacobian F'(u) = dF(u)/du
    void FormSystem();

    // Details of the nonliner function
    NonlinearFunctional* nlfunc;

    // PETSc nonlinear solver pointer
    SNES snes;

  };

}

#endif
