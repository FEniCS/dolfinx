// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2006-02-24

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/NonlinearProblem.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/Parametrized.h>
#include <dolfin/KrylovSolver.h>

namespace dolfin
{
  class BoundaryCondition;
  class LinearForm;
  class Matrix;
  class Mesh;
  class Vector;

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver : public Parametrized
  {
  public:

    /// Initialise nonlinear solver
    NewtonSolver();

    /// Destructor
    ~NewtonSolver();

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and 
    /// Jacobian dF/dx
    uint solve(NonlinearProblem& nonlinear_function, Vector& x);

    /// Return Newton iteration number
    uint getIteration() const;

  private:

    // Current number of Newton iterations
    uint newton_iteration;

    // Residuals
    real residual, relative_residual;

    // Solver
    KrylovSolver solver;

    // Jacobian matrix
    Matrix A;

    // Resdiual vector
    Vector b;

    // Solution vector
    Vector dx;

  };

}

#endif
