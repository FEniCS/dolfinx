// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2005-10-23
// Last changed: 2007-05-15

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/parameter/Parametrized.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/KrylovMethod.h>
#include <dolfin/la/Preconditioner.h>

namespace dolfin
{
  class Mesh;
  class NonlinearProblem;

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver : public Parametrized
  {
  public:

    /// Initialise nonlinear solver and choose LU solver
    NewtonSolver();

/*
#ifdef HAS_PETSC
    /// Initialise nonlinear solver and choose matrix type which defines LU solver
    NewtonSolver(Matrix::Type matrix_type);
#endif
*/
    /// Initialise nonlinear solver and choose Krylov solver and preconditioner
    NewtonSolver(KrylovMethod method, Preconditioner pc);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and 
    /// Jacobian dF/dx
    uint solve(NonlinearProblem& nonlinear_function, Vector& x);

    /// Return Newton iteration number
    uint getIteration() const;

  private:

    /// Convergence test 
    virtual bool converged(const Vector& b, const Vector& dx, 
        const NonlinearProblem& nonlinear_problem);

    /// Current number of Newton iterations
    uint newton_iteration;

    /// Residual
    real residual0;

    /// Solver
    LinearSolver* solver;

    /// Jacobian matrix
    Matrix A;

    /// Resdiual vector
    Vector b;

    /// Solution vector
    Vector dx;
  };

}

#endif

