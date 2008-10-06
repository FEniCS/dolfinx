// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2005-10-23
// Last changed: 2008-06-29

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/parameter/Parametrized.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/la/enums_la.h>

namespace dolfin
{
  class GenericLinearSolver;
  class Mesh;
  class NonlinearProblem;

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver : public Parametrized
  {
  public:

    /// Create nonlinear solver with default linear solver and default 
    /// linear algebra backend
    NewtonSolver(dolfin::SolverType solver_type = lu, 
                 dolfin::PreconditionerType pc_type = default_pc);

    /// Create nonlinear solver using provided linear solver and linear algebra
    /// backend determined by factory 
    NewtonSolver(GenericLinearSolver& solver, LinearAlgebraFactory& factory);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and 
    /// Jacobian dF/dx
    uint solve(NonlinearProblem& nonlinear_function, GenericVector& x);

    /// Return Newton iteration number
    uint getIteration() const;

  private:

    /// Convergence test 
    virtual bool converged(const GenericVector& b, const GenericVector& dx, 
                           const NonlinearProblem& nonlinear_problem);

    /// Current number of Newton iterations
    uint newton_iteration;

    /// Residual
    double residual0;

    /// Solver
    GenericLinearSolver* solver;
    GenericLinearSolver* local_solver;

    /// Solver
    dolfin::PreconditionerType pc;
    //Preconditioner* local_pc;

    /// Jacobian matrix
    GenericMatrix* A;

    /// Solution vector
    GenericVector* dx;

    /// Resdiual vector
    GenericVector* b;
  };

}

#endif

