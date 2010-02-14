// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2009.
//
// First added:  2005-10-23
// Last changed: 2009-06-29

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <utility>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

//#include <dolfin/la/GenericLinearSolver.h>
//#include <dolfin/la/GenericMatrix.h>
//#include <dolfin/la/GenericVector.h>
//#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearSolver;
  class GenericLinearAlgebraFactory;
  class GenericMatrix;
  class GenericVector;
  class Mesh;
  class NonlinearProblem;

  /// This class defines a Newton solver for equations of the form F(u) = 0.

  class NewtonSolver : public Variable
  {
  public:

    /// Create nonlinear solver with default linear solver and default
    /// linear algebra backend
    NewtonSolver(std::string solver_type = "lu",
                 std::string pc_type = "default");

    /// Create nonlinear solver using provided linear solver and linear algebra
    /// backend determined by factory
    NewtonSolver(GenericLinearSolver& solver, LinearAlgebraFactory& factory);

    /// Destructor
    virtual ~NewtonSolver();

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and
    /// Jacobian dF/dx
    std::pair<uint, bool> solve(NonlinearProblem& nonlinear_function,
                                GenericVector& x);

    /// Return Newton iteration number
    uint iteration() const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Convergence test
    virtual bool converged(const GenericVector& b, const GenericVector& dx,
                           const NonlinearProblem& nonlinear_problem);

    /// Current number of Newton iterations
    uint newton_iteration;

    /// Residual
    double residual0;

    /// Solver
    boost::shared_ptr<GenericLinearSolver> solver;

    /// Jacobian matrix
    boost::scoped_ptr<GenericMatrix> A;

    /// Solution vector
    boost::scoped_ptr<GenericVector> dx;

    /// Resdiual vector
    boost::scoped_ptr<GenericVector> b;
  };

}

#endif

