// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2009.
// Modified by Martin Alnes, 2008.
// Modified by Johan Hake, 2010.
//
// First added:  2005-10-23
// Last changed: 2010-03-04

#include <iostream>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/log.h>
#include "NonlinearProblem.h"
#include "NewtonSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters NewtonSolver::default_parameters()
{
  Parameters p("newton_solver");

  p.add("maximum_iterations",    50);
  p.add("relative_tolerance",    1e-9);
  p.add("absolute_tolerance",    1e-10);
  p.add("convergence_criterion", "residual");
  p.add("method",                "full");
  p.add("report",                true);

  return p;
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(std::string solver_type, std::string pc_type)
             : solver(new LinearSolver(solver_type, pc_type)),
               A(new Matrix), dx(new Vector), b(new Vector)
{
  // Set default parameters
  parameters = default_parameters();
  parameters.add(solver->parameters);
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(GenericLinearSolver& solver, LinearAlgebraFactory& factory)
            : solver(reference_to_no_delete_pointer(solver)), A(factory.create_matrix()),
              dx(factory.create_vector()), b(factory.create_vector())
{
  // Set default parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, bool> NewtonSolver::solve(NonlinearProblem& nonlinear_problem,
                                                  GenericVector& x)
{
  assert(A);
  assert(b);
  assert(dx);

  const uint maxiter = parameters["maximum_iterations"];

  begin("Starting Newton solve.");

  uint krylov_iterations = 0;
  newton_iteration = 0;
  bool newton_converged = false;

  // Compute F(u)
  nonlinear_problem.form(*A, *b, x);
  nonlinear_problem.F(*b, x);

  // Start iterations
  while (!newton_converged && newton_iteration < maxiter)
  {
    // Compute Jacobian
    nonlinear_problem.J(*A, x);

    // Perform linear solve and update total number of Krylov iterations
    if (dx->size() > 0)
      dx->zero();
    krylov_iterations += solver->solve(*A, *dx, *b);

    // Compute initial residual
    if (newton_iteration == 0)
      newton_converged = converged(*b, *dx, nonlinear_problem);

    // Update solution
    x -= (*dx);

    // Update number of iterations
    ++newton_iteration;

    //FIXME: this step is not needed if residual is based on dx and this has converged.
    // Compute F
    nonlinear_problem.form(*A, *b, x);
    nonlinear_problem.F(*b, x);

    // Test for convergence
    newton_converged = converged(*b, *dx, nonlinear_problem);
  }

  if (newton_converged)
    info(PROGRESS, "Newton solver finished in %d iterations and %d linear solver iterations.",
            newton_iteration, krylov_iterations);
  else
    warning("Newton solver did not converge.");

  end();

  return std::make_pair(newton_iteration, newton_converged);
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::iteration() const
{
  return newton_iteration;
}
//-----------------------------------------------------------------------------
bool NewtonSolver::converged(const GenericVector& b, const GenericVector& dx,
                             const NonlinearProblem& nonlinear_problem)
{
  const std::string convergence_criterion = parameters["convergence_criterion"];
  const double rtol = parameters["relative_tolerance"];
  const double atol = parameters["absolute_tolerance"];
  const bool report = parameters["report"];

  double residual = 1.0;

  // Compute resdiual
  if (convergence_criterion == "residual")
    residual = b.norm("l2");
  else if (convergence_criterion == "incremental")
    residual = dx.norm("l2");
  else
    error("Unknown Newton convergence criterion");

  // If this is the first iteration step, set initial residual
  if (newton_iteration == 0)
    residual0 = residual;

  // Relative residual
  double relative_residual = residual / residual0;

  // Output iteration number and residual
  //FIXME: allow precision to be set for dolfin::cout<<
  /*
    std::cout.precision(3);
    if (report && newton_iteration > 0)
    std::cout << "  Iteration " << newton_iteration
              << ":"
              << " r (abs) = " << std::scientific << residual
              << " (tol = " << std::scientific << atol << ")"
              << " r (rel) = " << std::scientific << relative_residual
              << " (tol = " << std::scientific << rtol << ")"
              << std::endl;
  */
  if (report && newton_iteration > 0)
    info("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)",
         newton_iteration, residual, atol, relative_residual, rtol);

  // Return true of convergence criterion is met
  if (relative_residual < rtol || residual < atol)
    return true;

  // Otherwise return false
  return false;
}
//-----------------------------------------------------------------------------
