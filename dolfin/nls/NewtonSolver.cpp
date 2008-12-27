// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2008.
// Modified by Martin Alnes, 2008.
//
// First added:  2005-10-23
// Last changed: 2008-12-27

#include "NewtonSolver.h"
#include "NonlinearProblem.h"
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/la/LinearSolver.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(SolverType solver_type, PreconditionerType pc_type) 
             : solver(new LinearSolver(solver_type, pc_type)), 
               A(new Matrix), dx(new Vector), b(new Vector)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(GenericLinearSolver& solver, LinearAlgebraFactory& factory) 
            : solver(reference_to_no_delete_pointer(solver)), A(factory.create_matrix()), 
              dx(factory.create_vector()), b(factory.create_vector())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  delete A;
  delete dx;
  delete b;
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearProblem& nonlinear_problem, GenericVector& x)
{
  dolfin_assert(A);
  dolfin_assert(b);
  dolfin_assert(dx);

  const uint maxiter = get("Newton maximum iterations");

  begin("Starting Newton solve.");

  set("output destination", "silent");

  // Compute F(u) and J
  nonlinear_problem.F(*b, x);
  nonlinear_problem.J(*A, x);

  uint krylov_iterations = 0;
  newton_iteration = 0;
  bool newton_converged = false;

  // Start iterations
  while (!newton_converged && newton_iteration < maxiter)
  {
    // Perform linear solve and update total number of Krylov iterations
    if (dx->size() > 0)
      dx->zero();
    set("output destination", "silent");
    krylov_iterations += solver->solve(*A, *dx, *b);
    set("output destination", "terminal");

    // Compute initial residual
    if (newton_iteration == 0)
      newton_converged = converged(*b, *dx, nonlinear_problem);

    // Update solution
    x -= (*dx);

    ++newton_iteration;
    
    //FIXME: this step is not needed if residual is based on dx and this has converged.
    // Compute F(u) and J
    nonlinear_problem.F(*b, x);
    nonlinear_problem.J(*A, x);

    // Test for convergence 
    newton_converged = converged(*b, *dx, nonlinear_problem);
  }

  if (newton_converged)
    message("Newton solver finished in %d iterations and %d linear solver iterations.", 
            newton_iteration, krylov_iterations);
  else
    warning("Newton solver did not converge.");
  
  end();

  return newton_iteration;
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::getIteration() const
{
  return newton_iteration; 
}
//-----------------------------------------------------------------------------
bool NewtonSolver::converged(const GenericVector& b, const GenericVector& dx, 
                             const NonlinearProblem& nonlinear_problem)
{
  const std::string convergence_criterion = get("Newton convergence criterion");
  const double rtol   = get("Newton relative tolerance");
  const double atol   = get("Newton absolute tolerance");
  const bool report = get("Newton report");

  double residual = 1.0;

  // Compute resdiual
  if(convergence_criterion == "residual")
    residual = b.norm(l2);
  else if (convergence_criterion == "incremental")
    residual = dx.norm(l2);
  else
    error("Unknown Newton convergence criterion");

  // If this is the first iteration step, set initial residual
  if(newton_iteration == 0)
    residual0 = residual;

  // Relative residual
  double relative_residual = residual/residual0;

  // Output iteration number and residual
  //FIXME: allow precision to be set for dolfin::cout<<
  std::cout.precision(3);
  if(report && newton_iteration >0) 
    std::cout<< "  Iteration = " << newton_iteration  << ", Absolute, relative residual (" 
    << convergence_criterion  << " criterion) = " << std::scientific << residual 
    << ", "<< std::scientific << relative_residual << std::endl;

  // Return true of convergence criterion is met
  if(relative_residual < rtol || residual < atol)
    return true;

  // Otherwise return false
  return false;
}
//-----------------------------------------------------------------------------
