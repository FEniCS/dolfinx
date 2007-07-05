// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2006.
//
// First added:  2005-10-23
// Last changed: 2007-05-15

#include <dolfin/NewtonSolver.h>
#include <dolfin/NonlinearProblem.h>
#include <dolfin/LUSolver.h>
#include <dolfin/KrylovSolver.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : Parametrized()
{
  solver = new LUSolver();
}
//-----------------------------------------------------------------------------
/*
#ifdef HAVE_PETSC_H
NewtonSolver::NewtonSolver(Matrix::Type matrix_type) : Parametrized()
{
  solver = new LUSolver();
  // FIXME: Need to select appropriate PETSc matrix
  //A = new Matrix(matrix_type);
  A = new Matrix;
}
#endif
*/
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(KrylovMethod method, Preconditioner pc)
  : Parametrized()
{
  solver = new KrylovSolver(method, pc);
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  delete solver; 
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearProblem& nonlinear_problem, Vector& x)
{
  const uint maxit= get("Newton maximum iterations");

  // FIXME: add option to compute F(u) anf J together or separately

  begin("Starting Newton solve.");

  // Compute F(u) and J
  set("output destination", "silent");
  nonlinear_problem.form(A, b, x);

  uint krylov_iterations = 0;
  newton_iteration = 0;
  bool newton_converged = false;

  // Start iterations
  while( !newton_converged && newton_iteration < maxit )
  {

      set("output destination", "silent");
      // Perform linear solve and update total number of Krylov iterations
      krylov_iterations += solver->solve(A, dx, b);
      set("output destination", "terminal");

      // Compute initial residual
      if(newton_iteration == 0)
        newton_converged = converged(b, dx, nonlinear_problem);

      // Update solution
      x += dx;

      ++newton_iteration;

      set("output destination", "silent");
      //FIXME: this step is not needed if residual is based on dx and this has converged.
      // Compute F(u) and J
      nonlinear_problem.form(A, b, x);

      set("output destination", "terminal");

      // Test for convergence 
      newton_converged = converged(b, dx, nonlinear_problem);
  }

  if(newton_converged)
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
bool NewtonSolver::converged(const Vector& b, const Vector& dx, 
      const NonlinearProblem& nonlinear_problem)
{
  const std::string convergence_criterion = get("Newton convergence criterion");
  const real rtol   = get("Newton relative tolerance");
  const real atol   = get("Newton absolute tolerance");
  const bool report = get("Newton report");

  real residual = 1.0;

  // Compute resdiual
  if(convergence_criterion == "residual")
    residual = b.norm();
  else if (convergence_criterion == "incremental")
    residual = dx.norm();
  else
    error("Unknown Newton convergence criterion");

  // If this is the first iteration step, set initial residual
  if(newton_iteration == 0)
    residual0 = residual;

  // Relative residual
  real relative_residual = residual/residual0;

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

