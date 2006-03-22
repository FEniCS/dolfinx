// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005-2006.
//
// First added:  2005-10-23
// Last changed: 2006-03-22

#include <dolfin/FEM.h>
#include <dolfin/NewtonSolver.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : Parametrized()
{
  solver = new LU;
  A = new Matrix(Matrix::umfpack);
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(KrylovSolver::Type linear_solver) : Parametrized()
{
  solver = new KrylovSolver(linear_solver);
  A = new Matrix;
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(KrylovSolver::Type linear_solver, 
    Preconditioner::Type preconditioner) : Parametrized()
{
  solver = new KrylovSolver(linear_solver, preconditioner);
  A = new Matrix;
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  delete solver; 
  delete A;
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearProblem& nonlinear_problem, Vector& x)
{
  const uint maxit= get("Newton maximum iterations");

  // FIXME: add option to compute F(u) anf J together or separately

  dolfin_begin("Starting Newton solve.");

  // Compute F(u) and J
  dolfin_log(false);
  nonlinear_problem.form(*A, b, x);

  uint krylov_iterations = 0;
  newton_iteration = 0;
  bool newton_converged = false;

  // Start iterations
  while( !newton_converged && newton_iteration < maxit )
  {

      dolfin_log(false);
      // Perform linear solve and update total number of Krylov iterations
      krylov_iterations += solver->solve(*A, dx, b);
      dolfin_log(true);

      // Compute initial residual
      if(newton_iteration == 0)
        newton_converged = converged(b, dx, nonlinear_problem);

      // Update solution
      x += dx;

      ++newton_iteration;

      dolfin_log(false);
      //FIXME: this step is not needed if residual is based on dx and this has converged.
      // Compute F(u) and J
      nonlinear_problem.form(*A, b, x);

      dolfin_log(true);

      // Test for convergence 
      newton_converged = converged(b, dx, nonlinear_problem);
  }

  if(newton_converged)
    dolfin_info("Newton solver finished in %d iterations and %d linear solver iterations.", 
        newton_iteration, krylov_iterations);
  else
    dolfin_warning("Newton solver did not converge.");

  dolfin_end();

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
    dolfin_error("Unknown Newton convergence criterion");

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
