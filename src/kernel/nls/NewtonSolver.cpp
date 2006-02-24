// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005-2006.
//
// First added:  2005-10-23
// Last changed: 2006-02-24

#include <dolfin/FEM.h>
#include <dolfin/NewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : KrylovSolver() 
{
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearProblem& nonlinear_problem, Function& u)
{
  // Associate vector u with function
  Vector& x = u.vector();

  // Solve nonlinear problem and return number of iterations
  return solve(nonlinear_problem, x);
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearProblem& nonlinear_problem, Vector& x)
{
  const uint maxit= get("Newton maximum iterations");
  const real rtol = get("Newton relative tolerance");
  const real atol = get("Newton absolute tolerance");

  // FIXME: add option to compute F(u) anf J together or separately

  dolfin_begin("Starting Newton solve.");

  bool report = get("Newton report");

  // Compute F(u) and J
  nonlinear_problem.form(A, b, x);
  
  real residual0    = b.norm();
  residual          = residual0;
  relative_residual = 1.0;

  uint krylov_iterations = 0;
  newton_iteration = 0;
  
  while( relative_residual > rtol && residual > atol && newton_iteration < maxit )
  {
      ++newton_iteration;

      dolfin_log(false);
 
      // Perform linear solve and update total number of Krylov iterations
      krylov_iterations += KrylovSolver::solve(A, dx, b);  

      // Update solution
      x += dx;

      // Compute F(u) and J
      nonlinear_problem.form(A, b, x);

      dolfin_log(true);

      // Compute residual norm
      residual = b.norm();
      relative_residual = residual/residual0;

      if(report) cout << "Iteration= " << newton_iteration << ", Relative residual = " 
          << relative_residual << endl;      
  }

  if ( relative_residual < rtol || residual < atol )
  {
    dolfin_info("Newton solver finished in %d iterations and %d linear solver iterations.", 
        newton_iteration, krylov_iterations);
  }
  else
  {
    dolfin_warning("Newton solver did not converge.");
  }

  dolfin_end();

   return newton_iteration;
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::getIteration() const
{
  return newton_iteration; 
}
//-----------------------------------------------------------------------------
