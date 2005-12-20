// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.
//
// First added:  2005-10-23
// Last changed: 2005


#include <dolfin/ParameterSystem.h>
#include <dolfin/FEM.h>
#include <dolfin/NewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : KrylovSolver(), method(newton), report(true)
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(BilinearForm& a, LinearForm& L, 
      BoundaryCondition& bc, Mesh& mesh, Vector& x)
{  

  uint maxit = get("NLS Newton maximum iterations");
  real rtol  = get("NLS Newton relative convergence tolerance");
  real atol  = get("NLS Newton absolute convergence tolerance");
    
  dolfin_begin("Starting Newton solve.");

  dolfin_log(false);

  // Assemble Jacobian and residual
  FEM::assemble(a, L, A, b, mesh);
  FEM::applyBC(A, mesh, a.test(), bc);
  FEM::assembleBCresidual(b, x, mesh, L.test(), bc);

  real residual0 = b.norm();
  residual  = residual0;
  relative_residual = 1.0;

  kryloviterations = 0;
  iteration = 0;

  while( relative_residual > rtol && residual > atol && iteration <= maxit )
  {         
      ++iteration;

      dolfin_log(false);

      // Perform linear solve and update total number of Krylov iterations
      kryloviterations += KrylovSolver::solve(A, dx, b);  

      // Update solution
      x += dx;
      
      // Assemble Jacobian and residual
      FEM::assemble(a, L, A, b, mesh);
      FEM::applyBC(A, mesh, a.test(), bc);
      FEM::assembleBCresidual(b, x, mesh, L.test(), bc);

      dolfin_log(true);

      residual = b.norm();
      relative_residual = residual/residual0;

      if(report) cout << "Iteration= " << iteration << ", Relative residual = " 
          << relative_residual << endl;      
  }

  if ( relative_residual < rtol || residual < atol )
  {
    dolfin_info("Newton solver finished in %d iterations and %d linear solver iterations.", 
        iteration, kryloviterations);
  }
  else
  {
    dolfin_warning("Newton solver did not converge.");
  }

  dolfin_end();

  return iteration;
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearFunction& nonlinearfunction, Vector& x)
{
  uint maxit = get("NLS Newton maximum iterations");
  real rtol  = get("NLS Newton relative convergence tolerance");
  real atol  = get("NLS Newton absolute convergence tolerance");
  
  // FIXME: add option to compute F(u) anf J together or separately

  dolfin_begin("Starting Newton solve.");

  // Compute F(u) and J
  nonlinearfunction.form(A, b, x);
  
  real residual0 = b.norm();
  residual  = residual0;
  relative_residual = 1.0;

  kryloviterations = 0;
  iteration = 0;
  
  while( relative_residual > rtol && residual > atol && iteration < maxit )
  {
      ++iteration;

      dolfin_log(false);
      
      // Perform linear solve and update total number of Krylov iterations
      kryloviterations += KrylovSolver::solve(A, dx, b);  

      // Update solution
      x += dx;

      // Compute F(u) and J
      nonlinearfunction.form(A, b, x);

      dolfin_log(true);

      // Compute residual norm
      residual = b.norm();
      relative_residual = residual/residual0;

      if(report) cout << "Iteration= " << iteration << ", Relative residual = " 
          << relative_residual << endl;      
  }

  if ( relative_residual < rtol || residual < atol )
  {
    dolfin_info("Newton solver finished in %d iterations and %d linear solver iterations.", 
        iteration, kryloviterations);
  }
  else
  {
    dolfin_warning("Newton solver did not converge.");
  }

  dolfin_end();

  return iteration;
} 
//-----------------------------------------------------------------------------
void NewtonSolver::setNewtonMaxiter(uint maxiter) const
{
  set("NLS Newton maximum iterations", maxiter);
  dolfin_info("Maximum number of Newton iterations: %d.",maxiter);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setNewtonRtol(real rtol) const
{
  set("NLS Newton relative convergence tolerance", rtol);
  dolfin_info("Relative increment tolerance for Newton solver: %e.", rtol);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setNewtonAtol(real atol) const
{
  set("NLS Newton absolute convergence tolerance", atol);
  dolfin_info("Absolute increment tolerance for Newton solver: %e.", atol);
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::getIteration() const
{
  return iteration; 
}
//-----------------------------------------------------------------------------
