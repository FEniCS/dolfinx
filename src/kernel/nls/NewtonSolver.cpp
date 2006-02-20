// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005-2006.
//
// First added:  2005-10-23
// Last changed: 2006-02-20

#include <dolfin/FEM.h>
#include <dolfin/NewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : KrylovSolver(), method(newton), report(true)
{
  // Set default parameters
  add("Newton maximum iterations", 50);
  add("Newton relative tolerance", 1e-9);
  add("Newton absolute tolerance", 1e-20);
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Do nothing 
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(BilinearForm& a, LinearForm& L, 
      BoundaryCondition& bc, Mesh& mesh, Function& u)
{  
  // Create nonlinear function
  NonlinearPDE nonlinear_pde(a, L, mesh, bc);

  // Inintialise function and associate vector u with function
  u.init(mesh, a.trial());
  Vector& x = u.vector();

  // Solve nonlinear problem and return number of iterations
  return solve(nonlinear_pde, x);

} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearPDE& nonlinear_pde, Function& u)
{  

  Mesh& mesh = *(nonlinear_pde._mesh);
  FiniteElement& element = (nonlinear_pde._a)->trial();

  // Inintialise function and associate vector u with function
  u.init(mesh, element);
  Vector& x = u.vector();

  // Solve nonlinear problem and return number of iterations
  return solve(nonlinear_pde, x);

} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearFunction& nonlinearfunction, Vector& x)
{
  const uint maxit= get("Newton maximum iterations");
  const real rtol = get("Newton relative tolerance");
  const real atol = get("Newton absolute tolerance");

  // FIXME: add option to compute F(u) anf J together or separately

  dolfin_begin("Starting Newton solve.");

  // Compute F(u) and J
  nonlinearfunction.form(A, b, x);
  
  real residual0 = b.norm();
  residual  = residual0;
  relative_residual = 1.0;

  uint krylov_iterations = 0;
  iteration = 0;
  
  while( relative_residual > rtol && residual > atol && iteration < maxit )
  {
      ++iteration;

      dolfin_log(false);
 
      // Perform linear solve and update total number of Krylov iterations
      krylov_iterations += KrylovSolver::solve(A, dx, b);  

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
        iteration, krylov_iterations);
  }
  else
  {
    dolfin_warning("Newton solver did not converge.");
  }

  dolfin_end();

  return iteration;
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::getIteration() const
{
  return iteration; 
}
//-----------------------------------------------------------------------------
