// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/ComplexODE.h>
#include <dolfin/HomotopyJacobian.h>
#include <dolfin/HomotopyODE.h>
#include <dolfin/Homotopy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Homotopy::Homotopy(uint n)
  : n(n), M(0), maxiter(0), tol(0.0), fp(0), mi(0), ci(0), x(2*n)
{
  dolfin_info("Creating homotopy for system of size %d.", n);
  
  // We should not solve the dual problem
  dolfin_set("solve dual problem", false);

  // System is implicit
  dolfin_set("implicit", true);

  // Need to use the new ODE solver
  dolfin_set("use new ode solver", true);

  // Get maximum number of iterations
  maxiter = dolfin_get("maximum iterations");

  // FIXME: Maybe this should be a parameter?
  tol = 1e-10;
  
  // Set tolerance for GMRES solver
  solver.setAtol(0.1*tol);

  // Don't want the GMRES solver to report the number of iterations
  solver.setReport(false);
  
  // Open file
  fp = fopen("solution.data", "w");

  // Initialize array mi
  mi = new uint[n];
  for (uint i = 0; i < n; i++)
    mi[i] = 0;

  // Initialize array ci
  ci = new complex[n];
  for (uint i = 0; i < n; i++)
    ci[i] = 0.0;

  // Randomize vector ci
  randomize();
}
//-----------------------------------------------------------------------------
Homotopy::~Homotopy()
{
  if ( mi ) delete [] mi;
  if ( ci ) delete [] ci;

  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void Homotopy::solve()
{
  // Compute the total number of paths
  M = countPaths();
  dolfin_info("Total number of paths is %d.", M);

  char filename[64];

  for (uint m = 0; m < M; m++)
  {
    dolfin_info("Computing path number %d out of %d.", m, M);

    // Change name of output file for each path
    sprintf(filename, "primal_%d.m", m);
    dolfin_set("file name", filename);

    // Compute the component paths from global path number
    computePath(m);

    // Create and solve ODE
    HomotopyODE ode(*this, n);
    ode.solve();

    // Use Newton's method to find the solution
    if ( ode.state() == HomotopyODE::endgame )
    {
      dolfin_info("Homotopy path converged, using Newton's method to improve solution.");
      computeSolution(ode);
      saveSolution();
    }
  }
}
//-----------------------------------------------------------------------------
dolfin::uint Homotopy::countPaths() const
{
  uint product = 1;
  for (uint i = 0; i < n; i++)
    product *= (degree(i) + 1);

  return product;
}
//-----------------------------------------------------------------------------
void Homotopy::computePath(uint m)
{
  // The path number for each component can vary between 0 and p_i + 1,
  // and we need to compute the local path number for a given component
  // from the global path number which varies between 0 and the product of
  // all p_i + 1. This algorithm is copied from FFC (compiler.multiindex).
  
  uint posvalue = M;
  uint sum = m;
  for (uint i = 0; i < n; i++)
  {
    const uint dim = degree(i) + 1;
    posvalue /= dim;
    const uint digit = sum / posvalue;
    mi[i] = digit;
    sum -= digit * posvalue;
  }
}
//-----------------------------------------------------------------------------
void Homotopy::computeSolution(HomotopyODE& ode)
{
  // Create right-hand side and increment vector
  NewVector F(2*n), dx(2*n);

  // Create matrix-free Jacobian
  HomotopyJacobian J(ode, x);
  J.init(dx, dx);
  
  cout << "Starting point: ";
  x.disp();

  // Solve system using Newton's method
  for (uint iter = 0; iter < maxiter; iter++)
  {
    // Evaluate right-hand side at current x
    feval(F, ode);

    // Check convergence
    real r = F.norm(NewVector::linf);
    if ( r < tol )
    {
      cout << "Solution converged: x = ";
      x.disp();
      return;
    }
    
    //cout << "x = "; x.disp();
    //cout << "F = "; F.disp();
    //cout << "dx = "; dx.disp();
    //cout << "J = "; J.disp();
    //cout << endl;

    // Solve linear system, seems like we need to scale the right-hand
    // side to make it work with the PETSc GMRES solver
    r += DOLFIN_EPS;
    F /= r;
    solver.solve(J, dx, F);
    dx *= r;

    // Subtract increment
    x -= dx;

    x.disp();
  }

  dolfin_error("Solution did not converge.");
}
//-----------------------------------------------------------------------------
void Homotopy::saveSolution()
{
  real* xx = x.array();
  for (uint i = 0; i < (2*n); i++)
  {
    fprintf(fp, "%.14e ", xx[i]);
  }
  fprintf(fp, "\n");
  x.restore(xx);
}
//-----------------------------------------------------------------------------
void Homotopy::randomize()
{
  // Randomize each c in the unit circle

  for (uint i = 0; i < n; i++)
  {
    const real r = rand();
    const real a = 2.0*DOLFIN_PI*rand();
    const complex c = std::polar(r, a);
    ci[i] = c;
  }
  
  //const complex tmp(0.00143289, 0.982727);
  //ci[0] = tmp;
}
//-----------------------------------------------------------------------------
void Homotopy::feval(NewVector& F, ComplexODE& ode)
{
  // Get arrays
  const real* xx = x.array();
  real* FF = F.array();
  
  // Evaluate F at current x
  ode.feval(xx, 0.0, FF);

  // Restore arrays
  x.restore(xx);
  F.restore(FF);
}
//-----------------------------------------------------------------------------
