// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005
//
// First added:  2005
// Last changed: 2005-12-28
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving 
// problems of the form F(u) = 0. The user must provide functions for the 
// function (Fu) and update of the (approximate) Jacobian.  
//
// This simple program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = t * x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = t  for x = 0
//     du/dn(x, y) = 0  otherwise
//
// where t is pseudo time.
//
// The nonlinear solver is forced to iterate by perturbing the Jacobian matrix
// J = DF(u)/Du = 1.1*(grad v, grad u) 
//
// An incremental Newton approach is used in which several Netwon solves are 
// performed. 
//
// To verify the output from the nonlinear solver, the result is compared to a
// linear solution. This demo is useful for testing the nonlinear solver.
//

#include <dolfin.h>
#include "Poisson.h"
#include "PoissonNl.h"
  
using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return time()*p.x*sin(p.y);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 0.0*time();
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Matrix A;
  Vector xx, b, e;
  Function u(xx);

  // Forms for linear problem and create PDE
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);
  PDE linear_pde(a, L, mesh, bc);

  // Forms for nonlinear problem
  PoissonNl::BilinearForm a_nl;
  PoissonNl::LinearForm L_nl(u, f);
  NonlinearPDE nonlinear_pde(a, L, mesh, bc);

  real dt = 1.0;  // time step
  real t  = 0.0;  // initial time
  real T  = 1.0;  // final time
  f.sync(t);      // Associate time with source term
  bc.sync(t);     // Associate time with boundary conditions

//---------------------------------------------------

  // Solve linear PDE
  dolfin_log(false);

  t = T;
  linear_pde.set("solver", "direct");  
  Function v = linear_pde.solve();

//---------------------------------------------------

  // Solve nonlinear PDE using using NewtonSolver

  NewtonSolver newtonsolver;
  newtonsolver.set("Newton relative tolerance", 1.e-10);
  newtonsolver.setRtol(1.e-5);
  newtonsolver.setType(KrylovSolver::bicgstab);

  
  t = 0.0;
  while( t < T)
  {
    t += dt;
    dolfin_log(true);
//    newtonsolver.solve(a_nl, L_nl, bc, mesh, u);  
    nonlinear_pde.solve(u);  
  }
  cout << "Finished nonlinear solve. " << endl;
  
//-------------------------------------------------------

  // Verify nonlinear solver by comparing difference 
  // between linear solve and nonlinear solve

  Vector& x = u.vector();
  Vector& y = v.vector();

  e = x; e-=  y;
  cout << "Relative error: ||u^nonlin-u^lin|| / ||u^lin||= " 
      << e.norm()/y.norm() << endl; 

//-------------------------------------------------------
  
  return 0;
}
