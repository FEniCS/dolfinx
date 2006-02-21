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
// This simple program solves a nonlinear variant of Poisson's equation
//
//     - div (1+u^2) grad u(x, y) = f(x, y)
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
// This is equivalent to solving: 
// F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0
//

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
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
      value = 1.0*time();
  }
};

// User defined nonlinear function 
class MyNonlinearPDE : public NonlinearPDE
{
  public:
  
    // Constructor 
    MyNonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh,
      BoundaryCondition& bc) : NonlinearPDE(a, L, mesh, bc) {}

    // User defined assemble of Jacobian and residual vector 
    void form(Matrix& A, Vector& b, const Vector& x)
    {
      dolfin_log(false);
      FEM::assemble(*_a, *_Lf, A, b, *_mesh);
      FEM::applyBC(A, *_mesh, _a->test(), *_bc);
      FEM::assembleBCresidual(b, x, *_mesh, _a->test(), *_bc);
      dolfin_log(true);
    }
};


int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Vector x;
  Function u(x, mesh);

  // Create forms and nonlinear PDE
  NonlinearPoisson::BilinearForm a(u);
  NonlinearPoisson::LinearForm L(u, f);
  MyNonlinearPDE nonlinear_pde(a, L, mesh, bc);  

  u.init(mesh, a.trial());

  // Create nonlinear solver
  NewtonSolver nonlinear_solver;

  // Set Newton parameters
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-10);
  nonlinear_solver.set("Newton absolute tolerance", 1e-10);

  // Set Krylov solver type
  nonlinear_solver.setType(KrylovSolver::bicgstab);

  // Solve nonlinear problem in a series of steps
  real dt = 1.0;
  real t  = 0.0;
  real T  = 3.0;
  f.sync(t);
  bc.sync(t);

//  while( t < T)
//  {
//    t += dt;
//    nonlinear_solver.solve(nonlinear_pde, u);
//  }


  while( t < T)
  {
    t += dt;
    nonlinear_pde.solve(u);
  }

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
