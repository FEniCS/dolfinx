// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005
//
// First added:  2005
// Last changed: 2005-11-30
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving 
// problems of the form F(u) = 0. The user must provide functions for the 
// function (Fu) and update of the (approximate) Jacobian.  
//
// This simple program solves a nonlinear variant of Poisson's equation
//
//     - div (1+u) grad u(x, y) = f(x, y)
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
// F(u) = (grad(v), (1-u)*grad(u)) - f(x,y) = 0
//
// The nonlinear solver is forced to iterate by perturbing the Jacobian matrix
// J = DF(u)/Du = 1.1*(grad(v), D(grad(u))) 
//
// To verify the output from the nonlinear solver, the result is compared to a
// linear solution. 
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
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 1.0*time();
    return value;
  }
};


// User defined nonlinear update class
class MyNonlinearFunction : public NonlinearFunction
{
  public:

    // Constructor 
    MyNonlinearFunction(BilinearForm& a, LinearForm& L, Mesh& mesh,  
      BoundaryCondition& bc) : NonlinearFunction(), 
      _a(&a), _L(&L), _mesh(&mesh), _bc(&bc) {}
 

    // Assemble Jacobian and residual vector 
    void form(Matrix&A, Vector& b, const Vector& x)
    {
      BilinearForm& a = *_a;
      LinearForm& L   = *_L;
      BoundaryCondition& bc = *_bc;
      Mesh& mesh = *_mesh;
 
      dolfin_log(false);
      FEM::assemble(a, L, A, b, mesh);
      FEM::applyBC(A, mesh, a.test(), bc);
      FEM::assembleBCresidual(b, x, mesh, a.test(), bc);
      dolfin_log(true);

    }
    
  private:

    // Pointer to forms, mesh data and boundary conditions
    BilinearForm* _a;
    LinearForm* _L;
    Mesh* _mesh;
    BoundaryCondition* _bc;
};


int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(256, 256);
  MyFunction f;
  MyBC bc;
  Matrix A;
  Vector x, b;
  Function u(x);

  // Forms
  NonlinearPoisson::BilinearForm a(u);
  NonlinearPoisson::LinearForm L(u, f);

  x.init(FEM::size(mesh, a.test()));

  // Create nonlinear function
  MyNonlinearFunction nonlinear_function(a, L, mesh, bc);  

  // Create nonlinear solver
  NewtonSolver nonlinear_solver;

  // Set Newton parameters
  nonlinear_solver.setNewtonMaxiter(50);
  nonlinear_solver.setNewtonRtol(1e-10);
  nonlinear_solver.setNewtonAtol(1e-10);
  nonlinear_solver.setType(KrylovSolver::bicgstab);

  // Solve nonlinear problem
  real dt = 1.0;
  real t  = 0.0;
  real T  = 3.0;
  f.sync(t);
  bc.sync(t);

  while( t < T)
  {
    t += dt;
    nonlinear_solver.solve(nonlinear_function, x);
  }

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
