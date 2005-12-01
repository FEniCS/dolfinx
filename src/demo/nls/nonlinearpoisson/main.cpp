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
      BoundaryCondition& bc, Function& u0) : NonlinearFunction(), 
      _a(&a), _L(&L), _mesh(&mesh), _bc(&bc), _u0(&u0) {}
 
    // Compute F(u) 
    void F(Vector& b, const Vector& x)
    {
      BilinearForm& a = *_a;
      LinearForm& L   = *_L;
      BoundaryCondition& bc = *_bc;
      Mesh& mesh = *_mesh;
 
      // Update function u0
      Vector& x0 = _u0->vector();
      x0 = x;

      // Assemble RHS vector 
      dolfin_log(false);
      FEM::assemble(L, b, mesh);

      // Assemble BC to RHS vector 
      FEM::applyBC(b, x, mesh, a.test(), bc);
      dolfin_log(true);

    }

    // Compute J
    void J(Matrix& A, const Vector& x)
    {
      BilinearForm& a = *_a;
      BoundaryCondition& bc = *_bc;
      Mesh& mesh = *_mesh;

      // Assemble Jacobian, and apply boundary conditions 
      dolfin_log(false);
      FEM::assemble(a, A, mesh);
      FEM::applyBC(A, mesh, a.test(), bc);
      dolfin_log(true);
    }

    // Compute system size
    dolfin::uint size()
    {      
      return FEM::size(*_mesh, (*_a).test());
    }

    // Compute maximum number of nonzero terms for a row
    dolfin::uint nzsize()
    {      
      return FEM::nzsize(*_mesh, (*_a).test());
    }

  private:

    // Pointer to forms, mesh data and boundary conditions
    BilinearForm* _a;
    LinearForm* _L;
    Mesh* _mesh;
    BoundaryCondition* _bc;

    // Pointer to functions in forms
    Function* _u0;
};


int main()
{
  // Set up problem
  UnitSquare mesh(4, 4);
  MyFunction f;
  MyBC bc;
  Matrix A;
  Vector x, x0, y, b;
  Function u0(x0);

  // Forms
  NonlinearPoisson::BilinearForm a(u0);
  NonlinearPoisson::LinearForm L(u0, f);

  // Create nonlinear function
  MyNonlinearFunction nonlinear_function(a, L, mesh, bc, u0);  

  // Initialise nonlinear solver
  NewtonSolver nonlinear_solver(nonlinear_function);

  // Set Newton parameters
  nonlinear_solver.setMaxiter(50);
  nonlinear_solver.setRtol(1e-10);
  nonlinear_solver.setAtol(1e-10);
  nonlinear_solver.setParameters();

  // Solve nonlinear problem
  cout << "Starting nonlinear assemble and solve. " << endl;

  real dt = 1.0;
  real t  = 0.0;
  real T  = 3.0;
  f.sync(t);
  bc.sync(t);

  // Associate matrix and vectors with solver
  nonlinear_solver.init(A, b, x);
  while( t < T)
  {
    t += dt;
    cout << "Starting step " << t << endl;
    nonlinear_solver.solve();
  }
  cout << "Finished nonlinear solve. " << endl;


  // Save function to file
  Function u(x, mesh, a.trial());
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
