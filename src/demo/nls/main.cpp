// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
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
//     f(x, y) = x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = 0  for x = 0
//     du/dn(x, y) = 0  otherwise
//
// This is equivalent to solving: 
// F(u) = (grad(v), grad(u)) - f(x,y) = 0
//
// The nonlinear solver is forced to iterate by perturbing the Jacobian matrix
// J = DF(u)/Du = 1.1*(grad(v), D(grad(u))) 
//
// To verify the output from the nonlinear solver, the result is compared to a
// linear solution. 
//

#include <dolfin.h>
#include "Poisson.h"
#include "PoissonNl.h"
  
using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real operator() (const Point& p) const
  {
    return p.x*sin(p.y);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 0.0;
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
 
    // Compute F(u) and J at same time
    void form(Matrix& A, Vector& b, const Vector& x)
    {
      BilinearForm& a = *_a;
      LinearForm& L   = *_L;
      BoundaryCondition& bc = *_bc;
      Mesh& mesh = *_mesh;

      // Update function u0
      Vector& x0 = _u0->vector();
      x0 = x;

      // Assemble RHS vector, Jacobian, and apply boundary conditions 
      dolfin_log(false);
      FEM::assemble(a, L, A, b, mesh, bc);
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
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Matrix A;
  Vector x, x0, y, b;
  Function u0(x0);

  // Forms for linear problem
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  // Forms for nonlinear problem
  PoissonNl::BilinearForm a_nl;
  PoissonNl::LinearForm L_nl(u0, f);

  // Define nonlinear function
  MyNonlinearFunction nonlinear_function(a_nl, L_nl, mesh, bc, u0);  

  // Initialise nonlinear solver
  NonlinearSolver nonlinear_solver;
  nonlinear_solver.setMaxiter(8);
  nonlinear_solver.setRtol(1e-5);

  // Solve nonlinear problem
  cout << "Starting nonlinear assemble and solve. " << endl;
  nonlinear_solver.solve(nonlinear_function, x);
  cout << "Finished nonlinear solve. " << endl;
  
  // Assemble and solve linear problem
  cout << "Starting linear assemble and solve. " << endl;
  dolfin_log(false);
  FEM::assemble(a, L, A, b, mesh, bc);
  dolfin_log(true);
  GMRES solver;
  solver.solve(A, y, b);  
  cout << "Finished linear solve. " << endl;
  
  // Verify nonlinear solver by comparing difference between linear solve
  // and nonlinear solve
  Vector e;
  e = x; e-=  y;
  cout << " norm || u^nonlin - u^lin || =  " << e.norm() << endl; 

  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson_nl.pvd");
  file << u;

  return 0;
}
