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
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 1.0*time();
    return value;
  }
};


// User defined nonlinear funnction class
class MyNonlinearFunction : public NonlinearFunction
{
  public:

    // Constructor 
    MyNonlinearFunction(BilinearForm& a, LinearForm& L, Mesh& mesh,  
      BoundaryCondition& bc, Function& u0) : NonlinearFunction(), 
      _a(&a), _L(&L), _mesh(&mesh), _bc(&bc), _u0(&u0) {}
 
/*
    // Compute F(u) and J at same time
    void form(Matrix& A, Vector& b, const Vector& x, real t)
    {
    }

*/
    // Compute F(u) 
    void F(Vector& b, const Vector& x)
    {
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
      FEM::applyBC(b, x, mesh, L.test(), bc);
      dolfin_log(true);
    }

    // Compute J
    void J(Matrix& A, const Vector& x)
    {
      BilinearForm& a   = *_a;
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

  // Forms for linear problem
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  // Forms for nonlinear problem
  PoissonNl::BilinearForm a_nl;
  PoissonNl::LinearForm L_nl(u0, f);

  // Create nonlinear function
  MyNonlinearFunction nonlinear_function(a_nl, L_nl, mesh, bc, u0);  

  // Initialise nonlinear solver
  NewtonSolver nonlinear_solver(nonlinear_function);

  // Set Newton parameters
  nonlinear_solver.setMaxiter(50);  // Set maximum number of Newton iterations
  nonlinear_solver.setRtol(1e-10);  // Set relative convergence tolerance
  nonlinear_solver.setParameters();


  real dt = 1.0;  // time step
  real t  = 0.0;  // initial time
  real T  = 3.0;  // final time
  f.sync(t);      // Associate time with source term
  bc.sync(t);     // Associate time with boundary conditions

  // Associate matrix and vectors with solver
  nonlinear_solver.init(A, b, x);

  // Solve nonlinear problem
  cout << "Starting nonlinear assemble and solve. " << endl;
  while( t < T)
  {
    t += dt;
    cout << "Starting Newton step. Time = " << t << endl;
    nonlinear_solver.solve();
  }
  cout << "Finished nonlinear solve. " << endl;


  // Assemble and solve linear problem
  cout << "Starting linear assemble and solve. " << endl;
  dolfin_log(false);
  FEM::assemble(a, L, A, b, mesh, bc);
  GMRES solver;
  solver.setRtol(1.e-15);
  solver.solve(A, y, b);  
  dolfin_log(true);
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
