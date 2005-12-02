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
      value = 0.0*time();
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
    // Compute the residual F(u) 
    void F(Vector& b, const Vector& x)
    {
      LinearForm& L   = *_L;
      BoundaryCondition& bc = *_bc;
      Mesh& mesh = *_mesh;
 
      // Update function u0
      Vector& x0 = _u0->vector();
      x0 = x;

      // Assemble residual vector 
      dolfin_log(false);
      FEM::assemble(L, b, mesh);

      // Assemble BC to RHS vector
      FEM::assembleBCresidual(b, x, mesh, L.test(), bc);
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


int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(128, 128);
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
  real T  = 1.0;  // final time
  f.sync(t);      // Associate time with source term
  bc.sync(t);     // Associate time with boundary conditions

//---------------------------------------------------
  // Assemble and solve linear problem and time
  dolfin_log(false);
  GMRES solver;
  solver.setRtol(1.e-10);
  dolfin_log(true);

  t = 1.0;
  cout << "Starting linear assemble and solve. " << endl;
  dolfin_log(false);
  tic();
  FEM::assemble(a, L, A, b, mesh, bc);
  dolfin_log(true);
  solver.solve(A, y, b);  
  cout << "**** Finished linear solve. Time = " << toc() << endl;

//-------------------------------------------------------
  // Homemade Netwon procedure
  GMRES solver2;
  solver2.setRtol(1.e-10);

  x0.init(mesh.noVertices());
  x0 = 0.0;
  x = x0;
  
  t = 0.0;
  int iteration;
  Vector dx;
  real residual0;
  real residual;
  real rel_residual;

  cout << "Using homemade nonlinear assemble and solve. " << endl;
  while( t < T)
  {
    t += dt;
    cout << "   Starting test Newton step. Time = " << t << endl;
    iteration = 0;
    rel_residual = 1.0;

    tic();
    while( rel_residual > 1e-10 && iteration < 30 )
    {
      x0 = x;

      dolfin_log(false);
      FEM::assemble(a_nl, L_nl, A, b, mesh);
      FEM::applyBC(A, mesh, a_nl.test(), bc);
      FEM::assembleBCresidual(b, x, mesh, L_nl.test(), bc);

      if( iteration == 0 ) residual0 = b.norm();

      b *= -1.0;
      solver2.solve(A, dx, b);  
      x += dx;
      dx = 0.0;

      rel_residual = b.norm()/residual0;

      dolfin_log(true);
      cout << "  Residual = " << rel_residual << " Iteration= " << iteration << endl;      
      ++iteration;
    }
      dolfin_log(true);
    cout << "   **** Newton test solve time = " << toc() << endl;
  }
  cout << "Finished homemade nonlinear solve. " << endl << endl;
  
//-------------------------------------------------------
  // Verify nonlinear solver by comparing difference between linear solve
  // and nonlinear solve
  Vector e;
  e = x; e-=  y;
  cout << " Relative error (homemade)  || u^nonlin - u^lin || / || u^lin ||=  " << e.norm()/y.norm() << endl; 
//-------------------------------------------------------
  // Using PETSc solvers
  t = 0.0;

  // Associate matrix and vectors with solver
  nonlinear_solver.init(A, b, x);

  x0 = 0.0;

  // Solve nonlinear problem
  cout << "Using PETSc nonlinear assemble and solve. " << endl;
  while( t < T)
  {
    t += dt;
    cout << "   Starting Newton step. Time = " << t << endl;
    tic();
    nonlinear_solver.solve();
    cout << "   **** Newton solve time " << toc() << endl;
  }
  cout << "Finished nonlinear solve. " << endl;


  
//-------------------------------------------------------
  // Verify nonlinear solver by comparing difference between linear solve
  // and nonlinear solve
//  Vector e;
  e = x; e-=  y;
  cout << " Relative error || u^nonlin - u^lin || / || u^lin ||=  " << e.norm()/y.norm() << endl; 
//-------------------------------------------------------
  
  return 0;
}
