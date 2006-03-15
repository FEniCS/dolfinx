// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005
//
// First added:  2005
// Last changed: 2006-03-02
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

// User defined nonlinear problem 
class MyNonlinearProblem : public NonlinearProblem
{
  public:

    // Constructor 
    MyNonlinearProblem(Mesh& mesh, BoundaryCondition& bc, Function& U, Function& f) 
        : NonlinearProblem(), _mesh(&mesh), _bc(&bc)
    {
      // Create forms
      a = new NonlinearPoisson::BilinearForm(U);
      L = new NonlinearPoisson::LinearForm(U, f);

      // Initialise solution vector u
      U.init(mesh, a->trial());
    }

    // Destructor 
    ~MyNonlinearProblem()
    {
      delete a;
      delete L;
    }
 
    // User defined assemble of Jacobian and residual vector 
    void form(Matrix& A, Vector& b, const Vector& x)
    {
      dolfin_log(false);
      FEM::assemble(*a, *L, A, b, *_mesh);
      FEM::applyBC(A, *_mesh, a->test(), *_bc);
      FEM::assembleBCresidual(b, x, *_mesh, a->test(), *_bc);
      dolfin_log(true);
    }

  private:

    // Pointers to forms, mesh and boundary conditions
    BilinearForm *a;
    LinearForm *L;
    Mesh* _mesh;
    BoundaryCondition* _bc;
};




int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Function U;

  // Create user-defined nonlinear problem
  MyNonlinearProblem nonlinear_problem(mesh, bc, U, f);

  // Create nonlinear solver (using BICGSTAB linear solver) and set parameters
  NewtonSolver nonlinear_solver(KrylovSolver::bicgstab, Preconditioner::hypre_amg);
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-10);
  nonlinear_solver.set("Newton absolute tolerance", 1e-10);

  // Solve nonlinear problem in a series of steps
  real dt = 1.0; real t  = 0.0; real T  = 3.0;
  f.sync(t);
  bc.sync(t);

  Vector& x = U.vector();
  while( t < T)
  {
    t += dt;
    nonlinear_solver.solve(nonlinear_problem, x);
  }

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << U;

  return 0;
}
