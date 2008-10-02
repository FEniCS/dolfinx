// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005, 2007
//
// First added:  2005
// Last changed: 2007-08-20
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

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
using namespace dolfin;

// Right-hand side
class Source : public Function, public TimeDependent
{
public:
  
    Source(Mesh& mesh, const double* t) : Function(mesh), TimeDependent(t) {}

    double eval(const double* x) const
    {
      return time()*x[0]*sin(x[1]);
    }

};

// Dirichlet boundary condition
class DirichletBoundaryCondition : public Function, public TimeDependent
{
public:
  DirichletBoundaryCondition(Mesh& mesh, const double* t) : Function(mesh), TimeDependent(t) {}
  
  double eval(const double* x) const
  {
    return 1.0*time();
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
  }
};

// User defined nonlinear problem 
class MyNonlinearProblem : public NonlinearProblem
{
  public:

    // Constructor 
    MyNonlinearProblem(Mesh& mesh, SubDomain& dirichlet_boundary, 
                       Function& g, Function& f, Function& u)  
                       : assembler(mesh)
    {
      // Create forms
      a = new NonlinearPoissonBilinearForm(u);
      L = new NonlinearPoissonLinearForm(u, f);

      // Create boundary conditions
      bc = new DirichletBC(g, mesh, dirichlet_boundary);

      // Initialise dof map
      dof_map_set.update(a->form(), mesh);

      // Initialise solution vector u
      u.init(mesh, *a, 1);
    }

    // Destructor 
    ~MyNonlinearProblem()
    {
      delete a;
      delete L;
      delete bc;
    }
 
    // User defined assemble of residual vector 
    void F(GenericVector& b, const GenericVector& x)
    {
      //dolfin_set("output destination", "silent");
      assembler.assemble(b, *L);
      bc->apply(b, x, *a);
      //dolfin_set("output destination", "terminal");
    }

    // User defined assemble of Jacobian matrix 
    void J(GenericMatrix& A, const GenericVector& x)
    {
      //dolfin_set("output destination", "silent");
      assembler.assemble(A, *a);
      bc->apply(A, *a);
      //dolfin_set("output destination", "terminal");
    }


  private:

    Assembler assembler;

    // Pointers to forms, mesh and boundary conditions
    Form *a;
    Form *L;
    DirichletBC* bc;
    DofMapSet dof_map_set;
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Create mesh
  UnitSquare mesh(64, 64);

  // Pseudo time
  double t = 0.0;

  // Create source function
  Source f(mesh, &t);

  // Dirichlet boundary conditions
  DirichletBoundary dirichlet_boundary;
  DirichletBoundaryCondition g(mesh, &t);

  // Solution vector
  Function u;

  // Create user-defined nonlinear problem
  MyNonlinearProblem nonlinear_problem(mesh, dirichlet_boundary, g, f, u);

  // Create nonlinear solver (using GMRES linear solver) and set parameters
  // NewtonSolver nonlinear_solver(gmres);
  NewtonSolver nonlinear_solver;
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-10);
  nonlinear_solver.set("Newton absolute tolerance", 1e-10);

  // Solve nonlinear problem in a series of steps
  GenericVector& x = u.vector();
  double dt = 1.0; double T  = 3.0;
  while( t < T)
  {
    t += dt;
    nonlinear_solver.solve(nonlinear_problem, x);
  }

  // Plot solution
  plot(u);

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
