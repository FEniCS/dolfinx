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
// This simple program solves a nonlinear variant of Poisson's equation,
//
//     - div (1+u^2) grad u(x, y) = f(x, y),
//
// on the unit square with source f given by
//
//     f(x, y) = t * x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = t  for x = 0,
//     du/dn(x, y) = 0  otherwise,
//
// where t is pseudo time.
//
// This is equivalent to solving
//
//     F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0.

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
using namespace dolfin;

// Right-hand side
class Source : public Function, public TimeDependent
{
public:
  
  Source(const double& t) : t(t) {}

  void eval(double* values, const Data& data) const
  {
    const double* x = data.x;
    values[0] = t*x[0]*sin(x[1]);
  }
  
private:

  const double & t;

};

// Dirichlet boundary condition
class DirichletBoundaryValue : public Function, public TimeDependent
{
public:
  
  DirichletBoundaryValue(const double& t) : t(t) {}
  
  void eval(double* values, const Data& data) const
  {
    values[0] = 1.0*t;
  }

private:

  const double& t;

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
    MyNonlinearProblem(FunctionSpace& V, SubDomain& dirichlet_boundary, 
                       Function& g, Function& f, Function& u)  
                       : V(V), a(V, V), L(V), bc(g, V, dirichlet_boundary)
    {
      // Attach functions
      a.u0 = u;
      L.u0 = u;
      L.f  = f;
    }

    // User defined assemble of residual vector 
    void F(GenericVector& b, const GenericVector& x)
    {
      Assembler::assemble(b, L);
      bc.apply(b, x);
    }

    // User defined assemble of Jacobian matrix 
    void J(GenericMatrix& A, const GenericVector& x)
    {
      Assembler::assemble(A, a);
      bc.apply(A);
    }

  private:

    FunctionSpace& V;
    NonlinearPoissonBilinearForm a;
    NonlinearPoissonLinearForm L;
    DirichletBC bc;

};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Create mesh
  UnitSquare mesh(16, 16);

  // Pseudo time
  double t = 0.0;

  // Create source function
  Source f(t);

  // Dirichlet boundary conditions
  DirichletBoundary dirichlet_boundary;
  NonlinearPoissonTrialSpace V(mesh);
  DirichletBoundaryValue g(t);

  // Solution vector
  Function u;

  // Create user-defined nonlinear problem
  MyNonlinearProblem nonlinear_problem(V, dirichlet_boundary, g, f, u);

  // Create linear solver for use in Newton solver
  DefaultFactory factory;
  KrylovSolver solver(gmres);
  solver.set("Krylov relative tolerance", 1e-10);
  solver.set("Krylov absolute tolerance", 1e-10);

  // Create Newton solver and set parameters
  NewtonSolver nonlinear_solver(solver, factory);
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-8);
  nonlinear_solver.set("Newton absolute tolerance", 1e-8);

  // Solve nonlinear problem in a series of steps
  GenericVector& x = u.vector();
  double dt = 1.0; double T  = 3.0;
  while(t < T)
  {
    t += dt;
    nonlinear_solver.solve(nonlinear_problem, x);
  }

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
