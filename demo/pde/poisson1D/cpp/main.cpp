// Copyright (C) 2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-23
// Last changed: 2009-10-05
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x) = f(x)
//
// on the unit interval with source f given by
//
//     f(x) = 9.0*DOLFIN_PI*DOLFIN_PI*sin(3.0*DOLFIN_PI*x[0]);
//
// and boundary conditions given by
//
//     u(x) = 0 for x = 0,
//    du/dx = 0 for x = 1.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return (std::abs(x[0]) < DOLFIN_EPS);
  }
};

// Source term
class Source : public Expression
{
public:

  Source() : Expression() {}

  void eval(Array<double>& values, const Array<const double>& x) const
  {
    values[0] = 9.0*DOLFIN_PI*DOLFIN_PI*sin(3.0*DOLFIN_PI*x[0]);
  }

};

// Neumann boundary condition
class Flux : public Expression
{
public:

  Flux() : Expression() {}

  void eval(Array<double>& values, const Array<const double>& x) const
  {
    values[0] = 3.0*DOLFIN_PI*cos(3.0*DOLFIN_PI*x[0]);
  }
};

int main()
{
  // Create mesh
  UnitInterval mesh(50);

  // Create function space
  Poisson::FunctionSpace V(mesh);

  // Set up BCs
  Constant zero(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, zero, boundary);

  // Create source and flux terms
  Source f;
  Flux g;

  // Define PDE
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;
  L.g = g;
  VariationalProblem pde(a, L, bc);

  // Solve PDE
  Function u(V);
  pde.solve(u);

  // Save solution in VTK format
  File file_u("poisson.pvd");
  file_u << u;

  // Plot solution
  plot(u);

  return 0;
}
