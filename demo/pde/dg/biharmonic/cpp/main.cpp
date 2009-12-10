// Copyright (C) 2009 Kristian Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-06-26
// Last changed: 2009-10-05
//
// This demo program solves the Biharmonic equation,
//
//     - nabla^4 u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 4 pi^4 sin(pi*x)*sin(pi*y)
//
// and boundary conditions given by
//
//     u(x, y)     = 0
//     nabla^2 u(x, y) = 0
//
// using a discontinuous Galerkin formulation (interior penalty method).

#include <dolfin.h>
#include "Biharmonic.h"

using namespace dolfin;

// Source term
class Source : public Expression
{
public:

  void eval(Array<double>& values, const Array<const double>& x) const
  {
    values[0] = 4.0 * DOLFIN_PI * DOLFIN_PI * DOLFIN_PI * DOLFIN_PI * sin(DOLFIN_PI*x[0]) * sin(DOLFIN_PI*x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int main()
{
  // Create mesh
  UnitSquare mesh(32, 32);

  // Use uBLAS
  parameters["linear_algebra_backend"] = "uBLAS";

  // Create functions
  Source f;
  CellSize h(mesh);
  Constant alpha(8.0);

  // Create funtion space
  Biharmonic::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define forms and attach functions
  Biharmonic::BilinearForm a(V, V);
  Biharmonic::LinearForm L(V);
  a.h = h; a.alpha = alpha; L.f = f;

  // Create PDE
  VariationalProblem pde(a, L, bc);
  pde.parameters["symmetric"] = true;

  // Solve PDE
  Function u(V);
  pde.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("biharmonic.pvd");
  file << u;

  return 0;
}
