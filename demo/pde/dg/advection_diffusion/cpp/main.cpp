// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2008-12-12
//
// Steady state advection-diffusion equation, discontinuous
// formulation using full upwinding.

#include <dolfin.h>

#include "AdvectionDiffusion.h"
#include "OutflowFacet.h"
#include "Projection.h"

using namespace dolfin;

// Dirichlet boundary condition
class BC : public Function
{
  void eval(double* values, const double* x) const
  {
    values[0] = sin(DOLFIN_PI*5.0*x[1]);
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

// Advective velocity
class Velocity : public Function
{
  void eval(double* values, const double* x) const
  {
    values[0] = -1.0;
    values[1] = -0.4;
  }

  dolfin::uint rank() const
  { return 1; }

  dolfin::uint dim(dolfin::uint i) const
  { return 2; }
};

int main(int argc, char *argv[])
{
  // Read simple velocity field (-1.0, -0.4) defined on a 64x64 unit square 
  // mesh and a quadratic vector Lagrange element
  Function velocity("../velocity.xml.gz");
  const Mesh& mesh = velocity.function_space().mesh();

  // Diffusivity
  Constant c(0.0); 

  //Source term
  Constant f(0.0);

  // Mesh-related functions
  FacetNormal N;
  AvgMeshSize h;

  // Definitions for outflow facet function (use to define flux upwinding)
  OutflowFacetFunctional M_of;
  M_of.velocity = velocity;
  M_of.n = N;

  // Outflow facet function From SpecialFunctions.h
  OutflowFacet of(M_of); 

  // Penalty parameter
  Constant alpha(5.0);

  // Create function space
  AdvectionDiffusionFunctionSpace V(mesh);

  // Create forms and attach functions
  AdvectionDiffusionBilinearForm a(V, V);
  a.b = velocity; a.n = N; a.h = h; a.of = of; a.kappa = c; a.alpha = alpha;
  AdvectionDiffusionLinearForm L(V);
  L.f = f;

  // Set up boundary condition (apply strong BCs)
  BC g;
  DirichletBoundary boundary;
  DirichletBC bc(V, g, boundary, geometric);

  // Solution function
  Function uh(V);

  // Assemble and apply boundary conditions
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);

  // Solve system
  solve(A, uh.vector(), b);

  // Define PDE for projection onto continuous P1 basis
  ProjectionFunctionSpace Vp(mesh);
  ProjectionBilinearForm ap(Vp, Vp);
  ProjectionLinearForm Lp(Vp);
  Lp.u0 = uh;
  LinearPDE pde(ap, Lp);

  // Compute projection
  Function up;
  pde.solve(up);

  // Save projected solution in VTK format
  File file("temperature.pvd");
  file << up;

  // Plot projected solution
  plot(up);
}
