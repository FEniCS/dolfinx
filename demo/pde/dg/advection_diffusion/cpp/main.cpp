// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2008-07-15
//
// Steady state advection-diffusion equation, discontinuous formulation using full upwinding.

#include <dolfin.h>

#include "AdvectionDiffusion.h"
#include "OutflowFacet.h"
#include "Projection.h"

using namespace dolfin;

// Dirichlet boundary condition
class BC : public Function
{
public:

  BC(Mesh& mesh) : Function(mesh) {}

  real eval(const real* x) const
  {
    return sin(DOLFIN_PI*5.0*x[1]);
  }
};

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
    }
  };

// Advective velocity
class Velocity : public Function
{
public:
    
  Velocity(Mesh& mesh) : Function(mesh) {}

  void eval(real* values, const real* x) const
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
  // Read simple velocity field (-1.0, -0.4)
  // defined on a 64x64 unit square mesh and a quadratic vector Lagrange element
  Function velocity("../velocity.xml.gz");

  UnitSquare mesh(64, 64);

  // Set up problem
  Matrix A;
  Vector b;
  Function c(mesh, 0.0); // Diffusivity constant
  Function f(mesh, 0.0); // Source term

  FacetNormal N(mesh);
  AvgMeshSize h(mesh);

  // Definitions for outflow facet function
  OutflowFacetFunctional M_of(velocity, N);
  OutflowFacet of(mesh, M_of); // From SpecialFunctions.h

  // Penalty parameter
  Function alpha(mesh, 20.0);

  AdvectionDiffusionBilinearForm a(velocity, N, h, of, c, alpha);
  AdvectionDiffusionLinearForm L(f);

  // Solution function
  Function uh(mesh, a);

  // Set up boundary condition (apply strong BCs)
  BC g(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(g, mesh, boundary, geometric);

  assemble(A, a, mesh);
  assemble(b, L, mesh);
  bc.apply(A, b, a);

  GenericVector& x = uh.vector(); 
  solve(A, x, b);

  // Define PDE for projection
  ProjectionBilinearForm ap;
  ProjectionLinearForm Lp(uh);
  LinearPDE pde(ap, Lp, mesh);

  // Solve PDE
  Function up;
  pde.solve(up);

  // Save projected solution
  File file("temperature.pvd");
  file << up;

  // Plot projected solution
  plot(up);
}
