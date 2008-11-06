// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2008-07-15
//
// Steady state advection-diffusion equation, discontinuous formulation using full upwinding.

#include <dolfin.h>

#include "AdvectionDiffusion.h"
//#include "OutflowFacet.h"
#include "Projection.h"

using namespace dolfin;

// Dirichlet boundary condition
class BC : public Function
{
public:

  void eval(double* values, const Data& data) const
  {
    real y = data.x[1];
    values[0] = sin(DOLFIN_PI*5.0*y);
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
public:
    
  void eval(double* values, const Data& data) const
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
  Mesh& mesh = const_cast<Mesh&>(velocity.function_space().mesh());

  // Set up problem
  Matrix A;
  Vector b;
  Constant c(0.0); // Diffusivity constant
  Constant f(0.0); // Source term

  FacetNormal N;
  AvgMeshSize h;

  // Definitions for outflow facet function
  //OutflowFacetFunctional M_of(velocity, N);
  //OutflowFacetFunctional M_of(velocity, N);
  //M_of.velocity = velocity;
  //M_of.n = N;

  // From SpecialFunctions.h
  //OutflowFacet of(M_of); 
  error("FFC wrapper code for functionals needs to be fixed");
  Constant of(0.0); 

  // Penalty parameter
  Constant alpha(20.0);

  AdvectionDiffusionFunctionSpace V(mesh);
  AdvectionDiffusionBilinearForm a(V, V);
  a.b  = velocity;
  a.n  = N;
  a.h  = h;
  a.of = of;
  a.kappa = c;
  a.alpha = alpha;

  AdvectionDiffusionLinearForm L(V);
  L.f = f;

  // Solution function
  Function uh(V);

  // Set up boundary condition (apply strong BCs)
  BC g;
  DirichletBoundary boundary;
  DirichletBC bc(g, V, boundary, geometric);

  Assembler::assemble(A, a);
  Assembler::assemble(b, L);
  bc.apply(A, b);

  GenericVector& x = uh.vector(); 
  solve(A, x, b);

  // Define PDE for projection
  ProjectionFunctionSpace Vp(mesh);
  ProjectionBilinearForm ap(Vp, Vp);
  ProjectionLinearForm Lp(Vp);
  Lp.u0 = uh;
  LinearPDE pde(ap, Lp);

  // Solve PDE
  Function up(Vp);
  pde.solve(up);

  // Save projected solution
  File file("temperature.pvd");
  file << up;

  // Plot projected solution
  plot(up);
}
