// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2008-05-05
//
// Steady state advection-diffusion equation, discontinuous formulation using full upwinding.
// Constant velocity field with homogeneous Dirichlet boundary conditions on all boundaries.

#include <dolfin.h>
#include "AdvectionDiffusion.h"
#include "Projection.h"

using namespace dolfin;

// Source term
class Source2D : public Function
{
public:
    
  Source2D(Mesh& mesh, real c) : Function(mesh), c(c) {}

  real eval(const real* x) const
  {

    real vx  = -exp(x[0])*(x[1]*cos(x[1]) + sin(x[1]));
    real vy  =  exp(x[0])*(x[1]*sin(x[1]));

    real ux  =  5.0*DOLFIN_PI*cos(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);
    real uy  =  5.0*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*cos(5.0*DOLFIN_PI*x[1]);
    real uxx = -25.0*DOLFIN_PI*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);
    real uyy = -25.0*DOLFIN_PI*DOLFIN_PI*sin(5.0*DOLFIN_PI*x[0])*sin(5.0*DOLFIN_PI*x[1]);

    return vx*ux + vy*uy - c*(uxx + uyy);
  }

private:

  real c;
};

// Advective velocity
class Velocity : public Function
{
public:
    
  Velocity(Mesh& mesh) : Function(mesh) {}

  void eval(real* values, const real* x) const
  {
    values[0] = -exp(x[0])*(x[1]*cos(x[1]) + sin(x[1]));
    values[1] =  exp(x[0])*(x[1]*sin(x[1]));
  }

  dolfin::uint rank() const
  { return 1; }

  dolfin::uint dim(dolfin::uint i) const
  { return 2; }
};

class OutflowFacet : public Function
{
public:

  OutflowFacet(Function& velocity, Mesh& mesh) : Function(mesh), velocity(velocity) {}

  real eval(const real* x) const
  {
    // If there is no facet (assembling on interior), return 0.0
    if (facet() < 0)
      return 0.0;
    else
    {
      real normal_vector[2];
      real velocities[2] = {0.0, 0.0};

      // Compute facet normal
      for (dolfin::uint i = 0; i < cell().dim(); i++)
        normal_vector[i] = cell().normal(facet(), i);

      // Get velocities
      velocity.eval(velocities, x);

      // Compute dot product of the facet outward normal and the velocity vector
      real dot = 0.0;
      for (dolfin::uint i = 0; i < cell().dim(); i++)
        dot += normal_vector[i]*velocities[i];

      // If dot product is positive the facet is an outflow facet, meaning the contribution
      // from this cell is on the upwind side.
      if (dot > DOLFIN_EPS)
        return 1.0;
      else
        return 0.0;
    }
  }

private:

  Function& velocity;
};

int main(int argc, char *argv[])
{
  // Set up problem
  Matrix A;
  Vector b;
  Vector x;

  UnitSquare mesh(64, 64);
  Velocity velocity(mesh);
  Source2D f(mesh, 0.0);
  Function c(mesh, 0.0); // Diffusivity constant
  FacetNormal N(mesh);
  AvgMeshSize h(mesh);
  OutflowFacet of(velocity, mesh);

  Function alpha(mesh, 20.0);

  AdvectionDiffusionBilinearForm a(velocity, N, h, of, c, alpha);
  AdvectionDiffusionLinearForm L(f);

  assemble(A, a, mesh);
  assemble(b, L, mesh);
  solve(A, x, b);

  Function uh(mesh, x, a);

  // Define PDE
  ProjectionBilinearForm ap;
  ProjectionLinearForm Lp(uh);
  LinearPDE pde(ap, Lp, mesh);

  // Solve PDE
  Function up;
  pde.solve(up);

  File file("temperature.pvd");
  file << up;
}
