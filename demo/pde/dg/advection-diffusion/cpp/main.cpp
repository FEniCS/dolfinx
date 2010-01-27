// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2010-01-27
//
// Steady state advection-diffusion equation, discontinuous
// formulation using full upwinding.

#include <dolfin.h>

#include "AdvectionDiffusion.h"
#include "Projection.h"
#include "Velocity.h"

using namespace dolfin;

// Dirichlet boundary condition
class BC : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
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

int main(int argc, char *argv[])
{
  // Read simple velocity field (-1.0, -0.4) defined on a 64x64 unit square
  // mesh and a quadratic vector Lagrange element

  // Read mesh
  Mesh mesh("../mesh.xml.gz");

  // Create velocity FunctionSpace
  Velocity::FunctionSpace V_u(mesh);

  // Create velocity function
  Function u(V_u, "../velocity.xml.gz");

  // Diffusivity
  Constant c(0.0);

  //Source term
  Constant f(0.0);

  // Mesh-related functions
  //CellSize h(mesh);
  Constant h(10.0);

  // Penalty parameter
  Constant alpha(5.0);

  // Create function space
  AdvectionDiffusion::FunctionSpace V(mesh);

  // Create forms and attach functions
  AdvectionDiffusion::BilinearForm a(V, V);
  a.u = u; a.h = h; a.kappa = c; a.alpha = alpha;
  AdvectionDiffusion::LinearForm L(V);
  L.f = f;

  // Set up boundary condition (apply strong BCs)
  BC g;
  DirichletBoundary boundary;
  DirichletBC bc(V, g, boundary, "geometric");

  // Solution function
  Function phi_h(V);

  // Assemble and apply boundary conditions
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);

  // Solve system
  solve(A, phi_h.vector(), b);

  // Define PDE for projection onto continuous P1 basis
  Projection::FunctionSpace Vp(mesh);
  Projection::BilinearForm ap(Vp, Vp);
  Projection::LinearForm Lp(Vp);
  Lp.phi0 = phi_h;
  VariationalProblem pde(ap, Lp);

  // Compute projection
  Function phi_p(Vp);
  pde.solve(phi_p);

  // Save projected solution in VTK format
  File file("temperature.pvd");
  file << phi_p;

  // Plot projected solution
  plot(phi_h);
}
