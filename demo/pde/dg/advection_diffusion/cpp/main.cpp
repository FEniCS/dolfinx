// Copyright (C) 2007-2008 Kristian B. Oelgaard, Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-06-29
// Last changed: 2008-07-15
//
// Steady state advection-diffusion equation, discontinuous formulation using full upwinding.

#include <dolfin.h>
#include <dolfin/fem/UFC.h>

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

// Determine if the current facet is an outflow facet with respect to the current cell
class OutflowFacet : public Function
{
public:

  OutflowFacet(Function& velocity, Mesh& mesh, UFC& ufc, Form& form) : Function(mesh), velocity(velocity), ufc(ufc), form(form) {}

  real eval(const real* x) const
  {
    // If there is no facet (assembling on interior), return 0.0
    if (facet() < 0)
      return 0.0;
    else
    {
      // Copy cell, cannot call interpolate with const cell()
      Cell cell0(cell());
      ufc.update(cell0);

      // Interpolate coefficients on cell and current facet
      for (dolfin::uint i = 0; i < form.coefficients().size(); i++)
        form.coefficients()[i]->interpolate(ufc.w[i], ufc.cell, *ufc.coefficient_elements[i], cell0, facet());

      // Get exterior facet integral (we need to be able to tabulate ALL facets of a given cell)
      ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];

      // Call tabulate_tensor on exterior facet integral, dot(velocity, facet_normal)
      integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, facet());
    }

    // If dot product is positive, the current facet is an outflow facet
    if (ufc.A[0] > DOLFIN_EPS)
    {
      return 1.0;
    }
    else
      return 0.0;
  }

private:

  Function& velocity;
  UFC& ufc;
  Form& form;

};

int main(int argc, char *argv[])
{
  // Read simple velocity field (-1.0, -0.4)
  // defined on a 64x64 unit square mesh and a quadratic vector Lagrange element
  Function velocity("../velocity.xml.gz");

  UnitSquare mesh(64, 64);

  // Set up problem
  Matrix A;
  Vector x, b;
  Function c(mesh, 0.0); // Diffusivity constant
  Function f(mesh, 0.0); // Source term

  FacetNormal N(mesh);
  AvgMeshSize h(mesh);

  // Definitions for outflow facet function
  OutflowFacetFunctional M_of(velocity, N);
  M_of.updateDofMaps(mesh);
  UFC ufc(M_of.form(), mesh, M_of.dofMaps());
  OutflowFacet of(velocity, mesh, ufc, M_of);

  // Penalty parameter
  Function alpha(mesh, 20.0);

  AdvectionDiffusionBilinearForm a(velocity, N, h, of, c, alpha);
  AdvectionDiffusionLinearForm L(f);

  // Set up boundary condition (apply strong BCs)
  BC g(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(g, mesh, boundary, geometric);

  assemble(A, a, mesh);
  assemble(b, L, mesh);
  bc.apply(A, b, a);

  solve(A, x, b);

  // Discontinuous solution
  Function uh(mesh, x, a);

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
