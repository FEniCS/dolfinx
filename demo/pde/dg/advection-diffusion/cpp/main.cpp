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

class OutflowFacet : public Function
{
public:

  // Constructor
  OutflowFacet(const Form& form): form(form), 
				  V(form.function_spaces()), ufc(form)                            
  {
    // Some simple sanity checks on form
    if (!(form.rank() == 0 && form.ufc_form().num_coefficients() == 2))
      error("Invalid form: rank = %d, number of coefficients = %d. Must be rank 0 form with 2 coefficients.", 
	    form.rank(), form.ufc_form().num_coefficients());
    
    if (!(form.ufc_form().num_cell_integrals() == 0 && form.ufc_form().num_exterior_facet_integrals() == 1 
	  && form.ufc_form().num_interior_facet_integrals() == 0))
      error("Invalid form: Must have exactly 1 exterior facet integral");
  }

  ~OutflowFacet(){}

  void eval(double* values, const Data& data) const
  {
    // If there is no facet (assembling on interior), return 0.0
    if (!data.on_facet())
    {
      values[0] = 0.0;
      return;
    }
    else
    {
      ufc.update( data.cell() );
      
      // Interpolate coefficients on cell and current facet
      for (dolfin::uint i = 0; i < form.coefficients().size(); i++)
        form.coefficient(i).interpolate(ufc.w[i], ufc.cell, data.facet());
  
      // Get exterior facet integral (we need to be able to tabulate ALL facets 
      // of a given cell)
      ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];
  
      // Call tabulate_tensor on exterior facet integral, 
      // dot(velocity, facet_normal)
      integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, data.facet());
    }
    
    // If dot product is positive, the current facet is an outflow facet
    if (ufc.A[0] > DOLFIN_EPS)
      values[0] = 1.0;
    else
      values[0] = 0.0;
  }
  
private:

  const Form& form;
  std::vector<const FunctionSpace*> V;
  mutable UFC ufc;
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
