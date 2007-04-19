// Copyright (C) 2006 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2006-10-18
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#include <dolfin.h>
#include "Elasticity.h"
//#include "ElasticityStrain.h"

using namespace dolfin;

int main()
{
  // Dirichlet boundary condition for rotation at right end
  class Rotation : public Function
  {
  public:

    Rotation(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, real* x)
    {
      /*
      // Center of rotation
      real y0 = 0.5;
      real z0 = 0.219;
      
      // Angle of rotation (30 degrees)
      real theta = 0.5236;
      
      // New coordinates
      real y = y0 + (p.y() - y0)*cos(theta) - (p.z() - z0)*sin(theta);
      real z = z0 + (p.y() - y0)*sin(theta) + (p.z() - z0)*cos(theta);
      
      // Clamp at left end
      value = 0.0;
      
      // Clamp at right end
      if ( p.x() > (1.0 - w) )
      {
	    if ( i == 1 )
	      value = y - p.y();
	    else if ( i == 2 )
	      value = z - p.z();
      }
      */
    }
  };

  // Sub domain for clamping at left end
  class Left : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] < 0.1 && on_boundary;
    }
  };

  // Sub domain for clamping at right end
  class Right : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] > 0.9 && on_boundary;
    }
  };

  /*
  MyBC bc;

  // Set up problem
  Mesh mesh("../../../../data/meshes/gear.xml.gz");
  Function f = 0.0;
  Elasticity::BilinearForm a;
  Elasticity::LinearForm L(f);
  PDE pde(a, L, mesh, bc);

  // Compute solution (using direct solver)
  Function U = pde.solve();

  // Save solution (displacement) to file
  File file("elasticity.pvd");
  file << U;

  // Set up post-processing problem to compute strain
  ElasticityStrain::BilinearForm a_strain;
  ElasticityStrain::LinearForm L_strain(U);
  PDE pde_strain(a_strain, L_strain, mesh);
  Function normal_strain, shear_strain;

  // Compute solution (using GMRES solver)
  pde_strain.set("PDE linear solver", "iterative");
  pde_strain.solve(normal_strain, shear_strain);

  // Save solution (strain) to files
  File file_normal_strain("normal_strain.pvd");
  File file_shear_strain("shear_strain.pvd");
  file_normal_strain << normal_strain;
  file_shear_strain  << shear_strain;
  */

  return 0;
}
