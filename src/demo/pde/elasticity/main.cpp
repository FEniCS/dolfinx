// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2007-04-24
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
  // Dirichlet boundary condition for clamp at left end
  class Clamp : public Function
  {
  public:

    Clamp(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

  };

  // Sub domain for clamp at left end
  class Left : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return x[0] < 0.5 && on_boundary;
    }
  };

  // Dirichlet boundary condition for rotation at right end
  class Rotation : public Function
  {
  public:

    Rotation(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      // Center of rotation
      real y0 = 0.5;
      real z0 = 0.219;
      
      // Angle of rotation (30 degrees)
      real theta = 0.5236;
      
      // New coordinates
      real y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta);
      real z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta);
      
      // Clamp at right end
      values[0] = 0.0;
      values[1] = y - x[1];
      values[2] = z - x[2];
    }

  };

  // Sub domain for rotation at right end
  class Right : public SubDomain
  {
    bool inside(const real* x, bool on_boundary) const
    {
      return x[0] > 0.9 && on_boundary;
    }
  };

  // Read mesh
  Mesh mesh("../../../../data/meshes/gear.xml.gz");
  
  // Create right-hand side
  Function f(mesh, 0.0);

  // Set up boundary condition at left end
  Clamp c(mesh);
  Left left;
  BoundaryCondition bcl(c, mesh, left);

  // Set up boundary condition at right end
  Rotation r(mesh);
  Right right;
  BoundaryCondition bcr(r, mesh, right);

  // Set up boundary conditions
  Array<BoundaryCondition*> bcs;
  bcs.push_back(&bcl);
  bcs.push_back(&bcr);

  // Set up PDE
  ElasticityBilinearForm a;
  ElasticityLinearForm L(f);
  LinearPDE pde(a, L, mesh, bcs);

  // Solve PDE (using direct solver)
  Function u;
  pde.set("PDE linear solver", "direct");
  pde.solve(u);

  // Plot solution
  plot(u, "displacement");

  // Save solution to VTK format
  File vtk_file("elasticity.pvd");
  vtk_file << u;

  // Save solution to XML format
  File xml_file("elasticity.xml");
  xml_file << u;

  /*
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
