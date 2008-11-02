// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2008
//
// First added:  2006-02-07
// Last changed: 2008-05-21
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#include <dolfin.h>
#include "Elasticity.h"

using namespace dolfin;

int main()
{
  class Zero : public Function
  {
  public:

    Zero(const FunctionSpace& V) : Function(V) {}

    void eval(double* values, const double* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }
  };

  // Dirichlet boundary condition for clamp at left end
  class Clamp : public Function
  {
  public:

    Clamp(const FunctionSpace& V) : Function(V) {}

    void eval(double* values, const double* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }
  };

  // Sub domain for clamp at left end
  class Left : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] < 0.5 && on_boundary;
    }
  };

  // Dirichlet boundary condition for rotation at right end
  class Rotation : public Function
  {
  public:

    Rotation(const FunctionSpace& V) : Function(V) {}

    void eval(double* values, const double* x) const
    {
      // Center of rotation
      double y0 = 0.5;
      double z0 = 0.219;
      
      // Angle of rotation (30 degrees)
      double theta = 0.5236;
      
      // New coordinates
      double y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta);
      double z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta);
      
      // Clamp at right end
      values[0] = 0.0;
      values[1] = y - x[1];
      values[2] = z - x[2];
    }
  };

  // Sub domain for rotation at right end
  class Right : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] > 0.9 && on_boundary;
    }
  };

  // Read mesh
  Mesh mesh("../../../../data/meshes/gear.xml.gz");
  mesh.order();

  ElasticityFunctionSpace V(mesh);

  // Create right-hand side
  //Function f(mesh, 3, 0.0);
  Zero f(V);

  // Set up boundary condition at left end
  Clamp c(V);
  Left left;
  DirichletBC bcl(c, V, left);

  // Set up boundary condition at right end
  Rotation r(V);
  Right right;
  DirichletBC bcr(r, V, right);

  // Set up boundary conditions
  Array<DirichletBC*> bcs;
  bcs.push_back(&bcl);
  bcs.push_back(&bcr);

  // Set elasticity parameters
  double E  = 10.0;
  double nu = 0.3;
  Constant mu(E / (2*(1 + nu)));
  Constant lambda(E*nu / ((1 + nu)*(1 - 2*nu)));

  // Set up PDE (symmetric)
  ElasticityBilinearForm a(V, V);
  a.mu = mu; a.lmbda = lambda;
  ElasticityLinearForm L(V);
  L.f = f;
  LinearPDE pde(a, L, mesh, bcs, symmetric);

  // Solve PDE (using direct solver)
  Function u(V);
  pde.set("PDE linear solver", "direct");
  pde.solve(u);

  // Plot solution
  plot(u, "displacement");

  // Save solution to VTK format
  File vtk_file("elasticity.pvd");
  vtk_file << u;

  // Save solution to XML format
  File xml_file("displacement.xml");
  xml_file << u;

  return 0;
}
