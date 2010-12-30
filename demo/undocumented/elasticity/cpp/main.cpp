// Copyright (C) 2006-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2008
//
// First added:  2006-02-07
// Last changed: 2010-09-02
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#include <dolfin.h>
#include "Elasticity.h"

using namespace dolfin;

int main()
{
  // Dirichlet boundary condition for clamp at left end
  class Clamp : public Expression
  {
  public:

    Clamp() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

  };

  // Sub domain for clamp at left end
  class Left : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] < 0.5 && on_boundary;
    }
  };

  // Dirichlet boundary condition for rotation at right end
  class Rotation : public Expression
  {
  public:

    Rotation() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      // Center of rotation
      const double y0 = 0.5;
      const double z0 = 0.219;

      // Angle of rotation (30 degrees)
      const double theta = 0.5236;

      // New coordinates
      const double y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta);
      const double z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta);

      // Clamp at right end
      values[0] = 0.0;
      values[1] = y - x[1];
      values[2] = z - x[2];
    }

  };

  // Sub domain for rotation at right end
  class Right : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] > 0.9 && on_boundary;
    }
  };

  //parameters["mesh_partitioner"] = "ParMETIS";
  parameters["mesh_partitioner"] = "SCOTCH";

  //parameters["linear_algebra_backend"] = "Epetra";
  parameters["linear_algebra_backend"] = "PETSc";

  // Read mesh and create function space
  Mesh mesh("gear.xml.gz");
  Elasticity::FunctionSpace V(mesh);

  // Create right-hand side
  Constant f(0.0, 0.0, 0.0);

  // Set up boundary condition at left end
  Clamp c;
  Left left;
  //DirichletBC bcl(V, c, left);
  DirichletBC bcl(V, f, left);

  // Set up boundary condition at right end
  Rotation r;
  Right right;
  //DirichletBC bcr(V, r, right);
  DirichletBC bcr(V, f, right);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bcl);
  bcs.push_back(&bcr);

  // Set elasticity parameters
  double E  = 10.0;
  double nu = 0.3;
  Constant mu(E / (2*(1 + nu)));
  Constant lambda(E*nu / ((1 + nu)*(1 - 2*nu)));

  // Set up PDE (symmetric)
  Elasticity::BilinearForm a(V, V);
  a.mu = mu; a.lmbda = lambda;
  Elasticity::LinearForm L(V);
  L.f = f;
  Matrix A;
  Vector b;
  assemble_system(A, b, a, L, bcl);
  std::cout.precision(15);
  std::cout << "Matrix norm: " << A.norm("frobenius") << std::endl;
  std::cout << "Vector norm: " << b.norm("l2") << std::endl;

  //cout <<  " --  " << endl;

  //assemble(A, a);
  //assemble(b, L);
  //std::cout.precision(15);
  //std::cout << "Matrix norm (2): " << A.norm("frobenius") << std::endl;
  //std::cout << "Vector norm (2): " << b.norm("l2") << std::endl;
  /*
  VariationalProblem problem(a, L, bcs);
  problem.parameters["symmetric"] = true;

  // Solve PDE (using direct solver)
  Function u(V);
  problem.parameters["linear_solver"] = "direct";
  problem.solve(u);

  Function ux = u[0];
  Function uy = u[1];
  Function uz = u[2];
  cout << "Norm (u): " << u.vector().norm("l2") << endl;
  cout << "Norm (ux, uy, uz): " << ux.vector().norm("l2") << "  "
                                   << uy.vector().norm("l2") << "  "
                                   << uz.vector().norm("l2") << endl;

  // Save solution in VTK format
  File vtk_file("elasticity.pvd", "compressed");
  vtk_file << u;

  // Save colored mesh paritions in VTK format if running in parallel
  if (MPI::num_processes() > 1)
  {
    CellFunction<dolfin::uint> partitions(mesh, MPI::process_number());
    File file("partitions.pvd");
    file << partitions;
  }

  // Plot solution
  plot(u, "Displacement", "displacement");

  // Displace mesh and plot displaced mesh
  mesh.move(u);
  plot(mesh, "Deformed mesh");
  */
 return 0;
}
