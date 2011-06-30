// Copyright (C) 2006-2009 Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2008
//
// First added:  2006-02-07
// Last changed: 2011-01-17
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
      return x[0] < 0.5;// && on_boundary;
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
      return x[0] > 0.9;// && on_boundary;
    }
  };

  // Read mesh and create function space
  Mesh mesh("gear.xml.gz");
  Elasticity::FunctionSpace V(mesh);

  // Create right-hand side
  Constant f(0.0, 0.0, 0.0);

  // Set up boundary condition at left end
  Clamp c;
  Left left;
  DirichletBC bcl(V, c, left);

  // Set up boundary condition at right end
  Rotation r;
  Right right;
  DirichletBC bcr(V, r, right);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bcl);
  bcs.push_back(&bcr);

  std::vector<const DirichletBC*> _bcs;
  _bcs.push_back(&bcl);
  _bcs.push_back(&bcr);

  // Set elasticity parameters
  double E  = 10.0;
  double nu = 0.3;
  Constant mu(E / (2*(1 + nu)));
  Constant lambda(E*nu / ((1 + nu)*(1 - 2*nu)));

  // Define variational problem
  Elasticity::BilinearForm a(V, V);
  a.mu = mu; a.lmbda = lambda;
  Elasticity::LinearForm L(V);
  L.f = f;
  Function u(V);
  LinearVariationalProblem problem(a, L, u, bcs);

  // Compute solution
  LinearVariationalSolver solver(problem);
  solver.parameters["symmetric"] = true;
  solver.parameters["linear_solver"] = "direct";
  solver.solve();

  // Extract solution components
  Function ux = u[0];
  Function uy = u[1];
  Function uz = u[2];
  std::cout << "Norm (u): " << u.vector().norm("l2") << std::endl;
  std::cout << "Norm (ux, uy, uz): " << ux.vector().norm("l2") << "  "
            << uy.vector().norm("l2") << "  "
            << uz.vector().norm("l2") << std::endl;

  // Save solution in VTK format
  File vtk_file("elasticity.pvd", "compressed");
  vtk_file << u;

  // Save colored mesh paritions in VTK format if running in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    CellFunction<dolfin::uint> partitions(mesh, dolfin::MPI::process_number());
    File file("partitions.pvd");
    file << partitions;
  }

  // Plot solution
  plot(u, "Displacement", "displacement");

  // Displace mesh and plot displaced mesh
  mesh.move(u);
  plot(mesh, "Deformed mesh");

  // Write boundary condition facets markers to VTK format
  MeshFunction<dolfin::uint> facet_markers(mesh, 2, 0);
  left.mark(facet_markers, 1);
  right.mark(facet_markers, 2);
  File facet_file("facet_markers.pvd");
  facet_file << facet_markers;

 return 0;
}
