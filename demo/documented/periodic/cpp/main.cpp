// Copyright (C) 2007-2008 Anders Logg
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
// First added:  2007-07-11
// Last changed: 2012-11-12
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with homogeneous Dirichlet boundary conditions
// at y = 0, 1 and periodic boundary conditions at x = 0, 1.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Expression
  {
  public:

    void eval(Array<double>& values, const Array<double>& x) const
    {
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) + 1.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return (x[1] < DOLFIN_EPS || x[1] > (1.0 - DOLFIN_EPS)) && on_boundary;
    }
  };

  // Sub domain for Periodic boundary condition
  class PeriodicBoundary : public SubDomain
  {
    // Left boundary is "target domain" G
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return (std::abs(x[0]) < DOLFIN_EPS);
    }

    // Map right boundary (H) to left boundary (G)
    void map(const Array<double>& x, Array<double>& y) const
    {
      y[0] = x[0] - 1.0;
      y[1] = x[1];
    }
  };

  // Create mesh
  UnitSquareMesh mesh(32, 32);

  // Create periodic boundary condition
  PeriodicBoundary periodic_boundary;

  /*
  // Create vertex mast-slave map
  const std::map<std::size_t, std::pair<std::size_t, std::size_t> > periodic_vertex_pairs
    = PeriodicBoundaryComputation::compute_periodic_pairs(mesh, periodic_boundary, 0);

  // Creat MehsFunction marking periodic boundary conditions for plotting
  MeshFunction<std::size_t> master_slave_entities(mesh, 0, 0);
  periodic_boundary.mark(master_slave_entities, 1);
  std::map<std::size_t, std::pair<std::size_t, std::size_t> >::const_iterator it;
  for (it = periodic_vertex_pairs.begin(); it != periodic_vertex_pairs.end(); ++it)
    master_slave_entities[it->first] = 2;
  File file("markers.pvd");
  file << master_slave_entities;

  // Attach periodic vertex pairs to mesh
  mesh.periodic_index_map[0] = periodic_vertex_pairs;
  */

  // Create functions
  Source f;

  // Define PDE
  Poisson::FunctionSpace V(mesh, periodic_boundary);
  //Poisson::FunctionSpace V(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Create Dirichlet boundary condition
  Constant u0(0.0);
  DirichletBoundary dirichlet_boundary;
  DirichletBC bc0(V, u0, dirichlet_boundary);

  // Collect boundary conditions
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc0);

  // Compute solution
  Function u(V);
  solve(a == L, u, bcs);

  cout << "Solution vector norm: " << u.vector()->norm("l2") << endl;

  // Save solution in VTK format
  File file_u("periodic.pvd");
  file_u << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
