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
// Last changed: 2012-07-05
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with homogeneous Dirichlet boundary conditions
// at y = 0, 1 and periodic boundary conditions at x = 0, 1.

#include <dolfin.h>
#include <dolfin/fem/AssemblerTools.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  //parameters["linear_algebra_backend"] = "Epetra";

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
      return x[0] < DOLFIN_EPS && x[0] > -DOLFIN_EPS && on_boundary;
    }

    // Map right boundary (H) to left boundary (G)
    void map(const Array<double>& x, Array<double>& y) const
    {
      y[0] = x[0] - 1.0;
      y[1] = x[1];
    }
  };

  // Create mesh
  UnitSquare mesh(32, 32);

  // Create functions
  Source f;

  // Define PDE
  Poisson::FunctionSpace V(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Create Dirichlet boundary condition
  Constant u0(0.0);
  DirichletBoundary dirichlet_boundary;
  DirichletBC bc0(V, u0, dirichlet_boundary);

  // Create periodic boundary condition
  PeriodicBoundary periodic_boundary;
  PeriodicBC bc1(V, periodic_boundary);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc0);
  bcs.push_back(&bc1);

  // Compute solution
  Function u(V);

  boost::shared_ptr<GenericMatrix> A(new Matrix);
  Vector b;

  // Get list of master-slave dofs
  std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > > dof_pairs;
  bc1.compute_dof_pairs(V, dof_pairs);

  // Intialise tensor, taking into account periodic dofs
  AssemblerTools::init_global_tensor(*A, a, dof_pairs, true, false);

  assemble(*A, a, false);
  assemble(b, L);

  for (uint i = 0; i < bcs.size(); ++i)
    bcs[i]->apply(*A, b);

  LUSolver lu(A);
  lu.solve(*u.vector(), b);

  //solve(a == L, u, bcs);

  // Save solution in VTK format
  File file("periodic.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
