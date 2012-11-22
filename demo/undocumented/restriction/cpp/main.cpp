// Copyright (C) 2012 Anders Logg
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
// First added:  2012-10-12
// Last changed: 2012-11-22

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Subdomain for domain (restriction)
class Domain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (x[0] > 0.25 - DOLFIN_EPS &&
            x[0] < 0.75 + DOLFIN_EPS &&
            x[1] > 0.25 - DOLFIN_EPS &&
            x[1] < 0.75 + DOLFIN_EPS);
  }
};

// Subdomain for boundary (left part of restriction)
class Boundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (std::abs(x[0] - 0.25) < DOLFIN_EPS &&
            x[1] > 0.25 - DOLFIN_EPS &&
            x[1] < 0.75 + DOLFIN_EPS);
  }
};

int main()
{
  // FIXME: Does not yet run in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    info("Sorry, this demo does not yet run in parallel.");
    return 0;
  }

  // Create mesh
  UnitSquare mesh(32, 32);

  // Define restriction
  Domain domain;
  Restriction restriction(mesh, domain);

  // Create restricted function space
  Poisson::FunctionSpace V(restriction);

  // Create forms and attach coefficients
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Constant f(100.0);
  L.f = f;

  // Define boundary condition
  Constant zero(0.0);
  Boundary boundary;
  DirichletBC bc(V, zero, boundary);

  // Compute solution
  Function u(V);
  solve(a == L, u, bc);

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
