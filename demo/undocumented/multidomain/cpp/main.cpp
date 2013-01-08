// Copyright (C) 2013 Anders Logg
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
// First added:  2013-01-08
// Last changed: 2013-01-08

#include <dolfin.h>
#include "MultiDomainPoisson.h"

using namespace dolfin;

// Subdomain for domain 1
class Domain1 : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < 0.5 + DOLFIN_EPS;
  }
};

// Subdomain for domain 1
class Domain2 : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 0.5 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh
  UnitSquareMesh mesh(8, 8);

  // Define restricted domains
  Domain1 D1;
  Domain1 D2;
  Restriction R1(mesh, D1);
  Restriction R2(mesh, D2);

  // Create restricted function space
  MultiDomainPoisson::FunctionSpace W(R1);

  // Create forms and attach coefficients
  MultiDomainPoisson::BilinearForm a(W, W);
  MultiDomainPoisson::LinearForm L(W);
  Constant k1(0.1);
  Constant k2(0.2);
  Constant f(100.0);
  a.k1 = k1;
  a.k2 = k2;
  L.f1 = f;
  L.f2 = f;

  // Define boundary condition
  //Constant zero(0.0, 0.0);
  //DomainBoundary boundary;
  //DirichletBC bc(W, zero, boundary);

  // Compute solution
  Function u(W);
  solve(a == L, u);

  // Extract components
  //Function& u1 = u[0];
  //Function& u2 = u[1];

  // Plot solution
  //plot(u1, "u1");
  //plot(u2, "u2");
  plot(u);
  interactive();

  return 0;
}
