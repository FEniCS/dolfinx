// Copyright (C) 2009 Harish Narayanyan
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
// Modified by Anders Logg, 2011
//
// First added:  2009-09-29
// Last changed: 2012-11-12
//
// This demo program solves a hyperelastic problem

// Begin demo

#include <dolfin.h>
#include "HyperElasticity.h"

using namespace dolfin;

// Sub domain for clamp at left end
class Left : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (std::abs(x[0]) < DOLFIN_EPS) && on_boundary;
  }
};

// Sub domain for rotation at right end
class Right : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (std::abs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary;
  }
};

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

// Dirichlet boundary condition for rotation at right end
class Rotation : public Expression
{
public:

  Rotation() : Expression(3) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    const double scale = 0.5;

    // Center of rotation
    const double y0 = 0.5;
    const double z0 = 0.5;

    // Large angle of rotation (60 degrees)
    double theta = 1.04719755;

    // New coordinates
    double y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta);
    double z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta);

    // Rotate at right end
    values[0] = 0.0;
    values[1] = scale*(y - x[1]);
    values[2] = scale*(z - x[2]);
  }

};

int main()
{
  // Create mesh and define function space
  UnitCubeMesh mesh (24, 16, 16);
  auto V = std::make_shared<HyperElasticity::FunctionSpace>(mesh);

  // Define Dirichlet boundaries
  auto left = std::make_shared<Left>();
  auto right = std::make_shared<Right>();

  // Define Dirichlet boundary functions
  auto c = std::make_shared<Clamp>();
  auto r = std::make_shared<Rotation>();

  // Create Dirichlet boundary conditions
  DirichletBC bcl(V, c, left);
  DirichletBC bcr(V, r, right);
  std::vector<const DirichletBC*> bcs = {{&bcl, &bcr}};

  // Define source and boundary traction functions
  Constant B(0.0, -0.5, 0.0);
  Constant T(0.1,  0.0, 0.0);

  // Define solution function
  Function u(V);

  // Set material parameters
  const double E  = 10.0;
  const double nu = 0.3;
  Constant mu(E/(2*(1 + nu)));
  Constant lambda(E*nu/((1 + nu)*(1 - 2*nu)));

  // Create (linear) form defining (nonlinear) variational problem
  HyperElasticity::ResidualForm F(V);
  F.mu = mu; F.lmbda = lambda; F.u = u;
  F.B = B; F.T = T;

  // Create jacobian dF = F' (for use in nonlinear solver).
  HyperElasticity::JacobianForm J(V, V);
  J.mu = mu; J.lmbda = lambda; J.u = u;

  // Solve nonlinear variational problem F(u; v) = 0
  solve(F == 0, u, bcs, J);

  // Save solution in VTK format
  File file("displacement.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
