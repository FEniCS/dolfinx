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
// First added:  2012-08-16
// Last changed: 2012-11-12
//
// This demo program solves the heat equation
//
//     du/dt - div kappa(x, y) grad u(x, y, t) = f(x, y, t)
//
// on the unit square with source f given by
//
//     f(x, y, t) = sin(5*t)*10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// heat conductivity kappa given by
//
//     kappa(x, y) = 1   for x <  0.5
//     kappa(x, y) = 0.1 for x >= 0.5
//
// boundary conditions given by
//
//     u(x, y, t) = 0        for x = 0 or x = 1
// du/dn(x, y, t) = sin(5*x) for y = 0 or y = 1
//
// and initial condition given by u(x, y, 0) = 0.

#include <dolfin.h>

// FIXME: Include standard file for now
#include "Heat_2D.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x, double t) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = sin(5*t)*10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x, double t) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Temporary until this works
  cout << "This demo is not yet working" << endl;
  return 0;

  // Create mesh and function space
  UnitSquareMesh mesh(32, 32);
  Heat_2D::Form_0::TrialSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  return 0;
}
