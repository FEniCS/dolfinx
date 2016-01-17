// Copyright (C) 2012 Imperial College London and others.
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
// Modified by David Ham 2012
//
// First added:  2012-12-04
// Last changed: 2012-12-04
//
// This demo program solves Poisson's equation
//
//     - div grad u(xi) = f(xi)
//
// on a unit interval embedded in 2D and rotated pi/4 radians
// anticlockwise from the x axis, where xi is the distance along the
// interval (ie the domain is 0<xi<1).  The source f is given by
//
//     f(xi) = 9*pi^2*sin(3*pi*xi)
//
// The boundary conditions are given by
//
//     u(xi) = 0 for xi = 0
//     du/dxi = 0 for xi = 1

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

#define THETA DOLFIN_PI/4.0

// Map the point x back to the horizontal line
double to_interval(const Array<double>& x)
{
  return (x[0]*cos(THETA) + x[1]*sin(THETA));
}

// Rotate the mesh through theta
void rotate(Mesh & mesh)
{
  std::vector<double>& x = mesh.coordinates();
  double tmpx;
  for (std::size_t i = 0; i < mesh.num_vertices(); i++)
  {
    tmpx = x[2*i]*cos(THETA) - x[2*i+1]*sin(THETA);

    x[2*i+1] = x[2*i]*sin(THETA) + x[2*i+1]*cos(THETA);
    x[2*i]   = tmpx;
  }
}

// Subdomain to extract the bottom boundary of the mesh.
class BottomEdge : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return (std::abs(x[1]) < DOLFIN_EPS); }
};

// Boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return (std::abs(to_interval(x)) < DOLFIN_EPS); }
};

// Source term
class Source : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = 9.0*DOLFIN_PI*DOLFIN_PI*sin(3.0*DOLFIN_PI*to_interval(x)); }

};

// Neumann boundary condition
class Flux : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = 3.0*DOLFIN_PI*cos(3.0*DOLFIN_PI*to_interval(x)); }

};

int main()
{
  // Create original square mesh
  UnitSquareMesh squaremesh(50, 2);

  // Grab the surface of the mesh
  BoundaryMesh boundarymesh(squaremesh, "exterior");

  // The actual mesh is just the bottom.
  SubMesh mesh(boundarymesh, BottomEdge());

  // Rotate mesh coordinates.
  rotate(mesh);

  // Create function space
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Set up BCs
  auto zero = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  DirichletBC bc(V, zero, boundary);

  // Create source and flux terms
  Source f;
  Flux g;

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;
  L.g = g;

  // Solve PDE
  Function u(V);
  solve(a == L, u, bc);

  // Save solution in VTK format
  File file_u("poisson.pvd");
  file_u << u;

  return 0;
}
