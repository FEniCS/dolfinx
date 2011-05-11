// Copyright (C) 2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-03-11
// Last changed: 2011-01-19
//
// Demonstrating function evaluation at arbitrary points.

#include <boost/assign/list_of.hpp>
#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

#ifdef HAS_CGAL

class F : public Expression
{
public:

  F() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
  }

};

int main()
{
  not_working_in_parallel("This demo");

  // Create mesh and a point in the mesh
  UnitCube mesh(8, 8, 8);
  Point x(0.31, 0.32, 0.33);

  // A user-defined function
  F f;

  // Project to a discrete function
  Projection::FunctionSpace V(mesh);
  Projection::BilinearForm a(V, V);
  Projection::LinearForm L(V);
  L.f = f;
  VariationalProblem pde(a, L);
  Function g(V);
  pde.solve(g);

  // Evaluate user-defined function f
  info("f(x) = %g", f(x));

  // Evaluate discrete function g (projection of f)
  info("g(x) = %g", g(x));
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
