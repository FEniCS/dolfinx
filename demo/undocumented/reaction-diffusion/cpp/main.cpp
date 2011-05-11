// Copyright (C) 2009 Anders Logg
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
// First added:  2009-06-15
// Last changed: 2010-01-27
//
// This demo program solves the reaction-diffusion equation
//
//    - div grad u + u = f
//
// on the unit square with f = sin(x)*sin(y) and homogeneous Neumann
// boundary conditions.

#include <dolfin.h>
#include "ReactionDiffusion.h"

using namespace dolfin;

// Source term
class Source : public Expression
{
public:

  Source() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(x[0])*sin(x[1]);
  }

};

int main()
{
  // Define variational problem
  UnitSquare mesh(32, 32);
  Source f;
  ReactionDiffusion::FunctionSpace V(mesh);
  ReactionDiffusion::BilinearForm a(V, V);
  ReactionDiffusion::LinearForm L(V, f);

  // Compute and plot solution
  VariationalProblem problem(a, L);
  Function u(V);
  problem.solve(u);
  plot(u);

  return 0;
}
