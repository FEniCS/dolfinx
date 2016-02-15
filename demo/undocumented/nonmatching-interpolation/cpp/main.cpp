// Copyright (C) 2009 Garth N. Wells
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
// First added:  2009-06-17
// Last changed: 2014-08-11

//
// This program demonstrates the interpolation of functions on non-matching
// meshes.
//

#include <dolfin.h>
#include "P1.h"
#include "P3.h"

using namespace dolfin;

class MyExpression : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(10.0*x[0])*sin(10.0*x[1]);
  }

};

int main()
{
  // Create meshes
  auto mesh0 = std::make_shared<UnitSquareMesh>(16, 16);
  auto mesh1 = std::make_shared<UnitSquareMesh>(64, 64);

  // Create function spaces
  auto V0 = std::make_shared<P3::FunctionSpace>(mesh0);
  auto V1 = std::make_shared<P1::FunctionSpace>(mesh1);

  // Create functions
  Function f0(V0);
  Function f1(V1);

  // Interpolate expression into V0
  MyExpression e;
  f0.interpolate(e);

  // Interpolate V0 function (coarse mesh) into V1 function space (fine mesh)
  f1.interpolate(f0);

  // Plot results
  plot(f0);
  plot(f1);
  interactive();

  return 0;
}
