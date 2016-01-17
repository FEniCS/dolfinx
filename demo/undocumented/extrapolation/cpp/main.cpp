// Copyright (C) 2010 Anders Logg
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
// First added:  2010-02-08
// Last changed: 2012-07-05
//
// This program demonstrates extrapolation of a P1 function to a P2
// function on the same mesh. This is useful for postprocessing a
// computed dual approximation for use in an error estimate.

#include <dolfin.h>
#include "P1.h"
#include "P2.h"

using namespace dolfin;

class Dual : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5.0*x[0])*sin(5.0*x[1]);
  }

};

int main()
{
  // Create mesh and function spaces
  auto mesh = std::make_shared<UnitSquareMesh>(8, 8);
  auto P1 = std::make_shared<P1::FunctionSpace>(mesh);
  auto P2 = std::make_shared<P2::FunctionSpace>(mesh);

  // Create exact dual
  Dual dual;

  // Create P1 approximation of exact dual
  Function z1(P1);
  z1.interpolate(dual);

  // Create P2 approximation from P1 approximation
  Function z2(P2);
  z2.extrapolate(z1);

  // Plot approximations
  plot(z1, "z1");
  plot(z2, "z2");
  interactive();


  return 0;
}
