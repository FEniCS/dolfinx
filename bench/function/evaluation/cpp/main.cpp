// Copyright (C) 2010 Andre Massing
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
// First added:  2010-06-10
// Last changed: 2012-12-12
//
// Description: Benchmark for the evaluations of functions at arbitrary points.

#include <dolfin.h>
#include "P1.h"

using namespace dolfin;

class F : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
  }

};

int main(int argc, char* argv[])
{
  not_working_in_parallel("Function evalutation benchmark");

  info("Evaluations of functions at arbitrary points.");

  const std::size_t mesh_max_size = 32;
  const std::size_t num_points  = 10000000;

  // Start timing
  tic();
  for (std::size_t N = 10; N < mesh_max_size; N += 2)
  {
    const auto mesh = std::make_shared<const UnitCubeMesh>(N, N, N);

    const auto V0 = std::make_shared<const P1::FunctionSpace>(mesh);
    Function f0(V0);
    F f;
    f0.interpolate(f);

    Array<double> X(3);
    Array<double> value(1);

    // Initialize random generator generator (produces same sequence each test).
    srand(1);

    for (std::size_t i = 1; i <= num_points; ++i)
    {
      X[0] = std::rand()/static_cast<double>(RAND_MAX);
      X[1] = std::rand()/static_cast<double>(RAND_MAX);
      X[2] = std::rand()/static_cast<double>(RAND_MAX);

      f.eval(value, X);
    }

    // Use X variable.
    info("x = %.12e\ty = %.12e\tz = %.12e\tf(x) = %.12e", X[0], X[1], X[2], value[0]);
  }
  info("BENCH  %g",toc());

  return 0;
}
