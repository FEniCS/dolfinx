// Copyright (C) 2010-06-10 André Massing.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-06-10
// Last changed: 2010-06-14
//
// Description: Benchmark for the evaluations of functions at arbitrary points.

#include <dolfin.h>
#include "P1.h"

using namespace dolfin;
using dolfin::uint;

class F : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
  }

};

#ifdef HAS_CGAL

int main(int argc, char* argv[])
{
  not_working_in_parallel("Function evalutation benchmark");

  info("Evaluations of functions at arbitrary points.");

  const uint mesh_max_size = 32;
  const uint num_points  = 10000000;

  // Start timing
  tic();
  for (uint N = 10; N < mesh_max_size; N += 2)
  {
    UnitCube mesh(N, N, N);

    P1::FunctionSpace V0(mesh);
    Function f0(V0);
    F f;
    f0.interpolate(f);

    Array<double> X(3);
    Array<double> value(1);

    // Initialize random generator generator (produces same sequence each test).
    srand(1);

    for (uint i = 1; i <= num_points; ++i)
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

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run function eval benchmark.");
  return 0;
}

#endif
