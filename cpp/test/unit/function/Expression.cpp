// Copyright (C) 2007 Anders Logg
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
// Unit tests for the function library

//#include "Projection.h"
#include <catch.hpp>
#include <dolfin.h>

using namespace dolfin;

namespace
{
void arbitrary_eval()
{
  /*
  class F0 : public Expression
  {
  public:
    F0() {}
    void eval(Array<double>& values, const Array<double>& x) const
    { values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]); }
  };

  class F1 : public Expression
  {
  public:
    F1() {}
    void eval(Array<double>& values, const Array<double>& x) const
    { values[0] = 1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]; }
  };

  auto mesh = std::make_shared<UnitCubeMesh>(8, 8, 8);

  Array<double> x(3);
  x[0] = 0.31; x[1] = 0.32; x[2] = 0.33;

  Array<double> u0(1);
  Array<double> u1(1);

  // User-defined functions (one from finite element space, one not)
  F0 f0;
  auto f1 = std::make_shared<F1>();

  // Test evaluation of a user-defined function
  f0.eval(u0, x);
  CHECK(u0[0] == Approx(sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])));

  // Test for single core only
  if (dolfin::MPI::size(mesh->mpi_comm()) == 1)
  {
  // Test evaluation of a discrete function
    auto V = std::make_shared<Projection::FunctionSpace>(mesh);
    Projection::BilinearForm a(V, V);
    Projection::LinearForm L(V);
    L.f = f1;
    Function g(V);
    solve(a == L, g);

    const double tol = 1.0e-6;
    f1->eval(u0, x);
    g.eval(u1, x);
    CHECK(std::abs(u0[0] - u1[0]) < tol);
  }
  */
}
} // namespace

//-----------------------------------------------------------------------------
TEST_CASE("Test arbitray eval", "[eval]") { CHECK_NOTHROW(arbitrary_eval()); }
