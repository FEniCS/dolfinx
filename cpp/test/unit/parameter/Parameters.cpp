// Copyright (C) 2011 Anders Logg
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
// Unit tests for the parameter library

#include <catch.hpp>
#include <dolfin.h>

using namespace dolfin;

TEST_CASE("parameters io", "[test_parameter_io]")
{
  SECTION("flat parameters")
  {
    if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
      return;

    // Create some parameters
    Parameters p0("test");
    p0.add("filename", "foo.txt");
    p0.add("maxiter", 100);
    p0.add("tolerance", 0.001);
    p0.add("monitor_convergence", true);
  }

  SECTION("nested parameters")
  {
    if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
      return;

    // Create some nested parameters
    Parameters p0("test");
    Parameters p00("sub0");
    p00.add("filename", "foo.txt");
    p00.add("maxiter", 100);
    p00.add("tolerance", 0.001);
    p00.add("monitor_convergence", true);
    p0.add("foo", "bar");
    Parameters p01(p00);
    p01.rename("sub1");
    p0.add(p00);
    p0.add(p01);
  }
}
