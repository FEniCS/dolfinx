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
// First added:  2011-03-28
// Last changed: 2011-03-31
//
// Unit tests for the parameter library

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

TEST(InputOutput, test_simple)
{
    // Create some parameters
    Parameters p0("test");
    p0.add("filename", "foo.txt");
    p0.add("maxiter", 100);
    p0.add("tolerance", 0.001);
    p0.add("monitor_convergence", true);

    // Save to file
    File f0("test_parameters.xml");
    f0 << p0;

    // Read from file
    Parameters p1;
    File f1("test_parameters.xml");
    f1 >> p1;

    // Get parameter values
    std::string filename(p1["filename"]);
    std::size_t maxiter(p1["maxiter"]);
    double tolerance(p1["tolerance"]);
    bool monitor_convergence(p1["monitor_convergence"]);

    // Check values
    ASSERT_EQ(filename, "foo.txt");
    ASSERT_EQ(maxiter, (std::size_t) 100);
    ASSERT_DOUBLE_EQ(tolerance, 0.001);
    ASSERT_TRUE(monitor_convergence);
}

TEST(InputOutput, test_nested)
{
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

    // Save to file
    File f0("test_parameters.xml");
    f0 << p0;

    // Read from file
    Parameters p1;
    File f1("test_parameters.xml");
    f1 >> p1;

    // Get parameter values
    std::string foo(p1["foo"]);
    std::string filename(p1("sub0")["filename"]);
    std::size_t maxiter(p1("sub0")["maxiter"]);
    double tolerance(p1("sub0")["tolerance"]);
    bool monitor_convergence(p1("sub0")["monitor_convergence"]);

    // Check values
    ASSERT_EQ(foo, "bar");
    ASSERT_EQ(filename, "foo.txt");
    ASSERT_EQ(maxiter, (std::size_t) 100);
    ASSERT_DOUBLE_EQ(tolerance, 0.001);
    ASSERT_TRUE(monitor_convergence);
}

// Test all
int main(int argc, char **argv)
{
  // Not working in parallel, even if only process 0 writes and
  // others wait for a barrier. Skipping this in parallel for now.
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Skipping unit test in parallel.");
    info("OK");
    return 0;
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
