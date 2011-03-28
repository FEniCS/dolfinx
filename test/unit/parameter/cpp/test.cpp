// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-28
// Last changed: 2011-03-28
//
// Unit tests for the parameter library

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class InputOutput : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(InputOutput);
  CPPUNIT_TEST(test_simple);
  CPPUNIT_TEST(test_nested);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_simple()
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
    dolfin::uint maxiter(p1["maxiter"]);
    double tolerance(p1["tolerance"]);
    bool monitor_convergence(p1["monitor_convergence"]);

    // Check values
    CPPUNIT_ASSERT(filename == "foo.txt");
    CPPUNIT_ASSERT(maxiter == 100);
    CPPUNIT_ASSERT(tolerance == 0.001);
    CPPUNIT_ASSERT(monitor_convergence == true);
  }

  void test_nested()
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
    dolfin::uint maxiter(p1("sub0")["maxiter"]);
    double tolerance(p1("sub0")["tolerance"]);
    bool monitor_convergence(p1("sub0")["monitor_convergence"]);

    // Check values
    CPPUNIT_ASSERT(foo == "bar");
    CPPUNIT_ASSERT(filename == "foo.txt");
    CPPUNIT_ASSERT(maxiter == 100);
    CPPUNIT_ASSERT(tolerance == 0.001);
    CPPUNIT_ASSERT(monitor_convergence == true);
  }

};


int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(InputOutput);
  DOLFIN_TEST;
}
