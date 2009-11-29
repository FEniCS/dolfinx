// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Johannes Ring, 2009.
//
// First added:  2007-05-24
// Last changed: 2009-09-07
//
// Unit tests for the function library

#include <boost/assign/list_of.hpp>
#include <dolfin.h>
#include <dolfin/common/unittest.h>
#include "Projection.h"

using namespace dolfin;

class Eval : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Eval);
  CPPUNIT_TEST(testArbitraryEval);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testArbitraryEval()
  {
    class F0 : public Expression
    {
    public:

      void eval(std::vector<double>& values, const Data& data) const
      { 
        const std::vector<double>& x = data.x;
        values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
      }
    };
 
    class F1 : public Expression
    {
    public:

      void eval(std::vector<double>& values, const Data& data) const
      { 
        const std::vector<double>& x = data.x;
        values[0] = 1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]; 
      } 
    };

    UnitCube mesh(8, 8, 8);
    std::vector<double> x = boost::assign::list_of(0.31)(0.32)(0.33);
    std::vector<double> u0(1);
    std::vector<double> u1(1);
  
    Data data;
    data.x = x;

    // User-defined functions (one from finite element space, one not)
    F0 f0;
    F1 f1;

    // Test evaluation of a user-defined function
    f0.eval(u0, data);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(u0[0], 
				                         sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]),
				                         DOLFIN_EPS);

#ifdef HAS_GTS
    // Test evaluation of a discrete function
    Projection::FunctionSpace V(mesh);
    Projection::BilinearForm a(V, V);
    Projection::LinearForm L(V);
    L.f = f1;
    VariationalProblem problem(a, L);
    Function g(V);
    problem.solve(g);

//    const double tol = 1.0e-6;
    f1.eval(u0, data);
    g.eval(u1, data);
    //CPPUNIT_ASSERT( std::abs(u0[0]-u1[0]) < tol );
    CPPUNIT_ASSERT_DOUBLES_EQUAL(u0[0], u1[0], 1.0e-6); 
//				                         sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]),
//				                         DOLFIN_EPS);
#endif
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(Eval);

int main()
{
  DOLFIN_TEST;
}
