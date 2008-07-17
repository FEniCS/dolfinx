// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-05-24
// Last changed: 2008-07-17
//
// Unit tests for the function library

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
    class F0 : public Function
    {
    public:
      F0(Mesh& mesh) : Function(mesh) {}
      void eval(real* values, const real* x) const
      { values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]); } 
    };
 
    class F1 : public Function
    {
    public:
      F1(Mesh& mesh) : Function(mesh) {}
      void eval(real* values, const real* x) const
      { values[0] = 1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]; } 
    };

    UnitCube mesh(8, 8, 8);
    real x[3] = {0.3, 0.3, 0.3};
    real u[1] = {0.0};
    real v[1] = {0.0};
  
    // User-defined functions (one from finite element space, one not)
    F0 f0(mesh);
    F1 f1(mesh);

    // Test evaluation of a user-defined function
    f0.eval(v, x);
    CPPUNIT_ASSERT(v[0] == sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]));

#ifdef HAS_GTS
    // Test evaluation of a discrete function
    ProjectionBilinearForm a;
    ProjectionLinearForm L(f1);
    LinearPDE pde(a, L, mesh);
    Function g;
    pde.solve(g);

    const real tol = 1.0e-6;
    f1.eval(u, x);
    g.eval(v, x);
    CPPUNIT_ASSERT( fabs(u[0]-v[0]) < tol );
#endif
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(Eval);

int main()
{
  DOLFIN_TEST;
}
