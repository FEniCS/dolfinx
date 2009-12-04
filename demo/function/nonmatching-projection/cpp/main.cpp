// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-10-10
// Last changed:

//
// This program demonstrates the L2 projection of a function onto a 
// non-matching mesh.
//

#include <dolfin.h>
#include "P1_projection.h"
#include "P3.h"

using namespace dolfin;

#ifdef HAS_CGAL

class MyExpression : public Expression
{
public:

  MyExpression() : Expression() {}

  void eval(double* values, const std::vector<double>& x) const
  {
    values[0] = sin(10.0*x[0])*sin(10.0*x[1]);
  }

};

int main()
{
  // Create meshes
  UnitSquare mesh0(16, 16);
  UnitSquare mesh1(64, 64);

  // Create P3 function space
  P3::FunctionSpace V0(mesh0);

  // Interpolate expression into V0
  MyExpression e;
  Function f0(V0);
  f0.interpolate(e);

  // Create forms for projection
  P1_projection::FunctionSpace V1(mesh1);
  P1_projection::BilinearForm a(V1, V1);
  P1_projection::LinearForm L(V1, f0);

  // Create projection problem
  VariationalProblem projection(a, L);

  // Project f0 into V1 
  Function f1(V1);
  projection.solve(f1);

  // Plot results
  plot(f0);
  plot(f1);

  return 0;
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
