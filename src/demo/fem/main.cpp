// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

real f0(real x, real y, real z, real t) { return 1 - x - y; }
real f1(real x, real y, real z, real t) { return x; }
real f2(real x, real y, real z, real t) { return y; }

void main()
{
  // Definition of shape functions
  FunctionSpace::ShapeFunction v0(f0);
  FunctionSpace::ShapeFunction v1(f1);
  FunctionSpace::ShapeFunction v2(f2);

  cout << v0 << endl;
  cout << v1 << endl;
  cout << v2 << endl << endl;
  
  // Some shape function algebra
  FunctionSpace::ElementFunction u = v0 + v1 * v2;
  FunctionSpace::ElementFunction v = (2.0*v2 + 5.0*v0) * v1;

  cout << u << endl;
  cout << v << endl << endl;
  
  cout << "u(0.1, 0.2, 0.3, 0.0) = " << u(0.1, 0.2, 0.3, 0.0) << endl;
  cout << "v(0.1, 0.2, 0.3, 0.0) = " << v(0.1, 0.2, 0.3, 0.0) << endl << endl;

  // Define a triangle
  Node n0(2.0, 1.0);
  Node n1(5.0, 1.0);
  Node n2(4.0, 4.0);
  
  Cell triangle(n0, n1, n2);

  // Define a quadrature rule
  TriangleMidpointQuadrature q;

  // Define a mapping
  TriLinMapping m;
  m.update(triangle);
  
  // Define integral measures
  Integral::InteriorMeasure dK(m, q);
  Integral::BoundaryMeasure dS(m, q);

  cout << "Area of triangle = " << 1.0 * dK << endl;
  
  cout << "u * dK = " << u * dK << endl;         // Should be 15 / 8 = 1.875
  cout << "v * dK = " << v * dK << endl << endl; // Should be 21 / 8 = 2.625

}
