// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

real f0(real x, real y, real z, real t) { return 1 - x - y; }
real f1(real x, real y, real z, real t) { return x; }
real f2(real x, real y, real z, real t) { return y; }

int main()
{
  // Definition of shape functions
  FunctionSpace::ShapeFunction v0(f0);
  FunctionSpace::ShapeFunction v1(f1);
  FunctionSpace::ShapeFunction v2(f2);

  dolfin::cout << v0 << dolfin::endl;
  dolfin::cout << v1 << dolfin::endl;
  dolfin::cout << v2 << dolfin::endl << dolfin::endl;
  
  // Some shape function algebra
  FunctionSpace::ElementFunction u = v0 + v1 * v2;
  FunctionSpace::ElementFunction v = (2.0*v2 + 5.0*v0) * v1;

  dolfin::cout << u << dolfin::endl;
  dolfin::cout << v << dolfin::endl << dolfin::endl;
  
  dolfin::cout << "u(0.1, 0.2, 0.3, 0.0) = " << u(0.1, 0.2, 0.3, 0.0) << dolfin::endl;
  dolfin::cout << "v(0.1, 0.2, 0.3, 0.0) = " << v(0.1, 0.2, 0.3, 0.0) << dolfin::endl << dolfin::endl;

  // Define a triangle
  Node n0(2.0, 1.0);
  Node n1(5.0, 1.0);
  Node n2(4.0, 4.0);
  
  Cell triangle(n0, n1, n2);

  // Define a quadrature rule
  TriangleMidpointQuadrature q;

  // Define a mapping
  P1TriMap m;
  m.update(triangle);
  
  // Define integral measures
  Integral::InteriorMeasure dK(m, q);
  Integral::BoundaryMeasure dS(m, q);

  dolfin::cout << "Area of triangle = " << 1.0 * dK << dolfin::endl;
  
  dolfin::cout << "u * dK = " << u * dK << dolfin::endl;         // Should be 15 / 8 = 1.875
  dolfin::cout << "v * dK = " << v * dK << dolfin::endl << dolfin::endl; // Should be 21 / 8 = 2.625

  // Derivatives
  P1Tri p1Tri;
 
  FunctionSpace::Iterator w(p1Tri);

  dolfin::cout << "w     = " << *w << dolfin::endl << dolfin::endl;
  
  dolfin::cout << "dw/dx = " << (*w).dx() << dolfin::endl;
  dolfin::cout << "dw/dy = " << (*w).dy() << dolfin::endl;
  dolfin::cout << "dw/dz = " << (*w).dz() << dolfin::endl << dolfin::endl;
  
  dolfin::cout << "Dw/dx = " << m.dx(*w) << dolfin::endl;
  dolfin::cout << "Dw/dy = " << m.dy(*w) << dolfin::endl;
  dolfin::cout << "Dw/dz = " << m.dz(*w) << dolfin::endl << dolfin::endl;

  return 0;
}
