// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Quadrature using the three vertices on the reference triangle.

#ifndef __TRIANGLE_VERTEX_QUADRATURE_H
#define __TRIANGLE_VERTEX_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  class TriangleVertexQuadrature : public Quadrature {
  public:

	 TriangleVertexQuadrature() : Quadrature(3) {

		// Area of triangle
		m = 0.5;
		
		// Quadrature points
		p[0] = Point(0.0, 0.0);
		p[1] = Point(1.0, 0.0);
		p[2] = Point(0.0, 1.0);
		
		// Quadrature weights
		w[0] = m / 3.0;
		w[1] = m / 3.0;
		w[2] = m / 3.0;

	 }
	 
  };

}

#endif
