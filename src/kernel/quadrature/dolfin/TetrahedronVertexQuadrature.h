// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Quadrature using the four vertices on the reference tetrahedron.

#ifndef __TETRAHEDRON_VERTEX_QUADRATURE_H
#define __TETRAHEDRON_VERTEX_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  class TetrahedronVertexQuadrature : public Quadrature {
  public:

	 TetrahedronVertexQuadrature() : Quadrature(4) {

		// Volume of tetrahedron
		m = 1.0 / 6.0;
		
		// Quadrature points
		p[0] = Point(0.0, 0.0, 0.0);
		p[1] = Point(1.0, 0.0, 0.0);
		p[2] = Point(0.0, 1.0, 0.0);
		p[3] = Point(0.0, 0.0, 1.0);
		
		// Quadrature weights
		w[0] = m / 4.0;
		w[1] = m / 4.0;
		w[2] = m / 4.0;
		w[3] = m / 4.0;

	 }
	 
  };

}

#endif
