// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Quadrature using the four midpoints on the edges
// of the reference tetrahedron.

#ifndef __TETRAHEDRON_MIDPOINT_QUADRATURE_H
#define __TETRAHEDRON_MIDPOINT_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  class TetrahedronMidpointQuadrature : public Quadrature {
  public:

	 TetrahedronMidpointQuadrature() : Quadrature(6) {

		// Volume of tetrahedron
		m = 1 / 6.0;
		
		// Quadrature points
		p[0] = Point(0.5, 0.0, 0.0);
		p[1] = Point(0.5, 0.5, 0.0);
		p[2] = Point(0.0, 0.5, 0.0);
		p[3] = Point(0.0, 0.0, 0.5);
		p[4] = Point(0.5, 0.0, 0.5);
		p[5] = Point(0.0, 0.5, 0.5);
		
		// Quadrature weights
		w[0] = m / 6.0;
		w[1] = m / 6.0;
		w[2] = m / 6.0;
		w[3] = m / 6.0;
		w[4] = m / 6.0;
		w[5] = m / 6.0;

	 }
	 
  };

}

#endif
