// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Quadrature using the three midpoints on the edges
// of the reference triangle.

#ifndef __TRIANGLE_MIDPOINT_QUADRATURE_H
#define __TRIANGLE_MIDPOINT_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  class TriangleMidpointQuadrature : public Quadrature {
  public:

	 TriangleMidpointQuadrature();
	 
  };

}

#endif
