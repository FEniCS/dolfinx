// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Quadrature using the three vertices on the reference triangle.

#ifndef __TRIANGLE_VERTEX_QUADRATURE_H
#define __TRIANGLE_VERTEX_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  class TriangleVertexQuadrature : public Quadrature {
  public:

	 TriangleVertexQuadrature();
	 
  };

}

#endif
