// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRIANGLE_VERTEX_QUADRATURE_H
#define __TRIANGLE_VERTEX_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  /// Quadrature using the three vertices on the reference triangle.
  class TriangleVertexQuadrature : public Quadrature {
  public:
    
    TriangleVertexQuadrature() : Quadrature(3) {
      
      // Area of triangle
      m = 0.5;
      
      // Quadrature points
      points[0] = Point(0.0, 0.0);
      points[1] = Point(1.0, 0.0);
      points[2] = Point(0.0, 1.0);
      
      // Quadrature weights
      weights[0] = m / 3.0;
      weights[1] = m / 3.0;
      weights[2] = m / 3.0;
      
    }
    
  };
  
}

#endif
