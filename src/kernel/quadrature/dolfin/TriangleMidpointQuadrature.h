// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRIANGLE_MIDPOINT_QUADRATURE_H
#define __TRIANGLE_MIDPOINT_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  /// Quadrature using the three midpoints on the edges
  /// of the reference triangle.
  class TriangleMidpointQuadrature : public Quadrature {
  public:
    
    TriangleMidpointQuadrature() : Quadrature(3) {
      
      // Area of triangle
      m = 0.5;
      
      // Quadrature points
      p[0] = Point(0.5, 0.0);
      p[1] = Point(0.5, 0.5);
      p[2] = Point(0.0, 0.5);
      
      // Quadrature weights
      w[0] = m / 3.0;
      w[1] = m / 3.0;
      w[2] = m / 3.0;
      
    }
    
  };
  
}

#endif
