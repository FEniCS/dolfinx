// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TETRAHEDRON_MIDPOINT_QUADRATURE_H
#define __TETRAHEDRON_MIDPOINT_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin {

  /// Quadrature using the four midpoints on the edges
  /// of the reference tetrahedron.  
  class TetrahedronMidpointQuadrature : public Quadrature {
  public:
    
    TetrahedronMidpointQuadrature() : Quadrature(6) {
      
      // Volume of tetrahedron
      m = 1 / 6.0;
      
      // Quadrature points
      points[0] = Point(0.5, 0.0, 0.0);
      points[1] = Point(0.5, 0.5, 0.0);
      points[2] = Point(0.0, 0.5, 0.0);
      points[3] = Point(0.0, 0.0, 0.5);
      points[4] = Point(0.5, 0.0, 0.5);
      points[5] = Point(0.0, 0.5, 0.5);
      
      // Quadrature weights
      weights[0] = m / 6.0;
      weights[1] = m / 6.0;
      weights[2] = m / 6.0;
      weights[3] = m / 6.0;
      weights[4] = m / 6.0;
      weights[5] = m / 6.0;
      
    }
    
  };
  
}

#endif
