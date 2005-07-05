// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005

#ifndef __TRIANGLE_MIDPOINT_QUADRATURE_H
#define __TRIANGLE_MIDPOINT_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin
{

  /// Quadrature using the three midpoints on the edges
  /// of the reference triangle.
  class TriangleMidpointQuadrature : public Quadrature
  {
  public:
    
    TriangleMidpointQuadrature() : Quadrature(3)
    {
      
      // Area of triangle
      m = 0.5;
      
      // Quadrature points
      points[0] = Point(0.5, 0.0);
      points[1] = Point(0.5, 0.5);
      points[2] = Point(0.0, 0.5);
      
      // Quadrature weights
      weights[0] = m / 3.0;
      weights[1] = m / 3.0;
      weights[2] = m / 3.0;
      
    }

    void show() const {};
    
  };
  
}

#endif
