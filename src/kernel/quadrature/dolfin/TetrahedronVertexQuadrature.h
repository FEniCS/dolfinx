// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-12
// Last changed: 2005

#ifndef __TETRAHEDRON_VERTEX_QUADRATURE_H
#define __TETRAHEDRON_VERTEX_QUADRATURE_H

#include <dolfin/Quadrature.h>

namespace dolfin
{

  /// Quadrature using the four vertices on the reference tetrahedron.  
  class TetrahedronVertexQuadrature : public Quadrature
  {
  public:
    
    TetrahedronVertexQuadrature() : Quadrature(4)
    {
      
      // Volume of tetrahedron
      m = 1.0 / 6.0;
      
      // Quadrature points
      points[0] = Point(0.0, 0.0, 0.0);
      points[1] = Point(1.0, 0.0, 0.0);
      points[2] = Point(0.0, 1.0, 0.0);
      points[3] = Point(0.0, 0.0, 1.0);
      
      // Quadrature weights
      weights[0] = m / 4.0;
      weights[1] = m / 4.0;
      weights[2] = m / 4.0;
      weights[3] = m / 4.0;
      
    }

    void show() const {};
    
  };
  
}

#endif
