// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <dolfin/ShortList.h>
#include <dolfin/Point.h>

namespace dolfin {
  
  class Quadrature {
  public:
    
    Quadrature(int n);
    ~Quadrature();
    
    int size() const;

    const Point& point(int i) const; // Get quadrature point
    real  weight(int i) const;       // Get quadrature weight
    real  measure() const;           // Sum of weights (area, volume)
    
  protected:
    
    int n;         // Number of quadrature points
    Point* points; // Quadrature points
    real* weights; // Quadrature weights
    real m;        // Sum of weights
    
  };
  
}

#endif
