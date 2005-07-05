// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <dolfin/Point.h>

namespace dolfin
{
  
  class Quadrature
  {
  public:
    
    Quadrature(unsigned int n);
    virtual ~Quadrature();
    
    /// Return number of quadrature points
    int size() const;

    /// Return quadrature point
    const Point& point(unsigned int i) const;

    /// Return quadrature weight
    real weight(unsigned int i) const;

    /// Return sum of weights (length, area, volume)
    real measure() const;
    
    /// Display quadrature data
    virtual void show() const = 0;

  protected:
    
    unsigned int n; // Number of quadrature points
    Point* points;  // Quadrature points
    real* weights;  // Quadrature weights
    real m;         // Sum of weights
    
  };
  
}

#endif
