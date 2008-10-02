// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-06
// Last changed: 2006-10-23

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <dolfin/common/types.h>

namespace dolfin
{
  
  class Quadrature
  {
  public:
    
    /// Constructor
    Quadrature(unsigned int n);

    /// Destructor
    virtual ~Quadrature();
    
    /// Return number of quadrature points
    int size() const;

    /// Return quadrature point
    double point(unsigned int i) const;

    /// Return quadrature weight
    double weight(unsigned int i) const;

    /// Return sum of weights (length, area, volume)
    double measure() const;
    
    /// Display quadrature data
    virtual void disp() const = 0;

  protected:
    
    uint n;        // Number of quadrature points
    double* points;  // Quadrature points
    double* weights; // Quadrature weights
    double m;        // Sum of weights
    
  };
  
}

#endif
