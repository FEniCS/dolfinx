// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2006-10-23

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <dolfin/constants.h>

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
    real point(unsigned int i) const;

    /// Return quadrature weight
    real weight(unsigned int i) const;

    /// Return sum of weights (length, area, volume)
    real measure() const;
    
    /// Display quadrature data
    virtual void disp() const = 0;

  protected:
    
    uint n;        // Number of quadrature points
    real* points;  // Quadrature points
    real* weights; // Quadrature weights
    real m;        // Sum of weights
    
  };
  
}

#endif
