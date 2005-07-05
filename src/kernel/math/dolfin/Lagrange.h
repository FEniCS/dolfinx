// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-06-12
// Last changed: 2005

#ifndef __LAGRANGE_H
#define __LAGRANGE_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin {

  /// Lagrange polynomial (basis) with given degree q determined by n = q + 1 nodal points.
  ///
  /// Example: q = 1 (n = 2)
  ///
  ///   Lagrange p(1);
  ///   p.set(0, 0.0);
  ///   p.set(1, 1.0);
  ///
  /// This creates a Lagrange polynomial (actually two Lagrange polynomials):
  ///
  ///   p(0,x) = 1 - x   (one at x = 0, zero at x = 1)
  ///   p(1,x) = x       (zero at x = 0, one at x = 1)

  class Lagrange {
  public:
 
    /// Constructor
    Lagrange(unsigned int q);
    
    /// Copy constructor
    Lagrange(const Lagrange& p);
    
    /// Destructor
    ~Lagrange();
    
    /// Specify point
    void set(unsigned int i, real x);

    /// Return number of points
    unsigned int size() const;
    
    /// Return degree
    unsigned int degree() const;
    
    /// Return point
    real point(unsigned int i) const;
    
    /// Return value of polynomial i at given point x
    real operator() (unsigned int i, real x);
    
    /// Return value of polynomial i at given point x
    real eval(unsigned int i, real x);

    /// Return derivate of polynomial i at given point x
    real ddx(unsigned int i, real x);

    /// Return derivative q (a constant) of polynomial
    real dqdx(unsigned int i);

    /// Output
    friend LogStream& operator<<(LogStream& stream, const Lagrange& p);
    void show() const;
    
  private:

    void init();
    
    unsigned int q;
    unsigned int n;
    real* points; 
    real* constants;

  };

}

#endif
