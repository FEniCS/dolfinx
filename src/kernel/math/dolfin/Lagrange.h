// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
    Lagrange(int q);
    
    /// Copy constructor
    Lagrange(const Lagrange& p);
    
    /// Destructor
    ~Lagrange();
    
    /// Specify point
    void set(int i, real x);

    /// Return number of points
    int size() const;
    
    /// Return degree
    int degree() const;
    
    /// Return point
    real point(int i) const;
    
    /// Return value of polynomial i at given point x
    real operator() (int i, real x);
    
    /// Return value of polynomial i at given point x
    real eval(int i, real x);

    /// Return derivate of polynomial i at given point x
    real dx(int i, real x);
    
    /// Return derivative q (a constant) of polynomial
    real dqx(int i);

    /// Output
    friend LogStream& operator<<(LogStream& stream, const Lagrange& p);
    void show() const;
    
  private:

    void init();
    
    int q;
    int n;
    real* points; 
    real* constants;

    bool updated;

  };

}

#endif
