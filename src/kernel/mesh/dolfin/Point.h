// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POINT_H
#define __POINT_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin {

  class Point {
  public:
    
    Point();
    Point(real x);
    Point(real x, real y);
    Point(real x, real y, real z);

    /// Return distance to given point p
    real dist(Point p) const;

    /// Return distance to given point (x,y,z)
    real dist(real x, real y = 0.0, real z = 0.0) const;
    
    /// Return midpoint on line to given point p
    Point midpoint(Point p) const;

    /// Assignment from real, giving p = (x,0,0)
    void operator= (real x);

    /// Cast to real, returning x
    operator real() const;

    /// Add point
    Point operator+= (const Point& p);

    /// Divide by scalar
    Point operator/= (real a);

    /// The three coordinates
    real x;
    real y;
    real z;
    
    /// Output
    friend LogStream& operator<<(LogStream& stream, const Point& p);
    
  };

}
  
#endif
