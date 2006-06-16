// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005-11-28

#ifndef __POINT_H
#define __POINT_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class Point
  {
  public:
    
    Point();
    Point(real x);
    Point(real x, real y);
    Point(real x, real y, real z);
    Point(const Point& p);

    ~Point();

    /// Return distance to given point p
    real dist(Point p) const;

    /// Return distance to given point (x,y,z)
    real dist(real x, real y = 0.0, real z = 0.0) const;

    /// Return norm of vector represented by point
    real norm() const;
    
    /// Return midpoint on line to given point p
    Point midpoint(Point p) const;

    /// Cast to real, returning x
    operator real() const;

    /// Assignment from real, giving p = (x,0,0)
    const Point& operator= (real x);

    /// Add two points
    Point operator+ (const Point& p) const;
    
    /// Subtract two points
    Point operator- (const Point& p) const;

    /// Scalar product
    real operator* (const Point& p) const;
    
    /// Add point
    const Point& operator+= (const Point& p);

    /// Subtract point
    const Point& operator-= (const Point& p);

    /// Multiply by scalar
    const Point& operator*= (real a);

    /// Divide by scalar
    const Point& operator/= (real a);

    /// Cross product
    Point cross(const Point& p) const;

    /// The three coordinates
    real x;
    real y;
    real z;
    
    /// Output
    friend LogStream& operator<<(LogStream& stream, const Point& p);
    
  };

}
  
#endif
