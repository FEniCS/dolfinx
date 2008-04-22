// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2007-04-16

#ifndef __POINT_H
#define __POINT_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  /// A Point represents a point in R^3 with coordinates x, y, z, or,
  /// alternatively, a vector in R^3, supporting standard operations
  /// like the norm, distances, scalar and vector products etc.

  class Point
  {
  public:

    /// Create a point at (x, y, z)
    Point(const real x = 0.0, const real y = 0.0, const real z =0.0) 
      { _x[0] = x; _x[1] = y; _x[2] = z; }

    /// Destructor
    ~Point() {};

    /// Return address of coordinate in direction i
    inline real& operator[] (uint i) { dolfin_assert(i < 3); return _x[i]; }

    /// Return coordinate in direction i
    inline real operator[] (uint i) const { dolfin_assert(i < 3); return _x[i]; }

    /// Return x-coordinate
    inline real x() const { return _x[0]; }

    /// Return y-coordinate
    inline real y() const { return _x[1]; }

    /// Return z-coordinate
    inline real z() const { return _x[2]; }

    /// Compute sum of two points
    Point operator+ (const Point& p) const { Point q(_x[0] + p._x[0], _x[1] + p._x[1], _x[2] + p._x[2]); return q; }
    
    /// Compute difference of two points
    Point operator- (const Point& p) const { Point q(_x[0] - p._x[0], _x[1] - p._x[1], _x[2] - p._x[2]); return q; }

    /// Add given point
    const Point& operator+= (const Point& p) { _x[0] += p._x[0]; _x[1] += p._x[1]; _x[2] += p._x[2]; return *this; }

    /// Subtract given point
    const Point& operator-= (const Point& p) { _x[0] -= p._x[0]; _x[1] -= p._x[1]; _x[2] -= p._x[2]; return *this; }

    /// Multiplication with scalar
    Point operator* (real a) const { Point p(a*_x[0], a*_x[1], a*_x[2]); return p; }

    /// Incremental multiplication with scalar
    const Point& operator*= (real a) { _x[0] *= a; _x[1] *= a; _x[2] *= a; return *this; }
    
    /// Division by scalar
    Point operator/ (real a) const { Point p(_x[0]/a, _x[1]/a, _x[2]/a); return p; }

    /// Incremental division by scalar
    const Point& operator/= (real a) { _x[0] /= a; _x[1] /= a; _x[2] /= a; return *this; }

    /// Compute distance to given point
    real distance(const Point& p) const;

    /// Compute norm of point representing a vector from the origin
    real norm() const;

    /// Compute cross product with given vector
    const Point cross(const Point& p) const;

    /// Compute dot product with given vector
    real dot(const Point& p) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const Point& p);

  private:

    real _x[3];

  };

  /// Multiplication with scalar
  inline Point operator*(real a, const Point& p) { return p*a; }
  LogStream& operator<< (LogStream& stream, const Point& p);
}

#endif
