// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2006-10-16

#ifndef __POINT_H
#define __POINT_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

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

    /// Return coordinate in direction i
    inline real operator[] (uint i) const { dolfin_assert(i < 3); return _x[i]; }

    /// Return address of x-coordinate
    inline real& x() { return _x[0]; }

    /// Return address of y-coordinate
    inline real& y() { return _x[1]; }

    /// Return address of z-coordinate
    inline real& z() { return _x[2]; }

    /// Return x-coordinate
    inline real x() const { return _x[0]; }

    /// Return y-coordinate
    inline real y() const { return _x[1]; }

    /// Return z-coordinate
    inline real z() const { return _x[2]; }

    /// Cast to real, returning x
    operator real() const { return _x[0]; }

    /// Assignment from real, giving p = (x,0,0)
    const Point& operator= (const real x)
    {
      _x[0] = x; _x[1] = 0.0; _x[2] = 0.0;
      return *this;
    }

    /// Output
    friend LogStream& operator<< (LogStream& stream, const Point& p);

  private:

    real _x[3];

  };

}

#endif
