// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-12
// Last changed: 2006-06-12

#ifndef __NEW_POINT_H
#define __NEW_POINT_H

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

    /// Create a point at (0, 0, 0)
    Point() { _x[0] = 0.0; _x[1] = 0.0; _x[2] = 0.0; }

    /// Create a point at (x, 0, 0)
    Point(real x) { _x[0] = x; _x[1] = 0.0; _x[2] = 0.0; }

    /// Create a point at (x, y, 0)
    Point(real x, real y) { _x[0] = x; _x[1] = y; _x[2] = 0.0; }

    /// Create a point at (x, y, z)
    Point(real x, real y, real z) { _x[0] = x; _x[1] = y; _x[2] = z; }

    /// Destructor
    ~Point() {};

    /// Return coordinate in direction i
    inline real operator[] (uint i) const { dolfin_assert(i < 3); return _x[i]; }

    /// Return x-coordinate
    inline real x() const { return _x[0]; }

    /// Return y-coordinate
    inline real y() const { return _x[1]; }

    /// Return z-coordinate
    inline real z() const { return _x[2]; }

    /// Cast to real, returning x
    operator real() const { return _x[0]; }

    /// Assignment from real, giving p = (x,0,0)
    const Point& operator= (real x)
    {
      _x[0] = x; _x[1] = 0.0; _x[2] = 0.0;
      return *this;
    }

  private:

    real _x[3];

  };

}

#endif
