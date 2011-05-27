// Copyright (C) 2006-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2006.
// Modified by Andre Massing, 2009.
//
// First added:  2006-06-12
// Last changed: 2011-04-13

#ifndef __POINT_H
#define __POINT_H

#include <iostream>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>

#ifdef HAS_CGAL
#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>
#endif

namespace dolfin
{
  /// A Point represents a point in R^3 with coordinates x, y, z, or,
  /// alternatively, a vector in R^3, supporting standard operations
  /// like the norm, distances, scalar and vector products etc.

  class Point
  {
  public:

    /// Create a point at (x, y, z)
    Point(const double x = 0.0, const double y = 0.0, const double z =0.0)
    { _x[0] = x; _x[1] = y; _x[2] = z; }

    /// Create point from array
    Point(uint dim, const double* x)
    { for (uint i = 0; i < 3; i++) _x[i] = (i < dim ? x[i] : 0.0); }

    /// Copy constructor
    Point(const Point& p)
    { _x[0] = p._x[0]; _x[1] = p._x[1]; _x[2] = p._x[2]; }

    /// Destructor
    ~Point() {};

    /// Return address of coordinate in direction i
    double& operator[] (uint i)
    { assert(i < 3); return _x[i]; }

    /// Return coordinate in direction i
    double operator[] (uint i) const
    { assert(i < 3); return _x[i]; }

    /// Return x-coordinate
    double x() const
    { return _x[0]; }

    /// Return y-coordinate
    double y() const
    { return _x[1]; }

    /// Return z-coordinate
    double z() const
    { return _x[2]; }

    /// Return coordinate array
    double* coordinates()
    { return _x; }

    /// Return coordinate array
    const double* coordinates() const
    { return _x; }

    /// Compute sum of two points
    Point operator+ (const Point& p) const
    { Point q(_x[0] + p._x[0], _x[1] + p._x[1], _x[2] + p._x[2]); return q; }

    /// Compute difference of two points
    Point operator- (const Point& p) const
    { Point q(_x[0] - p._x[0], _x[1] - p._x[1], _x[2] - p._x[2]); return q; }

    /// Add given point
    const Point& operator+= (const Point& p)
    { _x[0] += p._x[0]; _x[1] += p._x[1]; _x[2] += p._x[2]; return *this; }

    /// Subtract given point
    const Point& operator-= (const Point& p)
    { _x[0] -= p._x[0]; _x[1] -= p._x[1]; _x[2] -= p._x[2]; return *this; }

    /// Multiplication with scalar
    Point operator* (double a) const
    { Point p(a*_x[0], a*_x[1], a*_x[2]); return p; }

    /// Incremental multiplication with scalar
    const Point& operator*= (double a)
    { _x[0] *= a; _x[1] *= a; _x[2] *= a; return *this; }

    /// Division by scalar
    Point operator/ (double a) const
    { Point p(_x[0]/a, _x[1]/a, _x[2]/a); return p; }

    /// Incremental division by scalar
    const Point& operator/= (double a)
    { _x[0] /= a; _x[1] /= a; _x[2] /= a; return *this; }

    /// Assignment operator
    const Point& operator= (const Point& p)
    { _x[0] = p._x[0]; _x[1] = p._x[1]; _x[2] = p._x[2]; return *this; }

    #ifdef HAS_CGAL
    ///Conversion operator to appropriate CGAL Point_3 class.
    template <typename Kernel>
    operator CGAL::Point_3<Kernel>() const
    { return CGAL::Point_3<Kernel>(_x[0],_x[1],_x[2]); }

    ///Constructor taking a CGAL::Point_3. Allows conversion from CGAL Point_3 class to Point class.
    template <typename Kernel>
    Point (const CGAL::Point_3<Kernel> & point)
    { _x[0] = point.x(); _x[1] = point.y(); _x[2] = point.z(); }

    ///Provides a CGAL bounding box, using conversion operator.
    template <typename Kernel>
    CGAL::Bbox_3  bbox()
    { return CGAL::Point_3<Kernel>(*this).bbox(); }
    #endif

    /// Compute distance to given point
    double distance(const Point& p) const;

    /// Compute norm of point representing a vector from the origin
    double norm() const;

    /// Compute cross product with given vector
    const Point cross(const Point& p) const;

    /// Compute dot product with given vector
    double dot(const Point& p) const;

    // Note: Not a subclass of Variable for efficiency!

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose=false) const;

  private:

    double _x[3];

  };

  /// Multiplication with scalar
  inline Point operator*(double a, const Point& p) { return p*a; }

  inline std::ostream& operator<<(std::ostream& stream, const Point& point)
  { stream << point.str(false); return stream; }

}

#endif
