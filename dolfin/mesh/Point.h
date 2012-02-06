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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2006.
// Modified by Andre Massing, 2009.
//
// First added:  2006-06-12
// Last changed: 2011-04-13

#ifndef __POINT_H
#define __POINT_H

#include <iostream>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

#ifdef HAS_CGAL
#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>
#endif

namespace dolfin
{
  /// A Point represents a point in :math:`\mathbb{R}^3` with
  /// coordinates :math:`x, y, z,` or alternatively, a vector in
  /// :math:`\mathbb{R}^3`, supporting standard operations like the
  /// norm, distances, scalar and vector products etc.

  class Point
  {
  public:

    /// Create a point at (x, y, z). Default value (0, 0, 0).
    ///
    /// *Arguments*
    ///     x (double)
    ///         The x-coordinate.
    ///     y (double)
    ///         The y-coordinate.
    ///     z (double)
    ///         The z-coordinate.
    Point(const double x=0.0, const double y=0.0, const double z=0.0)
    { _x[0] = x; _x[1] = y; _x[2] = z; }

    /// Create point from array
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         Dimension of the array.
    ///     x (double)
    ///         The array to create a Point from.
    Point(uint dim, const double* x)
    { for (uint i = 0; i < 3; i++) _x[i] = (i < dim ? x[i] : 0.0); }

    /// Copy constructor
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         The object to be copied.
    Point(const Point& p)
    { _x[0] = p._x[0]; _x[1] = p._x[1]; _x[2] = p._x[2]; }

    /// Destructor
    ~Point() {};

    /// Return address of coordinate in direction i
    ///
    /// *Arguments*
    ///     i (uint)
    ///         Direction.
    ///
    /// *Returns*
    ///     double
    ///         Adress of coordinate in the given direction.
    double& operator[] (uint i)
    { dolfin_assert(i < 3); return _x[i]; }

    /// Return coordinate in direction i
    ///
    /// *Arguments*
    ///     i (uint)
    ///         Direction.
    ///
    /// *Returns*
    ///     double
    ///         The coordinate in the given direction.
    double operator[] (uint i) const
    { dolfin_assert(i < 3); return _x[i]; }

    /// Return x-coordinate
    ///
    /// *Returns*
    ///     double
    ///         The x-coordinate.
    double x() const
    { return _x[0]; }

    /// Return y-coordinate
    ///
    /// *Returns*
    ///     double
    ///         The y-coordinate.
    double y() const
    { return _x[1]; }

    /// Return z-coordinate
    ///
    /// *Returns*
    ///     double
    ///         The z-coordinate.
    double z() const
    { return _x[2]; }

    /// Return coordinate array
    ///
    /// *Returns*
    ///     list of doubles
    ///         The coordinates.
    double* coordinates()
    { return _x; }

    /// Return coordinate array (const. version)
    ///
    /// *Returns*
    ///     list of doubles
    ///         The coordinates.
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
    /// Conversion operator to appropriate CGAL Point_3 class.
    template <typename Kernel>
    operator CGAL::Point_3<Kernel>() const
    { return CGAL::Point_3<Kernel>(_x[0],_x[1],_x[2]); }

    /// Constructor taking a CGAL::Point_3. Allows conversion from
    /// CGAL Point_3 class to Point class.
    template <typename Kernel>
    Point (const CGAL::Point_3<Kernel> & point)
    { _x[0] = point.x(); _x[1] = point.y(); _x[2] = point.z(); }

    /// Provides a CGAL bounding box, using conversion operator.
    template <typename Kernel>
    CGAL::Bbox_3  bbox()
    { return CGAL::Point_3<Kernel>(*this).bbox(); }
    #endif

    /// Compute distance to given point
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         The point to compute distance to.
    ///
    /// *Returns*
    ///     double
    ///         The distance.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Point p1(0, 4, 0);
    ///         Point p2(2, 0, 4);
    ///         info("%g", p1.distance(p2));
    ///
    ///     output::
    ///
    ///         6
    double distance(const Point& p) const;

    /// Compute norm of point representing a vector from the origin
    ///
    /// *Returns*
    ///     double
    ///         The (Euclidean) norm of the vector from the origin to
    ///         the point.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Point p(1.0, 2.0, 2.0);
    ///         info("%g", p.norm());
    ///
    ///     output::
    ///
    ///         3
    double norm() const;

    /// Compute cross product with given vector
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         Another point.
    ///
    /// *Returns*
    ///     Point
    ///         The cross product.
    const Point cross(const Point& p) const;

    /// Compute dot product with given vector
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         Another point.
    ///
    /// *Returns*
    ///     double
    ///         The dot product.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Point p1(1.0, 4.0, 8.0);
    ///         Point p2(2.0, 0.0, 0.0);
    ///         info("%g", p1.dot(p2));
    ///
    ///     output::
    ///
    ///         2
    double dot(const Point& p) const;

    // Note: Not a subclass of Variable for efficiency!

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation of the function space.
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
