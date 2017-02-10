// Copyright (C) 2006-2014 Anders Logg
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
// Modified by Garth N. Wells 2006
// Modified by Andre Massing 2009
//
// First added:  2006-06-12
// Last changed: 2014-06-06

#ifndef __POINT_H
#define __POINT_H

#include <array>
#include <cmath>
#include <iostream>
#include <dolfin/log/log.h>
#include <dolfin/common/Array.h>

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
    explicit Point(const double x=0.0, const double y=0.0, const double z=0.0)
      : _x({{x, y, z}}) {}

    /// Create point from array
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         Dimension of the array.
    ///     x (double)
    ///         The array to create a Point from.
    Point(std::size_t dim, const double* x) : _x({{0.0, 0.0, 0.0}})
    {
      for (std::size_t i = 0; i < dim; i++)
        _x[i] = x[i];
    }

    /// Create point from Array
    ///
    /// *Arguments*
    ///     x (Array<double>)
    ///         Array of coordinates.
    Point(const Array<double>& x) : _x({{0.0, 0.0, 0.0}})
    {
      for (std::size_t i = 0; i < x.size(); i++)
        _x[i] = x[i];
    }

    /// Copy constructor
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         The object to be copied.
    Point(const Point& p) : _x({{p._x[0], p._x[1], p._x[2]}}) {}

    /// Destructor
    ~Point() {}

    /// Return address of coordinate in direction i
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Direction.
    ///
    /// *Returns*
    ///     double
    ///         Address of coordinate in the given direction.
    double& operator[] (std::size_t i)
    { dolfin_assert(i < 3); return _x[i]; }

    /// Return coordinate in direction i
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Direction.
    ///
    /// *Returns*
    ///     double
    ///         The coordinate in the given direction.
    double operator[] (std::size_t i) const
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
    { return _x.data(); }

    /// Return coordinate array (const. version)
    ///
    /// *Returns*
    ///     list of doubles
    ///         The coordinates.
    const double* coordinates() const
    { return _x.data(); }

    /// Return array for Python interface
    ///
    /// *Returns*
    ///     list of double
    ///         Coordinate array
    std::array<double, 3> array() const
    {
      return _x;
    }

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

    /// Unary minus
    Point operator- ()
    { Point p(-_x[0], -_x[1], -_x[2]); return p; }

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
    { _x = {{p._x[0], p._x[1], p._x[2]}}; return *this; }

    /// Compute squared distance to given point
    ///
    /// *Arguments*
    ///     p (_Point_)
    ///         The point to compute distance to.
    ///
    /// *Returns*
    ///     double
    ///         The squared distance.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Point p1(0, 4, 0);
    ///         Point p2(2, 0, 4);
    ///         info("%g", p1.squared_distance(p2));
    ///
    ///     output::
    ///
    ///         6
    double squared_distance(const Point& p) const;

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
    inline double distance(const Point& p) const
    { return sqrt(squared_distance(p)); }

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
    double norm() const
    { return std::sqrt(_x[0]*_x[0] + _x[1]*_x[1] + _x[2]*_x[2]); }

    /// Compute norm of point representing a vector from the origin
    ///
    /// *Returns*
    ///     double
    ///         The squared (Euclidean) norm of the vector from the
    ///         origin of the point.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         Point p(1.0, 2.0, 2.0);
    ///         info("%g", p.squared_norm());
    ///
    ///     output::
    ///
    ///         9
    double squared_norm() const
    { return _x[0]*_x[0] + _x[1]*_x[1] + _x[2]*_x[2]; }

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

    /// Rotate around a given axis
    ///
    /// *Arguments*
    ///     a (_Point_)
    ///         The axis to rotate around. Must be unit length.
    ///     theta (_double_)
    ///         The rotation angle.
    ///
    /// *Returns*
    ///     Point
    ///         The rotated point.
    Point rotate(const Point& a, double theta) const;

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

    std::array<double, 3> _x;

  };

  /// Multiplication with scalar
  inline Point operator*(double a, const Point& p)
  { return p*a; }

  inline std::ostream& operator<<(std::ostream& stream, const Point& point)
  { stream << point.str(false); return stream; }

}

#endif
