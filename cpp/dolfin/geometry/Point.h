// Copyright (C) 2006-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <dolfin/log/log.h>
#include <iostream>

namespace dolfin
{
namespace geometry
{

/// A Point represents a point in \f$ \mathbb{R}^3 \f$ with
/// coordinates (x, y, z) or alternatively, a vector in
/// \f$ \mathbb{R}^3 \f$, supporting standard operations like the
/// norm, distances, scalar and vector products etc.

class Point
{
public:
  /// Create a point at (x, y, z). Default value (0, 0, 0).
  ///
  /// @param    x (double)
  ///         The x-coordinate.
  /// @param    y (double)
  ///         The y-coordinate.
  /// @param    z (double)
  ///         The z-coordinate.
  explicit Point(const double x = 0.0, const double y = 0.0,
                 const double z = 0.0)
      : _x({{x, y, z}})
  {
  }

  /// Create point from array
  ///
  /// @param    dim (std::size_t)
  ///         Dimension of the array.
  /// @param    x (double)
  ///         The array to create a Point from.
  Point(std::size_t dim, const double* x) : _x({{0.0, 0.0, 0.0}})
  {
    for (std::size_t i = 0; i < dim; i++)
      _x[i] = x[i];
  }

  /// Create point from Array
  ///
  /// @param    x (Array<double>)
  ///         Array of coordinates.
  Point(const Eigen::VectorXd& x) : _x({{0.0, 0.0, 0.0}})
  {
    for (int i = 0; i < x.size(); i++)
      _x[i] = x[i];
  }

  /// Copy constructor
  ///
  /// @param    p (_Point_)
  ///         The object to be copied.
  Point(const Point& p) = default;

  /// Move constructor
  ///
  /// @param    p (_Point_)
  ///         The object to be moves.
  Point(Point&& p) = default;

  /// Destructor
  ~Point() = default;

  /// Return address of coordinate in direction i
  ///
  /// @param    i (std::size_t)
  ///         Direction.
  ///
  /// @returns    double
  ///         Address of coordinate in the given direction.
  double& operator[](std::size_t i)
  {
    assert(i < 3);
    return _x[i];
  }

  /// Return coordinate in direction i
  ///
  /// @param    i (std::size_t)
  ///         Direction.
  ///
  /// @return    double
  ///         The coordinate in the given direction.
  double operator[](std::size_t i) const
  {
    assert(i < 3);
    return _x[i];
  }

  /// Return coordinate array
  ///
  /// @return double*
  ///         The coordinates.
  double* coordinates() { return _x.data(); }

  /// Return coordinate array (const. version)
  ///
  /// @return double*
  ///         The coordinates.
  const double* coordinates() const { return _x.data(); }

  /// Return copy of coordinate array
  ///
  /// @returns std::array<double, 3>
  ///         The coordinates.
  std::array<double, 3> array() const { return _x; }

  /// Compute sum of two points
  /// @param p (Point)
  /// @return Point
  Point operator+(const Point& p) const
  {
    return Point(_x[0] + p._x[0], _x[1] + p._x[1], _x[2] + p._x[2]);
  }

  /// Compute difference of two points
  /// @param p (Point)
  /// @return Point
  Point operator-(const Point& p) const
  {
    return Point(_x[0] - p._x[0], _x[1] - p._x[1], _x[2] - p._x[2]);
  }

  /// Add given point
  const Point& operator+=(const Point& p)
  {
    _x[0] += p._x[0];
    _x[1] += p._x[1];
    _x[2] += p._x[2];
    return *this;
  }

  /// Subtract given point
  const Point& operator-=(const Point& p)
  {
    _x[0] -= p._x[0];
    _x[1] -= p._x[1];
    _x[2] -= p._x[2];
    return *this;
  }

  /// Unary minus
  Point operator-() { return Point(-_x[0], -_x[1], -_x[2]); }

  /// Multiplication with scalar
  Point operator*(double a) const
  {
    return Point(a * _x[0], a * _x[1], a * _x[2]);
  }

  /// Incremental multiplication with scalar
  const Point& operator*=(double a)
  {
    _x[0] *= a;
    _x[1] *= a;
    _x[2] *= a;
    return *this;
  }

  /// Division by scalar
  Point operator/(double a) const
  {
    return Point(_x[0] / a, _x[1] / a, _x[2] / a);
  }

  /// Incremental division by scalar
  const Point& operator/=(double a)
  {
    _x[0] /= a;
    _x[1] /= a;
    _x[2] /= a;
    return *this;
  }

  /// Assignment operator
  Point& operator=(const Point& p) = default;

  /// Move assignment operator
  Point& operator=(Point&& p) = default;

  /// Equal to operator
  bool operator==(const Point& p) const { return _x == p._x; }

  /// Not equal to operator
  bool operator!=(const Point& p) const { return _x != p._x; }

  /// Compute squared distance to given point
  ///
  /// @param p (Point)
  ///         The point to compute distance to.
  ///
  /// @return double
  ///         The squared distance.
  ///
  double squared_distance(const Point& p) const;

  /// Compute distance to given point
  ///
  /// @param    p (Point)
  ///         The point to compute distance to.
  ///
  /// @return   double
  ///         The distance.
  ///
  /// @code{.cpp}
  ///
  ///         Point p1(0, 4, 0);
  ///         Point p2(2, 0, 4);
  ///         log::info("%g", p1.distance(p2));
  ///
  /// @endcode
  inline double distance(const Point& p) const
  {
    return sqrt(squared_distance(p));
  }

  /// Compute norm of point representing a vector from the origin
  ///
  /// @return     double
  ///         The (Euclidean) norm of the vector from the origin to
  ///         the point.
  ///
  /// @code{.cpp}
  ///
  ///         Point p(1.0, 2.0, 2.0);
  ///         log::info("%g", p.norm());
  ///
  /// @endcode
  double norm() const
  {
    return std::sqrt(_x[0] * _x[0] + _x[1] * _x[1] + _x[2] * _x[2]);
  }

  /// Compute cross product with given vector
  ///
  /// @param    p (_Point_)
  ///         Another point.
  ///
  /// @return     Point
  ///         The cross product.
  const Point cross(const Point& p) const;

  /// Compute dot product with given vector
  ///
  /// @param    p (Point)
  ///         Another point.
  ///
  /// @return    double
  ///         The dot product.
  ///
  /// @code{.cpp}
  ///
  ///         Point p1(1.0, 4.0, 8.0);
  ///         Point p2(2.0, 0.0, 0.0);
  ///         log::info("%g", p1.dot(p2));
  ///
  /// @endcode
  double dot(const Point& p) const;

  /// Rotate around a given axis
  ///
  /// @param    a (Point)
  ///         The axis to rotate around. Must be unit length.
  /// @param    theta (double)
  ///         The rotation angle.
  ///
  /// @return     Point
  ///         The rotated point.
  Point rotate(const Point& a, double theta) const;

  // Note: Not a subclass of Variable for efficiency!

  /// Return informal string representation (pretty-print)
  ///
  /// @param    verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return     std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose = false) const;

private:
  std::array<double, 3> _x;
};
}
}
/// Output of Point to stream
inline std::ostream& operator<<(std::ostream& stream,
                                const dolfin::geometry::Point& point)
{
  stream << point.str(false);
  return stream;
}
