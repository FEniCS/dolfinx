// Copyright (C) 2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Point.h"
#include "predicates.h"

namespace dolfin
{
namespace geometry
{

/// This class provides useful tools (functions) for computational geometry.

class GeometryTools
{
public:
  /// Compute numerically stable cross product (a - c) x (b - c)
  static inline Point cross_product(const Point& a, const Point& b,
                                    const Point& c)
  {
    // See Shewchuk Lecture Notes on Geometric Robustness
    double ayz[2] = {a[1], a[2]};
    double byz[2] = {b[1], b[2]};
    double cyz[2] = {c[1], c[2]};
    double azx[2] = {a[2], a[0]};
    double bzx[2] = {b[2], b[0]};
    double czx[2] = {c[2], c[0]};
    double axy[2] = {a[0], a[1]};
    double bxy[2] = {b[0], b[1]};
    double cxy[2] = {c[0], c[1]};
    return Point(_orient2d(ayz, byz, cyz), _orient2d(azx, bzx, czx),
                 _orient2d(axy, bxy, cxy));
  }

  /// Compute determinant of 3 x 3 matrix defined by vectors, ab, dc, ec
  inline double determinant(const Point& ab, const Point& dc, const Point& ec)
  {
    const double a = ab[0], b = ab[1], c = ab[2];
    const double d = dc[0], e = dc[1], f = dc[2];
    const double g = ec[0], h = ec[1], i = ec[2];
    return a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g);
  }

  /// Compute major (largest) axis of vector (2D)
  static inline std::size_t major_axis_2d(const Point& v)
  {
    return (std::abs(v[0]) >= std::abs(v[1]) ? 0 : 1);
  }

  /// Compute major (largest) axis of vector (3D)
  static inline std::size_t major_axis_3d(const Point& v)
  {
    const double vx = std::abs(v[0]);
    const double vy = std::abs(v[1]);
    const double vz = std::abs(v[2]);
    if (vx >= vy && vx >= vz)
      return 0;
    if (vy >= vz)
      return 1;
    return 2;
  }

  /// Project point to axis (2D)
  static inline double project_to_axis_2d(const Point& p, std::size_t axis)
  {
    dolfin_assert(axis <= 1);
    return p[axis];
  }

  /// Project point to plane (3D)
  static inline Point project_to_plane_3d(const Point& p, std::size_t axis)
  {
    dolfin_assert(axis <= 2);
    switch (axis)
    {
    case 0:
      return Point(p[1], p[2]);
    case 1:
      return Point(p[0], p[2]);
    case 2:
      return Point(p[0], p[1]);
    }
    return p;
  }

  /// Check whether x in [a, b]
  static inline bool contains(double a, double b, double x)
  {
    return a <= x and x <= b;
  }

  /// Check whether x in (a, b)
  static inline bool contains_strict(double a, double b, double x)
  {
    return a < x and x < b;
  }
};
}
}
