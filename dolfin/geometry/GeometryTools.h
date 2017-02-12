// Copyright (C) 2017 Anders Logg
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
// First added:  2017-02-11
// Last changed: 2017-02-12

#ifndef __GEOMETRY_TOOLS_H
#define __GEOMETRY_TOOLS_H

#include "predicates.h"
#include "Point.h"

namespace dolfin
{

  /// This class provides useful tools (functions) for computational geometry.

  class GeometryTools
  {
  public:

    // Compute numerically stable cross product (a - c) x (b - c)
    static inline Point cross_product(const Point& a, const Point& b, const Point& c)
    {
      // See Shewchuk Lecture Notes on Geometric Robustness
      double ayz[2] = {a.y(), a.z()};
      double byz[2] = {b.y(), b.z()};
      double cyz[2] = {c.y(), c.z()};
      double azx[2] = {a.z(), a.x()};
      double bzx[2] = {b.z(), b.x()};
      double czx[2] = {c.z(), c.x()};
      double axy[2] = {a.x(), a.y()};
      double bxy[2] = {b.x(), b.y()};
      double cxy[2] = {c.x(), c.y()};
      return Point (_orient2d(ayz, byz, cyz),
                    _orient2d(azx, bzx, czx),
                    _orient2d(axy, bxy, cxy));
    }

    /// Compute major (largest) axis of vector (2D)
    static inline std::size_t major_axis_2d(const Point& v)
    {
      return (std::abs(v.x()) >= std::abs(v.y()) ? 0 : 1);
    }

    /// Compute major (largest) axis of vector (3D)
    static inline std::size_t major_axis_3d(const Point& v)
    {
      const double vx = std::abs(v.x());
      const double vy = std::abs(v.y());
      const double vz = std::abs(v.z());
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
      case 0: return Point(p.y(), p.z());
      case 1: return Point(p.x(), p.z());
      case 2: return Point(p.x(), p.y());
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

#endif
