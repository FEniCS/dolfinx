// Copyright (C) 2012 Anders Logg
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
// First added:  2012-04-11
// Last changed: 2012-04-13

#ifndef __CSG_PRIMITIVES_3D_H
#define __CSG_PRIMITIVES_3D_H

#include "CSGPrimitive.h"

namespace dolfin
{

  // Declare all primitives inside namespace csg so they can be easily
  // accessed and at the same time won't clutter the top level
  // namespace where they might also conflict with existing classes
  // like Rectangle and Box.

  namespace csg
  {

    /// Base class for 3D primitives
    class CSGPrimitive3D : public CSGPrimitive
    {
    public:

      /// Return dimension of geometry
      uint dim() const { return 3; }

    };

    /// This class describes a 3D sphere which can be used to build
    /// geometries using Constructive Solid Geometry (CSG).
    class Sphere : public CSGPrimitive3D
    {
    public:

      /// Create sphere at x = (x0, x1, x2) with radius r.
      ///
      /// *Arguments*
      ///     x0 (double)
      ///         x0-coordinate of center.
      ///     x1 (double)
      ///         x1-coordinate of center.
      ///     x2 (double)
      ///         x2-coordinate of center.
      ///     r (double)
      ///         radius.
      Sphere(double x0, double x1, double x2, double r);

      /// Informal string representation
      std::string str(bool verbose) const;

    private:

      double _x0, _x1, _x2, _r;

    };

    /// This class describes a 3D box which can be used to build
    /// geometries using Constructive Solid Geometry (CSG).
    class Box : public CSGPrimitive3D
    {
    public:

      /// Create box defined by two opposite corners
      /// x = (x0, x1, x2) and y = (y0, y1, y2).
      ///
      /// *Arguments*
      ///     x0 (double)
      ///         x0-coordinate of first corner.
      ///     x1 (double)
      ///         x1-coordinate of first corner.
      ///     x2 (double)
      ///         x2-coordinate of first corner.
      ///     y0 (double)
      ///         y0-coordinate of second corner.
      ///     y1 (double)
      ///         y1-coordinate of second corner.
      ///     y2 (double)
      ///         y2-coordinate of second corner.
      Box(double x0, double x1, double x2,
          double y0, double y1, double y2);

      /// Informal string representation
      std::string str(bool verbose) const;

    private:

      double _x0, _x1, _x2, _y0, _y1, _y2;

    };

  }

}

#endif
