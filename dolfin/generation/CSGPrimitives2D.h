// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-11
// Last changed: 2012-04-13

#ifndef __CSG_PRIMITIVES_2D_H
#define __CSG_PRIMITIVES_2D_H

#include "CSGPrimitive.h"

namespace dolfin
{

  // Declare all primitives inside namespace csg so they can be easily
  // accessed and at the same time won't clutter the top level
  // namespace where they might also conflict with existing classes
  // like Rectangle and Box.

  namespace csg
  {

    /// Base class for 2D primitives
    class CSGPrimitive2D : public CSGPrimitive
    {
    public:

      /// Return dimension of geometry
      uint dim() const { return 2; }

    };

    /// This class describes a 2D circle which can be used to build
    /// geometries using Constructive Solid Geometry (CSG).
    class Circle : public CSGPrimitive2D
    {
    public:

      /// Create circle at x = (x0, x1) with radius r.
      ///
      /// *Arguments*
      ///     x0 (double)
      ///         x0-coordinate of center.
      ///     x1 (double)
      ///         x1-coordinate of center.
      ///     r (double)
      ///         radius.
      Circle(double x0, double x1, double r);

      /// Informal string representation
      std::string str(bool verbose) const;

    private:

      double _x0, _x1, _r;

    };

    /// This class describes a 2D rectangle which can be used to build
    /// geometries using Constructive Solid Geometry (CSG).
    class Rectangle : public CSGPrimitive2D
    {
    public:

      /// Create rectangle defined by two opposite corners
      /// x = (x0, x1) and y = (y0, y1).
      ///
      /// *Arguments*
      ///     x0 (double)
      ///         x0-coordinate of first corner.
      ///     x1 (double)
      ///         x1-coordinate of first corner.
      ///     y0 (double)
      ///         y0-coordinate of second corner.
      ///     y1 (double)
      ///         y1-coordinate of second corner.
      Rectangle(double x0, double x1, double y0, double y1);

      /// Informal string representation
      std::string str(bool verbose) const;

    private:

      double _x0, _x1, _y0, _y1;

    };

  }

}

#endif
