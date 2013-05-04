// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-05-04
// Last changed:

#ifndef __ISOSURFACE_H
#define __ISOSURFACE_H

#include <string>
#include <dolfin/generation/CSGPrimitives3D.h>

namespace dolfin
{

  // Forward declaration
  class Point;

  /// This class is used to define an isosurface f(x) -> R, where for
  /// a point y on the surface f(y) = 0. It typically used for the
  /// implicit representation of a surface.

  class IsoSurface
  {
  public:

    /// Create an isosurface
    ///
    /// *Arguments*
    ///     s (Sphere)
    ///         Bounding sphere for surface.
    ///
    ///     type (std::string)
    ///         Isosurface type. One of "manifold", "manifold_with_edges"
    ///         or "non_manifold".
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         IsoSurface isosurface("manifold_with_edges");
    ///
    IsoSurface(Sphere s, std::string type);

    /// Destructor
    virtual ~IsoSurface();

    /// Return value of isosurfacce function. This function is
    /// overloaed by the user.
    ///
    /// *Arguments*
    ///     point (Point)
    ///         The point at which to evaluate the isosurface function.
    ///         or "non_manifold".
    ///
    /// *Returns*
    ///     double
    ///         Isosurface function value.
    //virtual double value(const Point& point) const = 0;
    virtual double operator()(const Point& point) const = 0;

    /// Bounding sphere
    const Sphere sphere;

  private:

    // Isosurface type
    const std::string _type;

  };

}

#endif
