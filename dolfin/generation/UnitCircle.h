// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-11-09
// Last changed: 2012-11-09

#ifndef __UNIT_CIRCLE_H
#define __UNIT_CIRCLE_H

#include <dolfin/generation/UnitCircleMesh.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class UnitCircle : public UnitCircleMesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the unit circle.
    /// This class is deprecated. Use _UnitCircleMesh_.
    ///
    /// *Arguments*
    ///     n (std::size_t)
    ///         Resolution of the mesh.
    ///     diagonal (std::string)
    ///         Optional argument: A std::string indicating
    ///         the direction of the diagonals.
    ///     transformation (std::string)
    ///         Optional argument: A std::string indicating
    ///         the type of transformation used.
    UnitCircle(std::size_t n,
               std::string diagonal="crossed",
               std::string transformation="rotsumn")
      : UnitCircleMesh(n, diagonal, transformation)
      {
	warning("UnitCircle is deprecated. Use UnitCircleMesh.");
      }
  };

}

#endif
