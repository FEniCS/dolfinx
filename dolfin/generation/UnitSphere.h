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

#ifndef __UNIT_SPHERE_H
#define __UNIT_SPHERE_H

#include "UnitSphereMesh.h"

namespace dolfin
{

  /// Tetrahedral mesh of the unit sphere.
  /// This class has been deprecated. Use _UnitSphereMesh_.
  class UnitSphere : public UnitSphereMesh
  {
  public:

    /// WARNING:
    ///
    /// The UnitSphere class is broken and should not be used for computations.
    /// It generates meshes of very bad quality (very thin tetrahedra).
    ///
    /// Create a uniform finite element _Mesh_ over the unit sphere.
    ///
    /// *Arguments*
    ///     n (uint)
    ///         Resolution of the mesh.
    ///
    /// This class is deprecated. Use _UnitSquareMesh_.
    UnitSphere(uint n) 
      : UnitSphereMesh(n)
    {
      warning("UnitSphere is deprecated. Use UnitSphereMesh.");
    }
  };
}

#endif
