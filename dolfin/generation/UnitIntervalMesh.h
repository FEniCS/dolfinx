// Copyright (C) 2007 Kristian B. Oelgaard
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
// Modified by Anders Logg, 2010.
// Modified by Benjamin Kehlet 2012
//
// First added:  2007-11-23
// Last changed: 2012-11-09

#ifndef __UNIT_INTERVAL_MESH_H
#define __UNIT_INTERVAL_MESH_H

#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// A mesh of the unit interval (0, 1) with a given number of cells
  /// (nx) in the axial direction. The total number of intervals will
  /// be nx and the total number of vertices will be (nx + 1).

  class UnitIntervalMesh : public Mesh
  {
  public:

    /// Create mesh of unit interval
    UnitIntervalMesh(uint nx=1);

  };

}

#endif
