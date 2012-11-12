// Copyright (C) 2008 Nuno Lopes
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
// Modified by Anders Logg 2012
// Modified by Benjamin Kehlet 2012
//
// First added:  2008-07-15
// Last changed: 2012-11-09

#ifndef __UNIT_SPHERE_MESH_H
#define __UNIT_SPHERE_MESH_H

#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Tetrahedral mesh of the unit sphere.

  class UnitSphereMesh : public Mesh
  {
  public:

    /// WARNING:
    ///
    /// The UnitSphereMesh class is broken and should not be used for computations.
    /// It generates meshes of very bad quality (very thin tetrahedra).
    ///
    /// Create a uniform finite element _Mesh_ over the unit sphere.
    ///
    /// *Arguments*
    ///     n (uint)
    ///         Resolution of the mesh.
    UnitSphereMesh(uint n);

  private:

    std::vector<double> transform(const std::vector<double>& x) const;

    double max(const std::vector<double>& x) const;

  };

}

#endif
