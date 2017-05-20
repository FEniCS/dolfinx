// Copyright (C) 2008 Anders Logg
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
// First added:  2008-08-11
// Last changed: 2013-03-06

#ifndef __HARMONIC_SMOOTHING_H
#define __HARMONIC_SMOOTHING_H

#include <memory>
#include "MeshDisplacement.h"

namespace dolfin
{

  class BoundaryMesh;
  class Mesh;

  /// This class implements harmonic mesh smoothing. Poisson's
  /// equation is solved with zero right-hand side (Laplace's
  /// equation) for each coordinate direction to compute new
  /// coordinates for all vertices, given new locations for the
  /// coordinates of the boundary.

  class HarmonicSmoothing
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    /// and return the displacement
    ///
    /// @param mesh (Mesh)
    ///   Mesh
    /// @param new_boundary (BoundaryMesh)
    ///   Boundary mesh
    ///
    /// @return MeshDisplacement
    ///   Displacement
    static std::shared_ptr<MeshDisplacement>
      move(std::shared_ptr<Mesh> mesh, const BoundaryMesh& new_boundary);

  };

}

#endif
