// Copyright (C) 2014-2015 Anders Logg
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
// First added:  2014-06-11
// Last changed: 2015-11-12

#ifndef __MULTI_MESH_SUB_SPACE_H
#define __MULTI_MESH_SUB_SPACE_H

#include <vector>
#include "MultiMeshFunctionSpace.h"

namespace dolfin
{

  /// This class represents a subspace (component) of a multimesh
  /// function space.
  ///
  /// The subspace is specified by an array of indices. For example,
  /// the array [3, 0, 2] specifies subspace 2 of subspace 0 of
  /// subspace 3.
  ///
  /// A typical example is the function space W = V x P for Stokes.
  /// Here, V = W[0] is the subspace for the velocity component and
  /// P = W[1] is the subspace for the pressure component. Furthermore,
  /// W[0][0] = V[0] is the first component of the velocity space etc.

  class MultiMeshSubSpace : public MultiMeshFunctionSpace
  {
  public:

    /// Create subspace for given component (one level)
    MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                      std::size_t component);

    /// Create subspace for given component (two levels)
    MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                      std::size_t component, std::size_t sub_component);

    /// Create subspace for given component (n levels)
    MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                      const std::vector<std::size_t>& component);

  private:

    // Build subspace
    void _build(MultiMeshFunctionSpace& V,
                const std::vector<std::size_t>& component);

  };

}

#endif
