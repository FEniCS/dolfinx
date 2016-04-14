// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg
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
// Modified by Jan Blechta 2013
//
// First added:  2008-05-02
// Last changed: 2013-03-06

#ifndef __ALE_H
#define __ALE_H

#include <memory>
#include "MeshDisplacement.h"

namespace dolfin
{

  class Mesh;
  class BoundaryMesh;
  class GenericFunction;

  /// This class provides functionality useful for implementation of
  /// ALE (Arbitrary Lagrangian-Eulerian) methods, in particular
  /// moving the boundary vertices of a mesh and then interpolating
  /// the new coordinates for the interior vertices accordingly.

  class ALE
  {
  public:


    /// Move coordinates of mesh according to new boundary coordinates.
    /// Works only for affine meshes.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The affine mesh to move.
    ///     boundary (_BoundaryMesh_)
    ///         An affine mesh containing just the boundary cells.
    ///
    /// *Returns*
    ///     MeshDisplacement
    ///         Displacement encapsulated in Expression subclass
    ///         MeshDisplacement.
    static std::shared_ptr<MeshDisplacement>
      move(std::shared_ptr<Mesh> mesh, const BoundaryMesh& new_boundary);

    /// Move coordinates of mesh according to adjacent mesh with
    /// common global vertices. Works only for affine meshes.
    //
    /// *Arguments*
    ///     mesh0 (_Mesh_)
    ///         The affine mesh to move.
    ///     mesh1 (_Mesh_)
    ///         The affine mesh to be fit.
    ///
    /// *Returns*
    ///     MeshDisplacement
    ///         Displacement encapsulated in Expression subclass
    static std::shared_ptr<MeshDisplacement> move(std::shared_ptr<Mesh> mesh0,
                                                  const Mesh& mesh1);

    /// Move coordinates of mesh according to displacement function.
    /// This works only for affine meshes.
    ///
    /// NOTE: This cannot be implemented for higher-order geometries
    ///       as there is no way of constructing function space for
    ///       position unless supplied as an argument.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The affine mesh to move.
    ///     displacement (_GenericFunction_)
    ///         A vectorial generic function.
    static void move(Mesh& mesh, const GenericFunction& displacement);

    /// Move coordinates of mesh according to displacement function.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to move.
    ///     displacement (_Function_)
    ///         A vectorial Lagrange function of matching degree.
    static void move(Mesh& mesh, const Function& displacement);

  };

}

#endif
