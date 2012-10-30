// Copyright (C) 2006-2012 Anders Logg
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
// Modified by Niclas Jansson 2009.
// Modified by Joachim B Haga 2012.
//
// First added:  2006-06-21
// Last changed: 2012-10-30

#ifndef __BOUNDARY_MESH_H
#define __BOUNDARY_MESH_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"
#include "Mesh.h"

namespace dolfin
{
  /// A BoundaryMesh is a mesh over the boundary of some given mesh.
  /// The cells of the boundary mesh (facets of the original mesh) are
  /// oriented to produce outward pointing normals relative to the
  /// original mesh.

  class BoundaryMesh : public Mesh
  {
  public:

    /// Create an empty boundary mesh
    BoundaryMesh();

    /// Create boundary mesh from given mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Another _Mesh_ object.
    ///     order (bool)
    ///         Optional argument which can be used to control whether
    ///         or not the boundary mesh should be ordered according
    ///         to the UFC ordering convention. If set to false, the
    ///         boundary mesh will be ordered with right-oriented
    ///         facets (outward-pointing unit normals). The default
    ///         value is true.
    BoundaryMesh(const Mesh& mesh, bool order=true);

    /// Destructor
    ~BoundaryMesh();

    /// Initialize exterior boundary of given mesh
    void init_exterior_boundary(const Mesh& mesh);

    /// Initialize interior boundary of given mesh
    void init_interior_boundary(const Mesh& mesh);

    MeshFunction<unsigned int>& cell_map()
    { return _cell_map; }

    /// Get cell mapping from the boundary mesh to the original full mesh
    const MeshFunction<unsigned int>& cell_map() const
    { return _cell_map; }

    /// Get vertex mapping from the boundary mesh to the original full mesh
    MeshFunction<unsigned int>& vertex_map()
    { return _vertex_map; }

    const MeshFunction<unsigned int>& vertex_map() const
    { return _vertex_map; }

  private:

    MeshFunction<unsigned int> _cell_map;

    MeshFunction<unsigned int> _vertex_map;

  };

}

#endif
