// Copyright (C) 2006-2008 Anders Logg
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
//
// First added:  2006-06-21
// Last changed: 2010-02-08

#ifndef __BOUNDARY_MESH_H
#define __BOUNDARY_MESH_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"
#include "Mesh.h"

namespace dolfin
{

  /// A BoundaryMesh is a mesh over the boundary of some given mesh.

  class BoundaryMesh : public Mesh
  {
  public:

    /// Create an empty boundary mesh
    BoundaryMesh();

    /// Create (interior) boundary mesh from given mesh
    BoundaryMesh(const Mesh& mesh);

    /// Destructor
    ~BoundaryMesh();

    /// Initialize exterior boundary of given mesh
    void init_exterior_boundary(const Mesh& mesh);

    /// Initialize interior boundary of given mesh
    void init_interior_boundary(const Mesh& mesh);

    MeshFunction<unsigned int>& cell_map()
    { return _cell_map; }

    const MeshFunction<unsigned int>& cell_map() const
    { return _cell_map; }

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
