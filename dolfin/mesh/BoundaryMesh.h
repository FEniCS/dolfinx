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

#include <string>
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

    /// Create boundary mesh from given mesh.
    ///
    /// @param      mesh (_Mesh_)
    ///         Another _Mesh_ object.
    /// @param     type (_std::string_)
    ///         The type of BoundaryMesh, which can be "exterior",
    ///         "interior" or "local". "exterior" is the globally
    ///         external boundary, "interior" is the inter-process mesh
    ///         and "local" is the boundary of the local (this process)
    ///         mesh.
    /// @param     order (bool)
    ///         Optional argument which can be used to control whether
    ///         or not the boundary mesh should be ordered according
    ///         to the UFC ordering convention. If set to false, the
    ///         boundary mesh will be ordered with right-oriented
    ///         facets (outward-pointing unit normals). The default
    ///         value is true.
    BoundaryMesh(const Mesh& mesh, std::string type, bool order=true);

    /// Destructor
    ~BoundaryMesh();

    /// Get index map for entities of dimension d in the boundary mesh
    /// to the entity in the original full mesh
    MeshFunction<std::size_t>& entity_map(std::size_t d);

    /// Get index map for entities of dimension d in the boundary mesh
    /// to the entity in the original full mesh (const version)
    const MeshFunction<std::size_t>& entity_map(std::size_t d) const;

  private:

    BoundaryMesh() {}

    MeshFunction<std::size_t> _cell_map;

    MeshFunction<std::size_t> _vertex_map;

  };

}

#endif
