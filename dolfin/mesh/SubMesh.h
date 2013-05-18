// Copyright (C) 2009 Anders Logg
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
// First added:  2009-02-11
// Last changed: 2013-02-11

#ifndef __SUB_MESH_H
#define __SUB_MESH_H

#include <map>
#include "Mesh.h"

namespace dolfin
{

  class SubDomain;
  template <typename T> class MeshFunction;

  /// A SubMesh is a mesh defined as a subset of a given mesh. It
  /// provides a convenient way to create matching meshes for
  /// multiphysics applications by creating meshes for subdomains as
  /// subsets of a single global mesh. A mapping from the vertices of
  /// the sub mesh to the vertices of the parent mesh is stored as the
  /// mesh data named "parent_vertex_indices".

  class SubMesh : public Mesh
  {
  public:

    /// Create subset of given mesh marked by sub domain
    SubMesh(const Mesh& mesh, const SubDomain& sub_domain);

    /// Create subset of given mesh marked by mesh function
    SubMesh(const Mesh& mesh, const MeshFunction<std::size_t>& sub_domains,
            std::size_t sub_domain);

    /// Create subset of given mesh from stored MeshValueCollection
    SubMesh(const Mesh& mesh, std::size_t sub_domain);

    /// Destructor
    ~SubMesh();

  private:

    /// Create sub mesh
    void init(const Mesh& mesh, const std::vector<std::size_t>& sub_domains,
              std::size_t sub_domain);

  };

}

#endif
