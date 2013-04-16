// Copyright (C) 2013 Anders Logg
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
// First added:  2013-04-09
// Last changed: 2013-04-16

#ifndef __BOUNDING_BOX_TREE_H
#define __BOUNDING_BOX_TREE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;

  /// This class implements a (distributed) axis aligned bounding box
  /// tree (AABB tree). Bounding box trees can be created from meshes
  /// and [other data structures, to be filled in].

  class BoundingBoxTree
  {
  public:

    /// Create bounding box tree for cells of mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh);

    /// Create bounding box tree for mesh entites of given dimension.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (std::size_t)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh, std::size_t dimension);

    /// Destructor
    ~BoundingBoxTree();

  private:

    // Build bounding box tree
    void build(const Mesh& mesh, std::size_t dimension);

    // Compute bounding box of mesh entity
    void compute_bbox(double* bbox,
                      const MeshEntity& entity) const;

    // Geometric dimension
    std::size_t _gdim;

    // List coordinates stored as one contiguous array [x_i^j y_i^j]
    // row major on (i, j), where x_i^j denotes the j:th component of
    // the first vertex of the i:th bounding box, and where y_i^j
    // denotes j:th component of the second vertex of the i:th
    // bounding box.
    std::vector<double> _coordinates;


  };

}

#endif
