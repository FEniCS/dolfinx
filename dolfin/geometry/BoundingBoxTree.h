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
// Last changed: 2013-04-17

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
    ///     dimension (unsigned int)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh, unsigned int dimension);

    /// Destructor
    ~BoundingBoxTree();

    /// Get bounding box for given node in tree
    ///
    /// *Arguments*
    ///     node (unsigned int)
    ///         The number of the node (breadth-first numbering, see above).
    inline double* get_bbox(unsigned int node)
    {
      dolfin_assert(2*_gdim*(node + 1) <= bbox_tree.size());
      return bbox_tree.data() + 2*_gdim*node;
    }

    /// Get bounding box for given node in tree (const version)
    ///
    /// *Arguments*
    ///     node (unsigned int)
    ///         The number of the node (breadth-first numbering, see above).
    inline const double* get_bbox(unsigned int node) const
    {
      dolfin_assert(2*_gdim*(node + 1) <= bbox_tree.size());
      return bbox_tree.data() + 2*_gdim*node;
    }

  private:

    // Build bounding box tree of mesh
    void build(const Mesh& mesh, unsigned int dimension);

    // Build bounding box tree of list of bounding boxes (recursive)
    void build(std::vector<double> bbox_tree,
               std::vector<unsigned int> leaf_partition,
               const std::vector<double> leaf_bboxes,
               unsigned int begin,
               unsigned int end,
               unsigned int position);

    // Compute bounding box of mesh entity
    void compute_bbox(double* bbox,
                      const MeshEntity& entity) const;

    // Compute bounding box of list of bounding boxes. Only the boxes
    // indexed by the given partition list in the range [begin, end)
    // are considered.
    void compute_bbox(double* bbox,
                      const std::vector<double> bboxes,
                      const std::vector<unsigned int> partition,
                      unsigned int begin,
                      unsigned int end) const;

    // Compute longest axis of bounding boxes
    unsigned int compute_longest_axis(const double* bbox) const;

    // Geometric dimension
    unsigned int _gdim;

    // The tree of bounding boxes, stored as a binary tree where the
    // two children of node i are numbered 2i + 1 and 2i + 2. Each
    // node takes up 2*_gdim doubles in the list. These numbers are
    // first the _gdim minimum coordinate values and then the _gdim
    // corresponding maximum coordinate values for each axis.
    std::vector<double> bbox_tree;

  };

}

#endif
