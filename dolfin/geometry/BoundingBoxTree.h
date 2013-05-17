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
// Last changed: 2013-05-17

#ifndef __BOUNDING_BOX_TREE_H
#define __BOUNDING_BOX_TREE_H

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <limits>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class Point;
  class GenericBoundingBoxTree;

  /// This class implements a (distributed) axis aligned bounding box
  /// tree (AABB tree). Bounding box trees can be created from meshes
  /// and [other data structures, to be filled in].

  class BoundingBoxTree
  {
  public:

    /// Create empty bounding box tree for cells of mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh);

    /// Create empty bounding box tree for cells of mesh (shared_ptr version).
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    BoundingBoxTree(boost::shared_ptr<const Mesh> mesh);

    /// Create empty bounding box tree for mesh entities of given
    /// dimension.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (unsigned int)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh, unsigned int dim);

    /// Create empty bounding box tree for mesh entities of given
    /// dimension (shared_ptr version).
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (unsigned int)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    BoundingBoxTree(boost::shared_ptr<const Mesh> mesh, unsigned int dim);

    /// Destructor
    ~BoundingBoxTree();

    /// Build bounding box tree
    void build();

    /// Compute all collisions between bounding boxes and given _Point_.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities contained in
    ///         (leaf) bounding boxes that intersect the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    std::vector<unsigned int>
    compute_collisions(const Point& point) const;

    /// Compute all collisions between entities and given _Point_.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities that intersect the
    ///         given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    std::vector<unsigned int>
    compute_entity_collisions(const Point& point) const;

    /// Compute first collision between bounding boxes and given _Point_.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the first found entity contained in
    ///         a (leaf) bounding box that intersects the given point.
    ///         If not found, std::numeric_limits<unsigned int>::max()
    ///         is returned.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    unsigned int
    compute_first_collision(const Point& point) const;

    /// Compute first collision between entities and given _Point_.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the first found entity that
    ///         intersects the given point. If not found,
    ///         std::numeric_limits<unsigned int>::max() is returned.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    unsigned int
    compute_first_entity_collision(const Point& point) const;

    // FIXME:
    //
    // [x] Store mesh as shared pointer
    // [ ] Access primitives directly from here, needed for closest point
    // [ ] Rename and change functions:
    // [ ] Check use of unsigned int vs size_t
    // [ ] Ignore reference version of constructors in Python interface
    //
    // compute_collisions()
    // Compute all collisions with given _Point_.
    //
    // compute_first_collision()
    // Compute first collision with given _Point_.
    //
    // compute_closest()
    // Compute closest entity to given _Point_.

  private:

    // FIXME: Remove
    friend class MeshPointIntersection;

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // Topological dimension of leaf entities
    unsigned int _tdim;

    // Dimension-dependent implementation
    boost::scoped_ptr<GenericBoundingBoxTree> _tree;

  };

}

#endif
