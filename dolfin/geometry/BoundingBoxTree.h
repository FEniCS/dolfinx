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
// Last changed: 2013-08-28

#ifndef __BOUNDING_BOX_TREE_H
#define __BOUNDING_BOX_TREE_H

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <vector>
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

    /// Create empty bounding box tree
    BoundingBoxTree();

    /// Destructor
    ~BoundingBoxTree();

    /// Build bounding box tree for cells of mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    void build(const Mesh& mesh);

    /// Build bounding box tree for mesh entities of given dimension.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (std::size_t)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    void build(const Mesh& mesh, std::size_t tdim);

    /// Build bounding box tree for point cloud.
    ///
    /// *Arguments*
    ///     points (std::vector<_Point_>)
    ///         The list of points.
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    void build(const std::vector<Point>& points, std::size_t gdim);

    /// Compute all collisions between bounding boxes and _Point_.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities contained in
    ///         (leaf) bounding boxes that collide with (intersect)
    ///         the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    std::vector<unsigned int>
    compute_collisions(const Point& point) const;

    /// Compute all collisions between bounding boxes and _BoundingBoxTree_.
    ///
    /// *Returns* std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    ///     Two lists of local indices for entities contained in
    ///     (leaf) bounding boxes that collide with (intersect) the
    ///     given bounding box tree. The first list contains entity
    ///     indices for the first tree (this tree) and the second
    ///     contains entity indices for the second tree (the input
    ///     argument).
    ///
    /// *Arguments*
    ///     tree (_BoundingBoxTree_)
    ///         The bounding box tree.
    std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    compute_collisions(const BoundingBoxTree& tree) const;

    /// Compute all collisions between entities and _Point_.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities that collide with
    ///         (intersect) the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    std::vector<unsigned int>
    compute_entity_collisions(const Point& point) const;

    /// Compute all collisions between entities and _BoundingBoxTree_.
    ///
    /// *Returns*
    ///     std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    ///         A list of local indices for entities that collide with
    ///         (intersect) the given bounding box tree. The first
    ///         list contains entity indices for the first tree (this
    ///         tree) and the second contains entity indices for the
    ///         second tree (the input argument).
    ///
    /// *Arguments*
    ///     tree (_BoundingBoxTree_)
    ///         The bounding box tree.
    std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    compute_entity_collisions(const BoundingBoxTree& tree) const;

    /// Compute first collision between bounding boxes and _Point_.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the first found entity contained
    ///         in a (leaf) bounding box that collides with
    ///         (intersects) the given point. If not found,
    ///         std::numeric_limits<unsigned int>::max() is returned.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    unsigned int
    compute_first_collision(const Point& point) const;

    /// Compute first collision between entities and _Point_.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the first found entity that
    ///         collides with (intersects) the given point. If not
    ///         found, std::numeric_limits<unsigned int>::max() is
    ///         returned.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    unsigned int
    compute_first_entity_collision(const Point& point) const;

    /// Compute closest entity to _Point_.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the entity that is closest to the
    ///         point. If more than one entity is at the same distance
    ///         (or point contained in entity), then the first entity
    ///         is returned.
    ///     double
    ///         The distance to the closest entity.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    std::pair<unsigned int, double>
    compute_closest_entity(const Point& point) const;

    /// Compute closest point to _Point_. This function assumes
    /// that the tree has been built for a point cloud.
    ///
    /// Developer note: This function should not be confused with
    /// computing the closest point in all entities of a mesh. That
    /// function could be added with relative ease since we actually
    /// compute the closest points to get the distance in the above
    /// function (compute_closest_entity) inside the specialized
    /// implementations in TetrahedronCell.cpp etc.
    ///
    /// *Returns*
    ///     unsigned int
    ///         The local index for the point that is closest to the
    ///         point. If more than one point is at the same distance
    ///         (or point contained in entity), then the first point
    ///         is returned.
    ///     double
    ///         The distance to the closest point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    std::pair<unsigned int, double>
    compute_closest_point(const Point& point) const;

  private:

    // Check that tree has been built
    void check_built() const;

    // Dimension-dependent implementation
    boost::scoped_ptr<GenericBoundingBoxTree> _tree;

    // Pointer to the mesh. We all know that we don't really want
    // to store a pointer to the mesh here, but without it we will
    // be forced to make calls like
    // tree_A.compute_entity_intersections(tree_B, mesh_A, mesh_B).
    const Mesh* _mesh;

  };

}

#endif
