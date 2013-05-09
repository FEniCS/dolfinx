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
// Last changed: 2013-05-10

#ifndef __BOUNDING_BOX_TREE_H
#define __BOUNDING_BOX_TREE_H

#include <boost/scoped_ptr.hpp>

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

    /// Build bounding box tree for mesh entites of given dimension.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (unsigned int)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    void build(const Mesh& mesh, unsigned int dimension);

    /// Find entities intersecting the given _Point_.
    ///
    /// Note that the bounding box tree only computes a list of
    /// possible candidates since the bounding box of an object may
    /// intersect even if the object itself does not.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities that might possibly
    ///         intersect with the given object (if any).
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    std::vector<unsigned int> find(const Point& point) const;

    // FIXME:
    //
    // [ ] Store mesh as shared pointer
    // [ ] Access primitives directly from here, needed for closest point
    // [ ] Rename and change functions:
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

    // Dimension-dependent implementation
    boost::scoped_ptr<GenericBoundingBoxTree> _tree;

  };

}

#endif
