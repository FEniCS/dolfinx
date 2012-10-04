// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Andre Massing, 2009.
// Modified by Garth N. Wells, 2012.
//
// First added:  2006-05-11
// Last changed: 2012-06-12

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <iostream>

#ifdef HAS_CGAL
#include <CGAL/Bbox_3.h>
#endif

#include <dolfin/common/types.h>
#include <dolfin/intersection/PrimitiveIntersector.h>
#include "Mesh.h"
#include "Point.h"

namespace dolfin
{

  //class Mesh;
  class Point;

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some _Mesh_.

  class MeshEntity
  {
  public:

    /// Default Constructor
    MeshEntity() : _mesh(0), _dim(0), _local_index(0) {}

    /// Constructor
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (uint)
    ///         The topological dimension.
    ///     index (uint)
    ///         The index.
    MeshEntity(const Mesh& mesh, uint dim, uint index);

    /// Destructor
    virtual ~MeshEntity();

    /// Initialize mesh entity with given data
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (uint)
    ///         The topological dimension.
    ///     index (uint)
    ///         The index.
    void init(const Mesh& mesh, uint dim, uint index);

    /// Comparision Operator
    ///
    /// *Arguments*
    ///     another (_MeshEntity_)
    ///         Another mesh entity
    ///
    /// *Returns*
    ///     bool
    ///         True if the two mesh entities are equal.
    bool operator==(const MeshEntity& e) const
    { return (_mesh == e._mesh && _dim == e._dim && _local_index == e._local_index); }

    /// Comparision Operator
    ///
    /// *Arguments*
    ///     another (MeshEntity)
    ///         Another mesh entity.
    ///
    /// *Returns*
    ///     bool
    ///         True if the two mesh entities are NOT equal.
    bool operator!=(const MeshEntity& e) const
    { return !operator==(e); }

    /// Return mesh associated with mesh entity
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    const Mesh& mesh() const
    { return *_mesh; }

    /// Return topological dimension
    ///
    /// *Returns*
    ///     uint
    ///         The dimension.
    uint dim() const
    { return _dim; }

    /// Return index of mesh entity
    ///
    /// *Returns*
    ///     uint
    ///         The index.
    uint index() const
    { return _local_index; }

    /// Return global index of mesh entity
    ///
    /// *Returns*
    ///     int
    ///         The global index. Set to -1 if global index has not been
    ///         computed
    uint global_index() const
    { return _mesh->topology().global_indices(_dim)[_local_index]; }

    /// Return number of incident mesh entities of given topological dimension
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         The number of incident MeshEntity objects of given dimension.
    uint num_entities(uint dim) const
    { return _mesh->topology()(_dim, dim).size(_local_index); }

    /// Return array of indices for incident mesh entitites of given
    /// topological dimension
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         The index for incident mesh entities of given dimension.
    const uint* entities(uint dim) const
    { return _mesh->topology()(_dim, dim)(_local_index); }

    /// Return unique mesh ID
    ///
    /// *Returns*
    ///     uint
    ///         The unique mesh ID.
    uint mesh_id() const
    { return _mesh->id(); }

    /// Check if given entity is incident
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The entity.
    ///
    /// *Returns*
    ///     bool
    ///         True if the given entity is incident
    bool incident(const MeshEntity& entity) const;

    /// Check if given point intersects (using inexact but fast
    /// numerics)
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True if the given point intersects.
    bool intersects(const Point& point) const
    { return PrimitiveIntersector::do_intersect(*this, point); }

    /// Check if given entity intersects (using inexact but fast
    /// numerics)
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// *Returns*
    ///     bool
    ///         True if the given entity intersects.
    bool intersects(const MeshEntity& entity) const
    { return PrimitiveIntersector::do_intersect(*this, entity); }

    /// Check if given point intersects (using exact numerics)
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True if the given point intersects.
    bool intersects_exactly(const Point& point) const
    { return PrimitiveIntersector::do_intersect_exact(*this, point); }

    /// Check if given entity intersects (using exact numerics)
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// *Returns*
    ///     bool
    ///         True if the given entity intersects.
    bool intersects_exactly(const MeshEntity& entity) const
    { return PrimitiveIntersector::do_intersect_exact(*this, entity); }

    /// Compute local index of given incident entity (error if not
    /// found)
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// *Returns*
    ///     uint
    ///         The local index of given entity.
    uint index(const MeshEntity& entity) const;

    /// Compute midpoint of cell
    ///
    /// *Returns*
    ///     _Point_
    ///         The midpoint of the cell.
    Point midpoint() const;

    #ifdef HAS_CGAL
    /// Returns a 3D bounding box of the mesh entity. For lower
    /// dimension it may be a degenerated box.
    template <typename K>
    CGAL::Bbox_3 bbox() const;
    #endif

    // Note: Not a subclass of Variable for efficiency!
    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation of the function space.
    std::string str(bool verbose) const;

  protected:

    // Friends
    friend class MeshEntityIterator;
    template<typename T> friend class MeshEntityIteratorBase;
    friend class SubsetIterator;

    // The mesh
    Mesh const * _mesh;

    // Topological dimension
    uint _dim;

    // Local index of entity within topological dimension
    uint _local_index;

  };

}

#endif
