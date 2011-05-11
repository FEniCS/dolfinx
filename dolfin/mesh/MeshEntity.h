// Copyright (C) 2006-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Andre Massing, 2009.
//
// First added:  2006-05-11
// Last changed: 2010-11-17

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <iostream>

#ifdef HAS_CGAL
#include <CGAL/Bbox_3.h>
#endif

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "MeshTopology.h"
#include "Point.h"
#include "PrimitiveIntersector.h"

namespace dolfin
{

  //class Mesh;

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some mesh.

  class MeshEntity
  {
  public:

    /// Default Constructor
    MeshEntity() : _mesh(0), _dim(0), _index(0) {}

    /// Constructor
    MeshEntity(const Mesh& mesh, uint dim, uint index);

    /// Destructor
    virtual ~MeshEntity();

    /// Comparision Operator
    bool operator==(const MeshEntity& another) const
    { return (_mesh == another._mesh && _dim == another._dim && _index == another._index); }

    bool operator!=(const MeshEntity& another) const
    { return !operator==(another); }

    /// Return mesh associated with mesh entity
    const Mesh& mesh() const
    { return *_mesh; }

    /// Return topological dimension
    uint dim() const
    { return _dim; }

    /// Return index of mesh entity
    uint index() const
    { return _index; }

    /// Return number of incident mesh entities of given topological dimension
    uint num_entities(uint dim) const
    { return _mesh->topology()(_dim, dim).size(_index); }

    /// Return array of indices for incident mesh entitites of given topological dimension
    const uint* entities(uint dim) const
    { return _mesh->topology()(_dim, dim)(_index); }

    /// Return unique mesh ID
    uint mesh_id() const
    { return _mesh->id(); }

    /// Check if given entity is indicent
    bool incident(const MeshEntity& entity) const;

    /// Check if given point intersects (using inexact but fast numerics)
    bool intersects(const Point& point) const
    { return PrimitiveIntersector::do_intersect(*this,point); }

    /// Check if given entity intersects (using inexact but fast numerics)
    bool intersects(const MeshEntity& entity) const
    { return PrimitiveIntersector::do_intersect(*this,entity); }

    /// Check if given point intersects (using exact numerics)
    bool intersects_exactly(const Point& point) const
    { return PrimitiveIntersector::do_intersect_exact(*this,point); }

    /// Check if given entity intersects (using exact numerics)
    bool intersects_exactly(const MeshEntity& entity) const
    { return PrimitiveIntersector::do_intersect_exact(*this,entity); }

    /// Compute local index of given incident entity (error if not found)
    uint index(const MeshEntity& entity) const;

    /// Compute midpoint of cell
    Point midpoint() const;

    #ifdef HAS_CGAL
    ///Returns a 3D bounding box of the mesh entity. For lower dimension it may be a degenerated box.
    template <typename K> CGAL::Bbox_3 bbox() const;
    #endif

    // Note: Not a subclass of Variable for efficiency!
    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  protected:

    // Friends
    friend class MeshEntityIterator;
    friend class SubsetIterator;

    // The mesh
    Mesh const * _mesh;

    // Topological dimension
    uint _dim;

    // Index of entity within topological dimension
    uint _index;

  };

}

#endif
