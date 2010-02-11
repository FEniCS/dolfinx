// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andre Massing, 2009.
//
// First added:  2006-05-11
// Last changed: 2010-02-11

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <iostream>

#ifdef HAS_CGAL
#include <CGAL/Bbox_3.h>
#endif

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include "Point.h"
#include "Mesh.h"

namespace dolfin
{


  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some mesh.
  class MeshEntity
  {
  public:

    /// Default Constructor
    MeshEntity() :_mesh(0), _dim(0), _index(0) {}

    /// Constructor
    MeshEntity(const Mesh& mesh, uint dim, uint index);

    /// Destructor
    virtual ~MeshEntity();

    ///Comparision Operator
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

    /// Check if given entity is indicent
    bool incident(const MeshEntity& entity) const;

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

    // The mesh
    Mesh const * _mesh;

    // Topological dimension
    uint _dim;

    // Index of entity within topological dimension
    uint _index;

  };

}

#endif
