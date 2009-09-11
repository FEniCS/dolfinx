// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-11
// Last changed: 2009-09-08

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"

namespace dolfin
{

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some mesh.

  class MeshEntity
  {
  public:

    /// Constructor
    MeshEntity(const Mesh& mesh, uint dim, uint index);

    /// Destructor
    virtual ~MeshEntity();

    /// Return mesh associated with mesh entity
    inline const Mesh& mesh() const { return _mesh; }

    /// Return topological dimension
    inline uint dim() const { return _dim; }

    /// Return index of mesh entity
    inline uint index() const { return _index; }

    /// Return number of incident mesh entities of given topological dimension
    inline uint num_entities(uint dim) const { return _mesh.topology()(_dim, dim).size(_index); }

    /// Return array of indices for incident mesh entitites of given topological dimension
    inline const uint* entities(uint dim) const { return _mesh.topology()(_dim, dim)(_index); }

    /// Check if given entity is indicent
    bool incident(const MeshEntity& entity) const;

    /// Compute local index of given incident entity (error if not found)
    uint index(const MeshEntity& entity) const;

    // Note: Not a subclass of Variable for efficiency!

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  protected:

    // Friends
    friend class MeshEntityIterator;

    // The mesh
    const Mesh& _mesh;

    // Topological dimension
    uint _dim;

    // Index of entity within topological dimension
    uint _index;

  };

}

#endif
