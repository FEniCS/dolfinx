// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-11
// Last changed: 2006-10-20

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>

namespace dolfin
{

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some mesh.

  class MeshEntity
  {
  public:

    /// Constructor
    MeshEntity(Mesh& mesh, uint dim, uint index) : _mesh(mesh), _dim(dim), _index(index) {}

    /// Destructor
    ~MeshEntity() {}

    /// Return mesh associated with mesh entity
    inline Mesh& mesh() { return _mesh; }

    /// Return mesh associated with mesh entity
    inline const Mesh& mesh() const { return _mesh; }

    /// Return topological dimension
    inline uint dim() const { return _dim; }

    /// Return index of mesh entity
    inline uint index() const { return _index; }

    /// Return number of incident mesh entities of given topological dimension
    inline uint numEntities(uint dim) const { return _mesh.topology()(_dim, dim).size(_index); }

    /// Return array of indices for incident mesh entitites of given topological dimension
    inline uint* entities(uint dim) { return _mesh.topology()(_dim, dim)(_index); }

    /// Return array of indices for incident mesh entitites of given topological dimension
    inline const uint* entities(uint dim) const { return _mesh.topology()(_dim, dim)(_index); }

    /// Check if given entity is indicent
    inline bool incident(const MeshEntity& entity) const;

    /// Compute local index of given incident entity (error if not found)
    uint index(const MeshEntity& entity) const;

    /// 
    inline real diameter() const
    {
      dolfin_error("diameter() not implemented");
      return 0.0;
    }

    /// Output
    friend LogStream& operator<< (LogStream& stream, const MeshEntity& entity);

  protected:

    // Friends
    friend class MeshEntityIterator;

    // The mesh
    Mesh& _mesh;

    // Topological dimension
    uint _dim;

    // Index of entity within topological dimension
    uint _index;

  };

}

#endif
