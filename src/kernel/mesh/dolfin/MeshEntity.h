// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-11
// Last changed: 2006-06-22

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/NewMesh.h>

namespace dolfin
{

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some mesh.

  class MeshEntity
  {
  public:

    /// Constructor
    MeshEntity(NewMesh& mesh, uint dim, uint index) : _mesh(mesh), _dim(dim), _index(index) {}

    /// Destructor
    ~MeshEntity() {}

    /// Return mesh associated with mesh entity
    inline NewMesh& mesh() { return _mesh; }

    /// Return mesh associated with mesh entity
    inline const NewMesh& mesh() const { return _mesh; }

    /// Return topological dimension
    inline uint dim() const { return _dim; }

    /// Return index of mesh entity
    inline uint index() const { return _index; }

    /// Return number of connections
    inline uint numConnections(uint dim) const { return _mesh.topology()(_dim, dim).size(_index); }

    /// Return array of connections to entities of given topological dimension
    inline uint* connections(uint dim) { return _mesh.topology()(_dim, dim)(_index); }

    /// Return array of connections to entities of given topological dimension
    inline const uint* connections(uint dim) const { return _mesh.topology()(_dim, dim)(_index); }

    /// Output
    friend LogStream& operator<< (LogStream& stream, const MeshEntity& entity);

  protected:

    // Friends
    friend class MeshEntityIterator;

    // The mesh
    NewMesh& _mesh;

    // Topological dimension
    uint _dim;

    // Index of entity within topological dimension
    uint _index;

  };

}

#endif
