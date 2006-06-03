// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-06-31

#ifndef __MESH_ENTITY_ITERATOR_H
#define __MESH_ENTITY_ITERATOR_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshEntity.h>

namespace dolfin
{

  // FIXME: Consistent use of incidence relations, connectivity, connections

  /// MeshEntityIterator provides a common iterator for mesh entities
  /// over meshes, boundaries and incidence relations. The basic use
  /// is illustrated below.
  ///
  /// The following example shows how to iterate over all mesh entities
  /// of a mesh of topological dimension dim:
  ///
  ///     for (MeshEntityIterator e(mesh, dim); !e.end(); ++e)
  ///     {
  ///       e->foo();
  ///     }
  ///
  /// The following example shows how to iterate over mesh entities of
  /// topological dimension dim connected (incident) to some mesh entity f:
  ///
  ///     for (MeshEntityIterator e(f, dim); !e.end(); ++e)
  ///     {
  ///       e->foo();
  ///     }
  ///
  /// In addition to the general iterator, a set of specific named iterators
  /// are provided for entities of type Vertex, Edge, Face, Facet and Cell.
  /// These iterators are defined along with their respective classes.

  class MeshEntityIterator
  {
  public:

    /// Create iterator for mesh entities over given topological dimension
    MeshEntityIterator(NewMesh& mesh, uint dim);
    
    /// Create iterator for entities of given dimension connected to given entity
    MeshEntityIterator(MeshEntity& entity, uint dim);

    /// Create iterator for entities of given dimension connected to given entity
    MeshEntityIterator(MeshEntityIterator& it, uint dim);
    
    /// Destructor
    ~MeshEntityIterator();
    
    // FIXME: No bounds check, should be ok? Check what STL does.

    /// Step to next mesh entity
    MeshEntityIterator& operator++() { ++pos; return *this; }

    /// Check if iterator has reached the end
    inline bool end() const { return pos >= pos_end; }
    
    /// Dereference operator
    inline MeshEntity& operator*() { entity._index = (index ? index[pos] : pos); return entity; }

    /// Member access operator
    inline MeshEntity* operator->() { return &entity; }

    /// Member access operator
    inline const MeshEntity* operator->() const { return &entity; }

    /// Output
    friend LogStream& operator<< (LogStream& stream, const MeshEntityIterator& it);
    
  private:

    // Mesh entity
    MeshEntity entity;

    // Current position
    uint pos;

    // End position
    uint pos_end;

    // Mapping from pos to index (if any)
    uint* index;
    
  };

}

#endif
