// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-09
// Last changed: 2007-05-02

#ifndef __MESH_ENTITY_ITERATOR_H
#define __MESH_ENTITY_ITERATOR_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshEntity.h>

namespace dolfin
{

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
    MeshEntityIterator(Mesh& mesh, uint dim) 
      : entity(mesh, dim, 0), _pos(0), pos_end(mesh.size(dim)), index(0)
    {
      // Compute entities if empty
      if ( pos_end == 0 )
        pos_end = mesh.init(dim);
    }

    /// Create iterator for entities of given dimension connected to given entity    
    MeshEntityIterator(MeshEntity& entity, uint dim)
      : entity(entity.mesh(), dim, 0), _pos(0)
    {
      // Get connectivity
      MeshConnectivity& c = entity.mesh().topology()(entity.dim(), dim);
      
      // Compute connectivity if empty
      if ( c.size() == 0 )
        entity.mesh().init(entity.dim(), dim);
      
      // Get size and index map
      if ( c.size() == 0 )
      {
        pos_end = 0;
        index = 0;
      }
      else
      {
        pos_end = c.size(entity.index());
        index = c(entity.index());
      }
    }

    /// Destructor
    virtual ~MeshEntityIterator() {}
    
    /// Step to next mesh entity (prefix increment)
    MeshEntityIterator& operator++() { ++_pos; return *this; }

    /// Return current position
    inline uint pos() const { return _pos; }

    /// Check if iterator has reached the end
    inline bool end() const { return _pos >= pos_end; }

    /// Dereference operator
    inline MeshEntity& operator*() { return *operator->(); }

    /// Member access operator
    inline MeshEntity* operator->() { entity._index = (index ? index[_pos] : _pos); return &entity; }

    /// Output
    friend LogStream& operator<< (LogStream& stream, const MeshEntityIterator& it);
    
  private:

    /// Copy constructor is private to disallow usage. If it were public (or not
    /// declared and thus a default version available) it would allow code like
    ///
    /// for (CellIterator c0(mesh); !c0.end(); ++c0)
    ///   for (CellIterator c1(c0); !c1.end(); ++c1)
    ///      ...
    ///
    /// c1 looks to be an iterator over the entities around c0 when it is in
    /// fact a copy of c0.
    MeshEntityIterator(MeshEntityIterator& entity) :  entity(entity.entity.mesh(), 0, 0), _pos(0)
    { error("Illegal use of mesh entity iterator."); }
    
    // Mesh entity
    MeshEntity entity;

    // Current position
    uint _pos;

    // End position
    uint pos_end;

    // Mapping from pos to index (if any)
    uint* index;
    
  };

}

#endif
