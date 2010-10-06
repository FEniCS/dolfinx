// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andre Massing, 2009.
//
// First added:  2006-05-09
// Last changed: 2010-03-02

#ifndef __MESH_ENTITY_ITERATOR_H
#define __MESH_ENTITY_ITERATOR_H

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "MeshEntity.h"

namespace dolfin
{

  /// MeshEntityIterator provides a common iterator for mesh entities
  /// over meshes, boundaries and incidence relations. The basic use
  /// is illustrated below.
  ///
  /// *Example*
  ///
  ///     The following example shows how to iterate over all mesh entities
  ///     of a mesh of topological dimension dim:
  ///
  ///     .. code-block:: c++
  ///
  ///         for (MeshEntityIterator e(mesh, dim); !e.end(); ++e)
  ///         {
  ///           e->foo();
  ///         }
  ///
  ///     The following example shows how to iterate over mesh entities of
  ///     topological dimension dim connected (incident) to some mesh entity f:
  ///
  ///     .. code-block:: c++
  ///
  ///         for (MeshEntityIterator e(f, dim); !e.end(); ++e)
  ///         {
  ///           e->foo();
  ///         }
  ///
  /// In addition to the general iterator, a set of specific named iterators
  /// are provided for entities of type _Vertex_, _Edge_, _Face_, _Facet_
  /// and _Cell_. These iterators are defined along with their respective
  /// classes.

  class MeshEntityIterator
  {
  public:

    ///Default constructor
    MeshEntityIterator() : _pos(0), pos_end(0), index(0) {}

    /// Create iterator for mesh entities over given topological dimension
    MeshEntityIterator(const Mesh& mesh, uint dim)
      : entity(mesh, dim, 0), _pos(0), pos_end(mesh.size(dim)), index(0)
    {
      // Compute entities if empty
      if (pos_end == 0)
        pos_end = mesh.init(dim);
    }

    /// Create iterator for entities of given dimension connected to given entity
    MeshEntityIterator(const MeshEntity& entity, uint dim)
      : entity(entity.mesh(), dim, 0), _pos(0), index(0)
    {
      // Get connectivity
      const MeshConnectivity& c = entity.mesh().topology()(entity.dim(), dim);

      // Compute connectivity if empty
      if (c.size() == 0)
        entity.mesh().init(entity.dim(), dim);

      // Get size and index map
      if (c.size() == 0)
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

    /// Copy Constructor
    MeshEntityIterator(const MeshEntityIterator& it) :  entity(it.entity),
        _pos(it._pos), pos_end(it.pos_end), index(it.index) {};

    ///Step to next mesh entity (prefix increment)
    MeshEntityIterator& operator++()
    {
      ++_pos;
      return *this;
    }

    /// Step to the previous mesh entity (prefix decrease)
    MeshEntityIterator& operator--()
    {
      --_pos;
      return *this;
    }

    /// Return current position
    uint pos() const
    { return _pos; }

    ///Comparison operator.
    ///@internal
    ///Uncommenting following  results into the warning message:
    //dolfin/mesh/MeshEntityIterator.h:94: Warning|508| Declaration of 'operator ==' shadows declaration accessible via operator->(),
    //Use const_cast to use operator* inside comparison, which automatically
    //updates the entity index corresponding to pos *before* comparison (since
    //update of entity delays until request for entity)
    bool operator==(const MeshEntityIterator & it) const
    {
      return ((const_cast<MeshEntityIterator *>(this))->operator*()
            == (const_cast<MeshEntityIterator *>(&it))->operator*()
            && _pos == it._pos && index == it.index);
    }

    bool operator!=(const MeshEntityIterator & it) const
    { return !operator==(it); }

    /// Dereference operator
    MeshEntity& operator*()
    { return *operator->(); }

    /// Member access operator
    MeshEntity* operator->()
    { entity._index = (index ? index[_pos] : _pos); return &entity; }

    ///Random access operator.
    MeshEntity& operator[] (uint pos)
    { _pos = pos; return *operator->();}

    /// Check if iterator has reached the end
    bool end() const { return _pos >= pos_end; }

    ///Provide a safeguard iterator pointing beyond the end of an iteration
    ///process, either iterating over the mesh /or incident entities. Added to
    ///be bit more like STL iteratoren, since many algorithms rely on a kind of
    ///beyond iterator.
    MeshEntityIterator end_iterator()
    {
      MeshEntityIterator
      sg(*this);
      sg.set_end();
      return sg;
    }

    // Note: Not a subclass of Variable for efficiency!
    // Commented out to avoid warning about shadowing str() for MeshEntity
    // Return informal string representation (pretty-print)
    // std::string str(bool verbose) const;

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

//    MeshEntityIterator(const MeshEntityIterator& entity) :  entity(entity.entity.mesh(), 0, 0), _pos(0)
//    { error("Illegal use of mesh entity iterator."); }

    /// Set pos to end position. To create a kind of mesh.end() iterator.
    void set_end()
    { _pos = pos_end; }

    // Mesh entity
    MeshEntity entity;

    // Current position
    uint _pos;

    // End position
    uint pos_end;

    // Mapping from pos to index (if any)
    const uint* index;

  };

}

#endif
