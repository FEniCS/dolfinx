// Copyright (C) 2012 Garth N. Wells
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
// Modified by Andre Massing 2009
//
// First added:  2006-05-09
// Last changed: 2014-07-02

#ifndef __MESH_ENTITY_ITERATOR_BASE_H
#define __MESH_ENTITY_ITERATOR_BASE_H

#include <cstddef>
#include "Mesh.h"
#include "MeshEntity.h"

namespace dolfin
{

  /// Base class for MeshEntityIterators

  template<class T>
  class MeshEntityIteratorBase
  {
  public:

    /// Create iterator for mesh entities over given topological dimension
    explicit MeshEntityIteratorBase(const Mesh& mesh)
      : _entity(mesh, 0), _pos(0), pos_end(0), index(0)
    {
      // Check if mesh is empty
      if (mesh.num_vertices() == 0)
        return;

      // Get number of entities (excluding ghosts)
      const std::size_t dim = _entity.dim();
      mesh.init(dim);

      // End at ghost cells for normal iterator
      pos_end = mesh.topology().ghost_offset(dim);
    }

    /// Iterator over MeshEntity of dimension dim on mesh, with string option
    /// to iterate over "regular", "ghost" or "all" entities
    MeshEntityIteratorBase(const Mesh& mesh, std::string opt)
      : _entity(mesh, 0), _pos(0), pos_end(0), index(0)
    {
      // Check if mesh is empty
      if (mesh.num_vertices() == 0)
        return;

      const std::size_t dim = _entity.dim();
      mesh.init(dim);

      pos_end = mesh.topology().size(dim);
      if (opt == "regular")
        pos_end = mesh.topology().ghost_offset(dim);
      else if (opt == "ghost")
        _pos = mesh.topology().ghost_offset(dim);
      else if (opt != "all")
        dolfin_error("MeshEntityIterator.h",
                     "initialize MeshEntityIterator",
                     "unknown opt=\"%s\", choose from "
                     "opt=[\"regular\", \"ghost\", \"all\"]", opt.c_str());
    }

    /// Create iterator for entities of given dimension connected to given entity
    explicit MeshEntityIteratorBase(const MeshEntity& entity)
      : _entity(entity.mesh(), 0), _pos(0), index(0)
    {
      // Get connectivity
      const MeshConnectivity& c = entity.mesh().topology()(entity.dim(), _entity.dim());

      // Compute connectivity if empty
      if (c.empty())
        entity.mesh().init(entity.dim(), _entity.dim());

      // Get size and index map
      if (c.empty())
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

    /// Copy constructor
    MeshEntityIteratorBase(const MeshEntityIteratorBase& it)
      : _entity(it._entity), _pos(it._pos), pos_end(it.pos_end), index(it.index) {}

    /// Destructor
    ~MeshEntityIteratorBase() {}

    /// Step to next mesh entity (prefix increment)
    MeshEntityIteratorBase& operator++()
    {
      ++_pos;
      return *this;
    }

    /// Step to the previous mesh entity (prefix decrease)
    MeshEntityIteratorBase& operator--()
    {
      --_pos;
      return *this;
    }

    /// Return current position
    std::size_t pos() const
    { return _pos; }

    /// Comparison operator.
    bool operator==(const MeshEntityIteratorBase & it) const
    {
      // Use const_cast to use operator* inside comparison, which automatically
      // updates the entity index corresponding to pos *before* comparison (since
      // update of entity delays until request for entity)

      return ((const_cast<MeshEntityIteratorBase<T> *>(this))->operator*()
            == (const_cast<MeshEntityIteratorBase<T> *>(&it))->operator*()
            && _pos == it._pos && index == it.index);
    }

    /// Comparison operator
    bool operator!=(const MeshEntityIteratorBase & it) const
    { return !operator==(it); }

    /// Dereference operator
    T& operator*()
    { return *operator->(); }

    /// Member access operator
    T* operator->()
    { _entity._local_index = (index ? index[_pos] : _pos); return &_entity; }

    /// Check if iterator has reached the end
    bool end() const
    { return _pos >= pos_end; }

    /// Provide a safeguard iterator pointing beyond the end of an iteration
    /// process, either iterating over the mesh /or incident entities. Added to
    /// be bit more like STL iterators, since many algorithms rely on a kind of
    /// beyond iterator.
    MeshEntityIteratorBase end_iterator()
    {
      MeshEntityIteratorBase sg(*this);
      sg.set_end();
      return sg;
    }

    // Note: Not a subclass of Variable for efficiency!
    // Commented out to avoid warning about shadowing str() for MeshEntity
    // Return informal string representation (pretty-print)
    // std::string str(bool verbose) const;

  private:

    /// Set pos to end position. To create a kind of mesh.end() iterator.
    void set_end()
    { _pos = pos_end; }

    // Mesh entity
    T _entity;

    // Current position
    std::size_t _pos;

    // End position
    std::size_t pos_end;

    // Mapping from pos to index (if any)
    const unsigned int* index;

  };

}

#endif
