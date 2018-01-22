// Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include "MeshTopology.h"
#include <iterator>

namespace dolfin
{
// Developer note: This code is performance critical as it appears in
// tight assembly loops. Any changes should be carefully profiled.
//
// Performance is favoured over code re-use in some parts of the
// implementations.

/// Iterator for entities of type T over a Mesh
template <class T>
class MeshIterator : public std::iterator<std::forward_iterator_tag, T>
{
public:
  /// Constructor for entities of dimension d
  MeshIterator(const Mesh& mesh, std::size_t dim, std::size_t pos)
      : _entity(mesh, dim, pos)
  {
  }

  /// Constructor for entities of type T
  MeshIterator(const Mesh& mesh, std::size_t pos) : _entity(mesh, pos) {}

  /// Copy constructor
  MeshIterator(const MeshIterator& it) = default;

  /// Copy assignment
  const MeshIterator& operator=(const MeshIterator& m)
  {
    _entity = m._entity;
    return *this;
  }

  /// Increment iterator
  MeshIterator& operator++()
  {
    ++_entity._local_index;
    return *this;
  }

  /// Return true if equal
  bool operator==(const MeshIterator& other) const
  {
    return _entity._local_index == other._entity._local_index;
  }

  /// Return true if not equal
  bool operator!=(const MeshIterator& other) const
  {
    return _entity._local_index != other._entity._local_index;
  }

  /// Member access
  T* operator->() { return &_entity; }

  /// Dereference
  T& operator*() { return _entity; }

private:
  // MeshEntity
  T _entity;
};

/// Iterator for entities of specified dimension that are incident to a
/// MeshEntity
template <class T>
class MeshEntityIteratorNew : public std::iterator<std::forward_iterator_tag, T>
{
public:
  // Constructor from MeshEntity and dimension
  MeshEntityIteratorNew(const MeshEntity& e, std::size_t dim, std::size_t pos)
      : _entity(e.mesh(), dim, 0), _connections(nullptr)
  {
    // FIXME: Handle case when number of attached entities is zero?

    if (e.dim() == dim)
    {
      dolfin_assert(pos < 2);
      _connections = &e._local_index + pos;
      _entity._local_index = e._local_index;
      return;
    }

    // Get connectivity
    const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity.dim());

    // Pointer to array of connections
    dolfin_assert(!c.empty());
    _connections = c(e.index()) + pos;
    _entity._local_index = *_connections;
  }

  // Constructor from MeshEntity
  MeshEntityIteratorNew(const MeshEntity& e, std::size_t pos)
      : _entity(e.mesh(), 0), _connections(nullptr)
  {
    // FIXME: Handle case when number of attached entities is zero?

    if (e.dim() == _entity.dim())
    {
      dolfin_assert(pos < 2);
      _connections = &e._local_index + pos;
      _entity._local_index = e._local_index;
      return;
    }

    // Get connectivity
    const MeshConnectivity& c = e.mesh().topology()(e.dim(), _entity.dim());

    // Pointer to array of connections
    dolfin_assert(!c.empty());
    _connections = c(e.index()) + pos;
    _entity._local_index = *_connections;
  }

  /// Copy constructor
  MeshEntityIteratorNew(const MeshEntityIteratorNew& it) = default;

  // Copy assignment
  const MeshEntityIteratorNew& operator=(const MeshEntityIteratorNew& m)
  {
    _entity = m._entity;
    _connections = m._connections;
    return *this;
  }

  MeshEntityIteratorNew& operator++()
  {
    ++_connections;
    return *this;
  }

  bool operator==(const MeshEntityIteratorNew& other) const
  {
    return _connections == other._connections;
  }

  bool operator!=(const MeshEntityIteratorNew& other) const
  {
    return _connections != other._connections;
  }

  T* operator->()
  { _entity._local_index = *_connections; return &_entity; }

  T& operator*()
  { _entity._local_index = *_connections; return _entity; }

  template <typename X>
  friend class EntityRange;

private:
  // MeshEntity
  T _entity;

  // Pointer to current entity index
  const std::uint32_t* _connections;
};

template <class T>
class MeshRange
{
public:

  MeshRange(const Mesh& mesh) : _mesh(mesh) {}

  const MeshIterator<T> begin() const { return MeshIterator<T>(_mesh, 0); }

  MeshIterator<T> begin() { return MeshIterator<T>(_mesh, 0); }

  const MeshIterator<T> end() const
  {
    auto it = MeshIterator<T>(_mesh, 0);
    std::size_t end = _mesh.topology().ghost_offset(it->dim());
    it->_local_index = end;
    return it;
  }

private:
  // Mesh being iterated over
  const Mesh& _mesh;
};

// FIXME: handled ghosted meshes
// Class represening a collection of entities of given dimension
/// over a mesh. Provides  with begin() and end() methods for
/// iterating over entities incident to a Mesh
template <>
class MeshRange<MeshEntity>
{
public:
  MeshRange(const Mesh& mesh, int dim) : _mesh(mesh), _dim(dim) {}

  const MeshIterator<MeshEntity> begin() const
  {
    return MeshIterator<MeshEntity>(_mesh, _dim, 0);
  }

  MeshIterator<MeshEntity> begin()
  {
    return MeshIterator<MeshEntity>(_mesh, _dim, 0);
  }

  const MeshIterator<MeshEntity> end() const
  {
    return MeshIterator<MeshEntity>(_mesh, _dim,
                                    _mesh.topology().ghost_offset(_dim));
  }

private:
  // Mesh being iterated over
  const Mesh& _mesh;

  // Dimension of MeshEntities
  const int _dim;
};


// FIXME: Add method 'entities MeshEntity::items(std::size_t dim);'

/// Range of incident entities (of type T) over a MeshEntity
template <class T>
class EntityRange
{
public:
  EntityRange(const MeshEntity& e) : _entity(e) {}

  const MeshEntityIteratorNew<T> begin() const
  {
    return MeshEntityIteratorNew<T>(_entity, 0);
  }

  MeshEntityIteratorNew<T> begin()
  {
    return MeshEntityIteratorNew<T>(_entity, 0);
  }

  const MeshEntityIteratorNew<T> end() const
  {
    auto it = MeshEntityIteratorNew<T>(_entity, 0);
    std::size_t n = (_entity._dim == it->_dim) ?
      1 : _entity.num_entities(it->_dim);
    it._connections += n;
    return it;
  }

private:
  // MeshEntity being iterated over
  const MeshEntity& _entity;
};

/// Class with begin() and end() methods for iterating over
/// entities incident to a MeshEntity
template <>
class EntityRange<MeshEntity>
{
public:
  EntityRange(const MeshEntity& e, int dim) : _entity(e), _dim(dim) {}

  const MeshEntityIteratorNew<MeshEntity> begin() const
  {
    return MeshEntityIteratorNew<MeshEntity>(_entity, _dim, 0);
  }

  MeshEntityIteratorNew<MeshEntity> begin()
  {
    return MeshEntityIteratorNew<MeshEntity>(_entity, _dim, 0);
  }

  const MeshEntityIteratorNew<MeshEntity> end() const
  {
    std::size_t n = (_entity._dim == _dim) ?
      1 : _entity.num_entities(_dim);
    return MeshEntityIteratorNew<MeshEntity>(_entity, _dim, n);
  }

private:
  // MeshEntity being iterated over
  const MeshEntity& _entity;

  // Dimension of incident entities
  const std::uint32_t _dim;
};

}
