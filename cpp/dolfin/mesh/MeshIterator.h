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

namespace mesh
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

  /// Move constructor
  MeshIterator(MeshIterator&& it) = default;

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
class MeshEntityIterator : public std::iterator<std::forward_iterator_tag, T>
{
public:
  /// Constructor from MeshEntity and dimension
  MeshEntityIterator(const MeshEntity& e, std::size_t dim, std::size_t pos)
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

  /// Constructor from MeshEntity
  MeshEntityIterator(const MeshEntity& e, std::size_t pos)
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
  MeshEntityIterator(const MeshEntityIterator& it) = default;

  /// Move constructor
  MeshEntityIterator(MeshEntityIterator&& it) = default;

  /// Copy assignment
  const MeshEntityIterator& operator=(const MeshEntityIterator& m)
  {
    _entity = m._entity;
    _connections = m._connections;
    return *this;
  }

  /// Increment iterator
  MeshEntityIterator& operator++()
  {
    ++_connections;
    return *this;
  }

  /// Equality operator
  bool operator==(const MeshEntityIterator& other) const
  {
    return _connections == other._connections;
  }

  /// Inequality operator
  bool operator!=(const MeshEntityIterator& other) const
  {
    return _connections != other._connections;
  }

  /// Dereference operator
  T* operator->()
  {
    _entity._local_index = *_connections;
    return &_entity;
  }

  /// Indirection operator
  T& operator*()
  {
    _entity._local_index = *_connections;
    return _entity;
  }

  template <typename X>
  friend class EntityRange;

private:
  // MeshEntity
  T _entity;

  // Pointer to current entity index
  const std::int32_t* _connections;
};

/// Range type
enum class MeshRangeType
{
  REGULAR,
  ALL,
  GHOST
};

/// Representation of a collection of entities of type T
/// over a mesh. Provides begin() and end() methods for
/// iterating over entities of the Mesh
template <class T>
class MeshRange
{
public:
  /// Constructor
  MeshRange(const Mesh& mesh, MeshRangeType type = MeshRangeType::REGULAR)
      : _mesh(mesh), _type(type)
  {
  }

  /// MeshIterator of type T pointing to start of range (const)
  const MeshIterator<T> begin() const
  {
    if (_type == MeshRangeType::GHOST)
    {
      auto it = MeshIterator<T>(_mesh, 0);
      it->_local_index = _mesh.topology().ghost_offset(it->_dim);
      return it;
    }

    return MeshIterator<T>(_mesh, 0);
  }

  /// MeshIterator of type T pointing to start of range (non-const)
  MeshIterator<T> begin()
  {
    if (_type == MeshRangeType::GHOST)
    {
      auto it = MeshIterator<T>(_mesh, 0);
      it->_local_index = _mesh.topology().ghost_offset(it->_dim);
      return it;
    }

    return MeshIterator<T>(_mesh, 0);
  }

  /// MeshIterator of type T pointing to end of range (const)
  const MeshIterator<T> end() const
  {
    auto it = MeshIterator<T>(_mesh, 0);
    if (_type == MeshRangeType::REGULAR)
      it->_local_index = _mesh.topology().ghost_offset(it->dim());
    else
      it->_local_index = _mesh.topology().size(it->dim());

    return it;
  }

private:
  // Mesh being iterated over
  const Mesh& _mesh;

  MeshRangeType _type;
};

/// Representation of a collection of entities of given dimension
/// over a mesh. Provides begin() and end() methods for
/// iterating over entities of the Mesh
template <>
class MeshRange<MeshEntity>
{
public:
  /// Constructor
  MeshRange(const Mesh& mesh, int dim,
            MeshRangeType type = MeshRangeType::REGULAR)
      : _mesh(mesh), _dim(dim), _type(type)
  {
  }

  /// MeshIterator of MeshEntity pointing to start of range (const)
  const MeshIterator<MeshEntity> begin() const
  {
    if (_type == MeshRangeType::GHOST)
      return MeshIterator<MeshEntity>(_mesh, _dim,
                                      _mesh.topology().ghost_offset(_dim));

    return MeshIterator<MeshEntity>(_mesh, _dim, 0);
  }

  /// MeshIterator of MeshEntity pointing to start of range (non-const)
  MeshIterator<MeshEntity> begin()
  {
    if (_type == MeshRangeType::GHOST)
      return MeshIterator<MeshEntity>(_mesh, _dim,
                                      _mesh.topology().ghost_offset(_dim));

    return MeshIterator<MeshEntity>(_mesh, _dim, 0);
  }

  /// MeshIterator of MeshEntity pointing to end of range (const)
  const MeshIterator<MeshEntity> end() const
  {
    if (_type == MeshRangeType::REGULAR)
      return MeshIterator<MeshEntity>(_mesh, _dim,
                                      _mesh.topology().ghost_offset(_dim));

    return MeshIterator<MeshEntity>(_mesh, _dim, _mesh.topology().size(_dim));
  }

private:
  // Mesh being iterated over
  const Mesh& _mesh;

  // Dimension of MeshEntities
  const int _dim;

  MeshRangeType _type;
};

// FIXME: Add method 'entities MeshEntity::items(std::size_t dim);'

/// Range of incident entities (of type T) over a MeshEntity
template <class T>
class EntityRange
{
public:
  /// Constructor
  EntityRange(const MeshEntity& e) : _entity(e) {}

  /// MeshEntityIterator of type T pointing to start of range (const)
  const MeshEntityIterator<T> begin() const
  {
    return MeshEntityIterator<T>(_entity, 0);
  }

  /// MeshEntityIterator of type T pointing to start of range (non-const)
  MeshEntityIterator<T> begin() { return MeshEntityIterator<T>(_entity, 0); }

  /// MeshEntityIterator of type T pointing to end of range (const)
  const MeshEntityIterator<T> end() const
  {
    auto it = MeshEntityIterator<T>(_entity, 0);
    std::size_t n
        = (_entity._dim == it->_dim) ? 1 : _entity.num_entities(it->_dim);
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
  /// Constructor
  EntityRange(const MeshEntity& e, int dim) : _entity(e), _dim(dim) {}

  /// MeshEntityIterator of MeshEntity pointing to start of range (const)
  const MeshEntityIterator<MeshEntity> begin() const
  {
    return MeshEntityIterator<MeshEntity>(_entity, _dim, 0);
  }

  /// MeshEntityIterator of MeshEntity pointing to start of range (non-const)
  MeshEntityIterator<MeshEntity> begin()
  {
    return MeshEntityIterator<MeshEntity>(_entity, _dim, 0);
  }

  /// MeshEntityIterator of MeshEntity pointing to end of range (const)
  const MeshEntityIterator<MeshEntity> end() const
  {
    std::size_t n = (_entity._dim == _dim) ? 1 : _entity.num_entities(_dim);
    return MeshEntityIterator<MeshEntity>(_entity, _dim, n);
  }

private:
  // MeshEntity being iterated over
  const MeshEntity& _entity;

  // Dimension of incident entities
  const std::uint32_t _dim;
};
} // namespace mesh
} // namespace dolfin
