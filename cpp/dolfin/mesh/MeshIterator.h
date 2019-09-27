// Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Connectivity.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "Topology.h"
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
class MeshIterator : public std::iterator<std::forward_iterator_tag, MeshEntity>
{
public:
  /// Constructor for entities of dimension d
  MeshIterator(const Mesh& mesh, int dim, std::size_t pos)
      : _entity(mesh, dim, pos)
  {
  }

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
  MeshEntity* operator->() { return &_entity; }

  /// Dereference
  MeshEntity& operator*() { return _entity; }

private:
  // MeshEntity
  MeshEntity _entity;
};

/// Iterator for entities of specified dimension that are incident to a
/// MeshEntity
class MeshEntityIterator
    : public std::iterator<std::forward_iterator_tag, MeshEntity>
{
public:
  /// Constructor from MeshEntity and dimension
  MeshEntityIterator(const MeshEntity& e, int dim, std::size_t pos)
      : _entity(e.mesh(), dim, 0), _connections(nullptr)
  {
    // FIXME: Handle case when number of attached entities is zero?

    if (e.dim() == dim)
    {
      assert(pos < 2);
      _connections = &e._local_index + pos;
      return;
    }

    // Get connectivity
    assert(e.mesh().topology().connectivity(e.dim(), _entity.dim()));
    const Connectivity& c
        = *e.mesh().topology().connectivity(e.dim(), _entity.dim());

    // Pointer to array of connections
    _connections = c.connections(e.index()) + pos;
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
  MeshEntity* operator->()
  {
    _entity._local_index = *_connections;
    return &_entity;
  }

  /// Indirection operator
  MeshEntity& operator*()
  {
    _entity._local_index = *_connections;
    return _entity;
  }

  friend class EntityRange;

private:
  // MeshEntity
  MeshEntity _entity;

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

/// Representation of a collection of entities of given dimension over a
/// mesh. Provides begin() and end() methods for iterating over entities
/// of the Mesh
class MeshRange
{
public:
  /// Constructor
  MeshRange(const Mesh& mesh, int dim,
            MeshRangeType type = MeshRangeType::REGULAR)
      : _mesh(mesh), _dim(dim), _type(type)
  {
  }

  /// MeshIterator of MeshEntity pointing to start of range (const)
  const MeshIterator begin() const
  {
    if (_type == MeshRangeType::GHOST)
      return MeshIterator(_mesh, _dim, _mesh.topology().ghost_offset(_dim));

    return MeshIterator(_mesh, _dim, 0);
  }

  /// MeshIterator of MeshEntity pointing to start of range (non-const)
  MeshIterator begin()
  {
    if (_type == MeshRangeType::GHOST)
      return MeshIterator(_mesh, _dim, _mesh.topology().ghost_offset(_dim));

    return MeshIterator(_mesh, _dim, 0);
  }

  /// MeshIterator of MeshEntity pointing to end of range (const)
  const MeshIterator end() const
  {
    if (_type == MeshRangeType::REGULAR)
      return MeshIterator(_mesh, _dim, _mesh.topology().ghost_offset(_dim));

    return MeshIterator(_mesh, _dim, _mesh.topology().size(_dim));
  }

private:
  // Mesh being iterated over
  const Mesh& _mesh;

  // Dimension of MeshEntities
  const int _dim;

  MeshRangeType _type;
};

// FIXME: Add method 'entities MeshEntity::items(std::size_t dim);'

/// Class with begin() and end() methods for iterating over entities
/// incident to a MeshEntity
class EntityRange
{
public:
  /// Constructor
  EntityRange(const MeshEntity& e, int dim) : _entity(e), _dim(dim) {}

  /// MeshEntityIterator of MeshEntity pointing to start of range
  /// (const)
  const MeshEntityIterator begin() const
  {
    return MeshEntityIterator(_entity, _dim, 0);
  }

  /// MeshEntityIterator of MeshEntity pointing to start of range
  /// (non-const)
  MeshEntityIterator begin() { return MeshEntityIterator(_entity, _dim, 0); }

  /// MeshEntityIterator of MeshEntity pointing to end of range (const)
  const MeshEntityIterator end() const
  {
    const int n = (_entity._dim == _dim) ? 1
                                         : _entity.mesh()
                                               .topology()
                                               .connectivity(_entity._dim, _dim)
                                               ->size(_entity.index());
    return MeshEntityIterator(_entity, _dim, n);
  }

private:
  // MeshEntity being iterated over
  const MeshEntity& _entity;

  // Dimension of incident entities
  const int _dim;
};
} // namespace mesh
} // namespace dolfin
