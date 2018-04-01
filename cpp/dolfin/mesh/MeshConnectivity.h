// Copyright (C) 2006-2007 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/log/log.h>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// Mesh connectivity stores a sparse data structure of connections
/// (incidence relations) between mesh entities for a fixed pair of
/// topological dimensions.
///
/// The connectivity can be specified either by first giving the
/// number of entities and the number of connections for each entity,
/// which may either be equal for all entities or different, or by
/// giving the entire (sparse) connectivity pattern.

class MeshConnectivity
{
public:
  /// Create empty connectivity between given dimensions (d0 -- d1)
  MeshConnectivity(std::size_t d0, std::size_t d1);

  /// Copy constructor
  MeshConnectivity(const MeshConnectivity& connectivity) = default;

  /// Move constructor
  MeshConnectivity(MeshConnectivity&& connectivity) = default;

  /// Destructor
  ~MeshConnectivity() = default;

  /// Assignment
  MeshConnectivity& operator=(const MeshConnectivity& connectivity) = default;

  /// Move assignment
  MeshConnectivity& operator=(MeshConnectivity&& connectivity) = default;

  /// Return true if the total number of connections is equal to zero
  bool empty() const { return _connections.empty(); }

  /// Return total number of connections
  inline std::size_t size() const { return _connections.size(); }

  /// Return number of connections for given entity
  std::size_t size(std::size_t entity) const
  {
    return (entity + 1) < _index_to_position.size()
               ? _index_to_position[entity + 1] - _index_to_position[entity]
               : 0;
  }

  /// Return global number of connections for given entity
  std::size_t size_global(std::size_t entity) const
  {
    if (_num_global_connections.empty())
      return size(entity);
    else
    {
      dolfin_assert(entity < _num_global_connections.size());
      return _num_global_connections[entity];
    }
  }

  /// Return array of connections for given entity
  const std::int32_t* operator()(std::size_t entity) const
  {
    return (entity + 1) < _index_to_position.size()
               ? &_connections[_index_to_position[entity]]
               : nullptr;
  }
  // Eigen::Ref<const EigenArrayXi32> operator()(std::size_t entity) const
  // {
  //   if ((entity + 1) < _index_to_position.size())
  //   {
  //     const std::size_t size
  //         = _index_to_position[entity + 1] - _index_to_position[entity];
  //     return Eigen::Map<const EigenArrayXi32>(
  //         &_connections[_index_to_position[entity]], size);
  //   }
  //   else
  //     return Eigen::Map<const EigenArrayXi32>(nullptr, 0);
  // }

  /// Return contiguous array of connections for all entities
  const std::vector<std::int32_t>& connections() const { return _connections; }

  /// Clear all data
  void clear();

  /// Initialize number of entities and number of connections (equal
  /// for all)
  void init(std::size_t num_entities, std::size_t num_connections);

  /// Initialize number of entities and number of connections
  /// (individually)
  void init(std::vector<std::size_t>& num_connections);

  /// Set given connection for given entity
  void set(std::size_t entity, std::size_t connection, std::size_t pos);

  /// Set all connections for given entity. T is a contains,
  /// e.g. std::vector<std::size_t>
  template <typename T>
  void set(std::size_t entity, const T& connections)
  {
    dolfin_assert((entity + 1) < _index_to_position.size());
    dolfin_assert(connections.size()
                  == _index_to_position[entity + 1]
                         - _index_to_position[entity]);

    // Copy data
    std::copy(connections.begin(), connections.end(),
              _connections.begin() + _index_to_position[entity]);
  }

  /// Set all connections for given entity
  template <typename T>
  void set(std::size_t entity, T* connections)
  {
    dolfin_assert((entity + 1) < _index_to_position.size());
    dolfin_assert(connections);

    // Copy data
    const std::size_t num_connections
        = _index_to_position[entity + 1] - _index_to_position[entity];
    std::copy(connections, connections + num_connections,
              _connections.begin() + _index_to_position[entity]);
  }

  /// Set all connections for all entities (T is a '2D' container, e.g. a
  /// std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  template <typename T>
  void set(const T& connections)
  {
    // Clear old data if any
    clear();

    // Initialize offsets and compute total size
    _index_to_position.resize(connections.size() + 1);
    std::int32_t size = 0;
    for (std::size_t e = 0; e < connections.size(); e++)
    {
      _index_to_position[e] = size;
      size += connections[e].size();
    }
    _index_to_position[connections.size()] = size;

    // Initialize connections
    _connections.reserve(size);
    for (auto e = connections.begin(); e != connections.end(); ++e)
      _connections.insert(_connections.end(), e->begin(), e->end());

    _connections.shrink_to_fit();
  }

  /// Set global number of connections for all local entities
  void set_global_size(const std::vector<std::uint32_t>& num_global_connections)
  {
    dolfin_assert(num_global_connections.size()
                  == _index_to_position.size() - 1);
    _num_global_connections = num_global_connections;
  }

  /// Hash of connections
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Dimensions (only used for pretty-printing)
  std::size_t _d0, _d1;

  // Connections for all entities stored as a contiguous array
  std::vector<std::int32_t> _connections;

  // Global number of connections for all entities (possibly not
  // computed)
  std::vector<std::uint32_t> _num_global_connections;

  // Position of first connection for each entity (using local index)
  std::vector<std::uint32_t> _index_to_position;
};
}
}