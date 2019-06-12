// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <numeric>
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

class Connectivity
{
public:
  /// Initialize with all connections and pointer to each entity
  /// position
  Connectivity(const std::vector<std::int32_t>& connections,
               const std::vector<std::int32_t>& positions);

  /// Initialize with all connections for case where each entity has the
  /// same number of connections
  Connectivity(
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>
          connections);

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  template <typename T>
  Connectivity(const std::vector<T>& connections)
      : _index_to_position(connections.size() + 1)
  {
    // Initialize offsets and compute total size
    std::int32_t size = 0;
    for (std::size_t e = 0; e < connections.size(); e++)
    {
      _index_to_position[e] = size;
      size += connections[e].size();
    }
    _index_to_position[connections.size()] = size;

    std::vector<std::int32_t> c;
    c.reserve(size);
    for (auto e = connections.begin(); e != connections.end(); ++e)
      c.insert(c.end(), e->begin(), e->end());

    _connections = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>(c.size());
    std::copy(c.begin(), c.end(), _connections.data());
  }

  /// Copy constructor
  Connectivity(const Connectivity& connectivity) = default;

  /// Move constructor
  Connectivity(Connectivity&& connectivity) = default;

  /// Destructor
  ~Connectivity() = default;

  /// Assignment
  Connectivity& operator=(const Connectivity& connectivity) = default;

  /// Move assignment
  Connectivity& operator=(Connectivity&& connectivity) = default;

  /// Return number of connections for given entity
  std::size_t size(std::int32_t entity) const;

  /// Return global number of connections for given entity
  std::size_t size_global(std::int32_t entity) const;

  /// Return array of connections for given entity
  std::int32_t* connections(int entity);

  /// Return array of connections for given entity (const version)
  const std::int32_t* connections(int entity) const;

  /// Return contiguous array of connections for all entities
  Eigen::Ref<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> connections();

  /// Return contiguous array of connections for all entities (const
  /// version)
  Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  connections() const;

  /// Position of first connection in connections() for each entity
  /// (using local index)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& entity_positions();

  /// Position of first connection in connections() for each entity
  /// (using local index) (const version)
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& entity_positions() const;

  /// Set global number of connections for each local entities
  void set_global_size(const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
                           num_global_connections);

  /// Hash of connections
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Connections for all entities stored as a contiguous array
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _connections;

  // Position of first connection for each entity (using local index)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _index_to_position;

  // Global number of connections for each entity (possibly not
  // computed)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _num_global_connections;
};
} // namespace mesh
} // namespace dolfin
