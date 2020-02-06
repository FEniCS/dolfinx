// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <boost/functional/hash.hpp>
#include <cassert>
#include <numeric>
#include <sstream>
#include <vector>

namespace dolfinx
{
namespace mesh
{

/// This class provides an a adjacency representation of graphs, and is
/// typically used to store mesh connectivity. For each node in the list
/// of nodes [0, 1, 2, ..., n) it stores the connected nodes. It
/// represents a directed graph. The representation is strictly local,
/// i.e. it is not parallel aware.

template <typename T>
class AdjacencyGraph
{
public:
  /// Initialize with all edges and pointer to each entity
  /// node
  /// @param [in] connections TODO
  /// @param [in] positions TODO
  AdjacencyGraph(const std::vector<T>& connections,
                 const std::vector<std::int32_t>& positions)
      : _array(connections.size()), _offsets(positions.size())
  {
    assert(positions.back() == (std::int32_t)connections.size());
    for (std::size_t i = 0; i < connections.size(); ++i)
      _array[i] = connections[i];
    for (std::size_t i = 0; i < positions.size(); ++i)
      _offsets[i] = positions[i];
  }

  /// Initialize with all edges for case where each node has the
  /// same number of outgoing edges
  /// @param [in] connections TODO
  AdjacencyGraph(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>& connections)
      : _array(connections.rows() * connections.cols()),
        _offsets(connections.rows() + 1)
  {
    // NOTE: cannot directly copy data from connections because it may be
    // a view into a larger array, e.g. for non-affine cells
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < connections.rows(); ++i)
      for (Eigen::Index j = 0; j < connections.cols(); ++j)
        _array[k++] = connections(i, j);

    const std::int32_t num_connections_per_entity = connections.cols();
    for (Eigen::Index e = 0; e < _offsets.rows(); e++)
      _offsets[e] = e * num_connections_per_entity;
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  /// @param [in] connections TODO
  template <typename X>
  AdjacencyGraph(const std::vector<X>& connections)
      : _offsets(connections.size() + 1)
  {
    // Initialize offsets and compute total size
    std::int32_t size = 0;
    for (std::size_t e = 0; e < connections.size(); e++)
    {
      _offsets[e] = size;
      size += connections[e].size();
    }
    _offsets[connections.size()] = size;

    std::vector<T> c;
    c.reserve(size);
    for (auto e = connections.begin(); e != connections.end(); ++e)
      c.insert(c.end(), e->begin(), e->end());

    _array = Eigen::Array<T, Eigen::Dynamic, 1>(c.size());
    std::copy(c.begin(), c.end(), _array.data());
  }

  /// Copy constructor
  AdjacencyGraph(const AdjacencyGraph& connectivity) = default;

  /// Move constructor
  AdjacencyGraph(AdjacencyGraph&& connectivity) = default;

  /// Destructor
  ~AdjacencyGraph() = default;

  /// Assignment
  AdjacencyGraph& operator=(const AdjacencyGraph& connectivity) = default;

  /// Move assignment
  AdjacencyGraph& operator=(AdjacencyGraph&& connectivity) = default;

  /// Number of nodes
  /// @return The number of nodes
  std::int32_t num_nodes() const { return _offsets.rows() - 1; }

  /// Number of connections for given node
  /// @param [in] Node index
  /// @return The number of outgoing edges from the node
  int num_edges(int node) const
  {
    return (node + 1) < _offsets.size() ? _offsets[node + 1] - _offsets[node]
                                        : 0;
  }

  /// @todo Can this be removed?
  /// Return global number of connections for given entity
  std::int64_t size_global(std::int32_t entity) const
  {
    if (_num_global_connections.size() == 0)
      return this->num_edges(entity);
    else
      return _num_global_connections[entity];
  }

  /// Edges for given node
  /// @param [in] node Node index
  /// @return Array of outgoing edges for the node. The length will be
  ///   AdjacencyGraph:num_edges(node).
  std::int32_t* edges(int node)
  {
    return (node + 1) < _offsets.size() ? &_array[_offsets[node]] : nullptr;
  }

  /// Edges for given node (const version)
  /// @param [in] node Node index
  /// @return Array of outgoing edges for the node. The length will be
  ///   AdjacencyGraph:num_edges(node).
  const std::int32_t* edges(int node) const
  {
    return (node + 1) < _offsets.size() ? &_array[_offsets[node]] : nullptr;
  }

  /// Return contiguous array of edges for all nodes
  Eigen::Array<T, Eigen::Dynamic, 1>& array() { return _array; }

  /// Return contiguous array of edges for all nodes (const version)
  const Eigen::Array<T, Eigen::Dynamic, 1>& array() const { return _array; }

  /// Offset for each node in array()
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets() { return _offsets; }

  /// Offset for each node in array() (const version)
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets() const
  {
    return _offsets;
  }

  /// @todo Move this outside of this class
  /// Set global number of connections for each local entities
  void set_global_size(const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
                           num_global_connections)
  {
    assert(num_global_connections.size() == _offsets.size() - 1);
    _num_global_connections = num_global_connections;
  }

  /// Hash of graph
  std::size_t hash() const
  {
    return boost::hash_range(_array.data(), _array.data() + _array.size());
  }

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const
  {
    std::stringstream s;
    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      for (Eigen::Index e = 0; e < _offsets.size() - 1; e++)
      {
        s << "  " << e << ":";
        for (std::int32_t i = _offsets[e]; i < _offsets[e + 1]; i++)
          s << " " << _array[i];
        s << std::endl;
      }
    }
    else
      s << "<Adjacency graph with " << this->num_nodes() << "  nodes>";

    return s.str();
  }

private:
  // Connections for all entities stored as a contiguous array
  Eigen::Array<T, Eigen::Dynamic, 1> _array;

  // Position of first connection for each entity (using local index)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _offsets;

  // Global number of connections for each entity (possibly not
  // computed)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _num_global_connections;
};
} // namespace mesh
} // namespace dolfinx
