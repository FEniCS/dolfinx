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
namespace graph
{

/// This class provides a static adjacency list data structure. It is
/// commonly used to store directed graphs. For each node in the
/// contiguous list of nodes [0, 1, 2, ..., n) it stores the connected
/// nodes. The representation is strictly local, i.e. it is not parallel
/// aware.

template <typename T>
class AdjacencyList
{
public:
  /// Construct trivial adjacency list where each of the n nodes is connected to
  /// itself
  /// @param [in] n Number of nodes
  AdjacencyList(const std::int32_t n) : _array(n), _offsets(n + 1)
  {
    std::iota(_array.data(), _array.data() + n, 0);
    std::iota(_offsets.data(), _offsets.data() + n + 1, 0);
  }

  /// Construct adjacency list from array of data
  /// @param [in] data Adjacency array
  /// @param [in] offsets The index to the adjacency list in the data
  ///   array for node i
  AdjacencyList(const std::vector<T>& data,
                const std::vector<std::int32_t>& offsets)
      : _array(data.size()), _offsets(offsets.size())
  {
    assert(offsets.back() == (std::int32_t)data.size());
    std::copy(data.begin(), data.end(), _array.data());
    std::copy(offsets.begin(), offsets.end(), _offsets.data());
  }

  /// Construct adjacency list for a problem with a fixed number of
  /// links (edges) for each node
  /// @param [in] matrix Two-dimensional array of adjacency data where
  ///   matrix(i, j) is the jth neighbor of the ith node
  AdjacencyList(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>& matrix)
      : _array(matrix.rows() * matrix.cols()), _offsets(matrix.rows() + 1)
  {
    const std::int32_t num_links = matrix.cols();
    for (Eigen::Index e = 0; e < _offsets.rows(); e++)
      _offsets[e] = e * num_links;

    // NOTE: Do not directly copy data from matrix because it may be a
    // view into a larger array
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
      _array.segment(_offsets(i), num_links) = matrix.row(i);
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  /// @param [in] data TODO
  template <typename X>
  AdjacencyList(const std::vector<X>& data) : _offsets(data.size() + 1)
  {
    // Initialize offsets and compute total size
    std::int32_t size = 0;
    for (std::size_t e = 0; e < data.size(); e++)
    {
      _offsets[e] = size;
      size += data[e].size();
    }
    _offsets[data.size()] = size;

    std::vector<T> c;
    c.reserve(size);
    for (auto e = data.begin(); e != data.end(); ++e)
      c.insert(c.end(), e->begin(), e->end());

    _array = Eigen::Array<T, Eigen::Dynamic, 1>(c.size());
    std::copy(c.begin(), c.end(), _array.data());
  }

  /// Copy constructor
  AdjacencyList(const AdjacencyList& connectivity) = default;

  /// Move constructor
  AdjacencyList(AdjacencyList&& connectivity) = default;

  /// Destructor
  ~AdjacencyList() = default;

  /// Assignment
  AdjacencyList& operator=(const AdjacencyList& connectivity) = default;

  /// Move assignment
  AdjacencyList& operator=(AdjacencyList&& connectivity) = default;

  /// Number of nodes
  /// @return The number of nodes
  std::int32_t num_nodes() const { return _offsets.rows() - 1; }

  /// Number of connections for given node
  /// @param [in] node Node index
  /// @return The number of outgoing links (edges) from the node
  int num_links(int node) const
  {
    assert((node + 1) < _offsets.rows());
    return _offsets[node + 1] - _offsets[node];
  }

  /// Links (edges) for given node
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  ///   AdjacencyList:num_links(node).
  auto links(int node)
  {
    return _array.segment(_offsets[node], _offsets[node + 1] - _offsets[node]);
  }

  /// Links (edges) for given node (const version)
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  ///   AdjacencyList:num_links(node).
  auto links(int node) const
  {
    return _array.segment(_offsets[node], _offsets[node + 1] - _offsets[node]);
  }

  /// TODO: attempt to remove
  const std::int32_t* links_ptr(int node) const
  {
    return &_array[_offsets[node]];
  }

  /// Return contiguous array of links for all nodes (const version)
  const Eigen::Array<T, Eigen::Dynamic, 1>& array() const { return _array; }

  /// Offset for each node in array() (const version)
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets() const
  {
    return _offsets;
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
};
} // namespace graph
} // namespace dolfinx
