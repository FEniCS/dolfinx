// Copyright (C) 2019-2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

namespace dolfinx::graph
{
/// @brief This class provides a static adjacency list data structure.
///
/// It is commonly used to store directed graphs. For each node in the
/// contiguous list of nodes [0, 1, 2, ..., n) it stores the connected
/// nodes. The representation is strictly local, i.e. it is not parallel
/// aware.
///
/// The link (edge) type is template parameter, which allows link data
/// to be stored, e.g. a pair with the target node index and the link
/// weight.
///
/// Node data can also be stored.
///
/// @tparam LinkData_t Graph link (edge) type.
/// @tparam NodeData_t Data type for graph node data.
template <typename U, typename V = std::vector<std::int32_t>,
          typename NodeData = std::nullptr_t>
class AdjacencyList
{
private:
  template <typename X>
  X create_iota(std::size_t n)
  {
    X x(n);
    std::iota(x.begin(), x.end(), 0);
    return x;
  }

public:
  /// @brief Adjacency list link (edge) type
  using link_type = typename std::decay_t<U>::value_type;
  /// @brief Adjacency list node data type
  using node_data_type = NodeData;

  /// @brief Construct adjacency list from arrays of link (edge) data
  /// and offsets.
  /// @param[in] data Adjacency lost data array.
  /// @param[in] offsets Offsets into `data` for each node, where
  /// `offsets[i]` is the first index in `data` for node `i`. The last
  /// index in `offsets` is the equal to the length of `data`. array for
  /// node `i`.
  template <typename W0, typename W1>
  // requires std::is_convertible_v<std::remove_cvref_t<U>,
  //                                std::vector<link_type>>
  //              and std::is_convertible_v<std::remove_cvref_t<V>,
  //                                        std::vector<std::int32_t>>
  AdjacencyList(W0&& data, W1&& offsets)
      : _array(std::forward<W0>(data)), _offsets(std::forward<W1>(offsets))
  {
    // _array.reserve(_offsets.back());
    assert(_offsets.back() == (std::int32_t)_array.size());
  }

  /// @brief Construct adjacency list from arrays of link (edge) data,
  /// offsets, and node data.
  /// @param[in] data Adjacency lost data array.
  /// @param[in] offsets Offsets into `data` for each node, where
  /// `offsets[i]` is the first index in `data` for node `i`. The last
  /// index in `offsets` is the equal to the length of `data`.
  /// @param[in] node_data Node data array where `node_data[i]` is the
  /// data attached to node `i`.
  template <typename W0, typename W1, typename W2>
  // requires std::is_convertible_v<std::remove_cvref_t<U>,
  //                                std::vector<link_type>>
  //              and std::is_convertible_v<std::remove_cvref_t<V>,
  //                                        std::vector<std::int32_t>>
  //              and std::is_convertible_v<std::remove_cvref_t<W>,
  //                                        std::vector<NodeData>>
  AdjacencyList(W0&& data, W1&& offsets, W2&& node_data)
      : _array(std::forward<W0>(data)), _offsets(std::forward<W1>(offsets)),
        _node_data(std::forward<W2>(node_data))
  {
    assert(_node_data.has_value()
           and _node_data->size() == _offsets.size() - 1);
    // _array.reserve(_offsets.back());
    assert(_offsets.back() == (std::int32_t)_array.size());
  }

  /// @brief Construct trivial adjacency list where each of the n nodes
  /// is connected to itself.
  ///
  /// @param[in] n Number of nodes.
  explicit AdjacencyList(std::int32_t n)
      : AdjacencyList(create_iota<std::vector<std::int32_t>>(n),
                      create_iota<std::vector<std::int32_t>>(n + 1))
  {
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a `std::vector<<std::vector<std::size_t>>`,
  /// `std::vector<<std::set<std::size_t>>`, etc).
  ///
  /// @param[in] data Adjacency list data, where `std::next(data, i)`
  /// points to the container of links (edges) for node `i`.
  template <typename X>
  explicit AdjacencyList(const std::vector<X>& data)
  {
    // Initialize offsets and compute total size
    _offsets.reserve(data.size() + 1);
    _offsets.push_back(0);
    for (auto& row : data)
      _offsets.push_back(_offsets.back() + row.size());

    _array.reserve(_offsets.back());
    for (auto& e : data)
      _array.insert(_array.end(), e.begin(), e.end());
  }

  /// Copy constructor
  AdjacencyList(const AdjacencyList& list) = default;

  /// Move constructor
  AdjacencyList(AdjacencyList&& list) = default;

  /// Destructor
  ~AdjacencyList() = default;

  /// Assignment operator
  AdjacencyList& operator=(const AdjacencyList& list) = default;

  /// Move assignment operator
  AdjacencyList& operator=(AdjacencyList&& list) = default;

  /// Equality operator
  /// @return True is the adjacency lists are equal
  bool operator==(const AdjacencyList& list) const
  {
    return this->_array == list._array and this->_offsets == list._offsets;
  }

  /// @brief Get the number of nodes.
  /// @return The number of nodes in the adjacency list
  std::int32_t num_nodes() const { return _offsets.size() - 1; }

  /// @brief Number of connections for given node.
  /// @param[in] node Node index.
  /// @return The number of outgoing links (edges) from the node.
  int num_links(std::size_t node) const
  {
    assert((node + 1) < _offsets.size());
    return _offsets[node + 1] - _offsets[node];
  }

  /// @brief Get the links (edges) for given node.
  /// @param[in] node Node index.
  /// @return Array of outgoing links for the node. The length will be
  /// `AdjacencyList::num_links(node)`.
  std::span<link_type> links(std::size_t node)
  {
    return std::span<link_type>(_array.data() + _offsets[node],
                                _offsets[node + 1] - _offsets[node]);
  }

  /// @brief Get the links (edges) for given node (const version).
  /// @param[in] node Node index.
  /// @return Array of outgoing links for the node. The length will be
  /// `AdjacencyList:num_links(node)`.
  std::span<const link_type> links(std::size_t node) const
  {
    return std::span<const link_type>(_array.data() + _offsets[node],
                                      _offsets[node + 1] - _offsets[node]);
  }

  /// Return contiguous array of links for all nodes (const version).
  std::span<const link_type> array() const
  {
    return std::span<const link_type>(_array);
  }

  /// Return contiguous array of links for all nodes.
  std::span<link_type> array() { return _array; }

  /// Offset for each node in array() (const version).
  const std::vector<std::int32_t>& offsets() const { return _offsets; }

  /// Offset for each node in array().
  std::vector<std::int32_t>& offsets() { return _offsets; }

  /// Return node data (if present), where `node_data()[i]` is the data
  /// for node `i` (const version).
  const std::optional<std::vector<NodeData>>& node_data() const
  {
    return _node_data;
  }

  /// Return node data (if present), where `node_data()[i]` is the data for node
  /// `i`.
  std::optional<std::vector<NodeData>>& node_data() { return _node_data; }

  /// @brief Informal string representation (pretty-print).
  /// @return String representation of the adjacency list.
  std::string str() const
  {
    std::stringstream s;
    s << "<AdjacencyList> with " + std::to_string(this->num_nodes()) + " nodes"
      << std::endl;
    for (std::size_t e = 0; e < _offsets.size() - 1; ++e)
    {
      s << "  " << e << ": [";
      for (auto link : this->links(e))
        s << link << " ";
      s << "]" << '\n';
    }
    return s.str();
  }

private:
  // Connections (links/edges) for all entities stored as a contiguous
  // array
  // std::vector<link_type> _array;
  U _array;

  // Position of first connection for each entity (using local index)
  // std::vector<std::int32_t> _offsets;
  V _offsets;

  // Node data, where _node_data[i] is the data associated with node `i`
  std::optional<std::vector<NodeData>> _node_data = std::nullopt;
};

/// @private Deduction
// template <typename T, typename U>
// AdjacencyList(T, U) -> AdjacencyList<typename T::value_type, std::nullptr_t>;
template <typename T, typename U>
AdjacencyList(T, U) -> AdjacencyList<T, U, std::nullptr_t>;

/// @private Deduction
// template <typename T, typename U, typename W>
// AdjacencyList(T, U, W)
//     -> AdjacencyList<typename T::value_type, typename W::value_type>;
template <typename T, typename U, typename W>
AdjacencyList(T, U, W) -> AdjacencyList<T, U, typename W::value_type>;

/// @brief Construct a constant degree (valency) adjacency list.
///
/// A constant degree graph has the same number of links (edges) for
/// every node.
///
/// @param[in] data Adjacency array.
/// @param[in] degree Number of (outgoing) links for each node.
/// @return An adjacency list.
template <typename V = std::nullptr_t, typename U>
  requires requires {
    typename std::decay_t<U>::value_type;
    requires std::convertible_to<
        U, std::vector<typename std::decay_t<U>::value_type>>;
  }
AdjacencyList<std::vector<typename std::decay_t<U>::value_type>,
              std::vector<std::int32_t>, V>
regular_adjacency_list(U&& data, int degree)
{
  if (degree == 0 and !data.empty())
  {
    throw std::runtime_error("Degree is zero but data is not empty for "
                             "constant degree AdjacencyList");
  }

  if (degree > 0 and data.size() % degree != 0)
  {
    throw std::runtime_error(
        "Incompatible data size and degree for constant degree AdjacencyList");
  }

  std::int32_t num_nodes = degree == 0 ? data.size() : data.size() / degree;
  std::vector<std::int32_t> offsets(num_nodes + 1, 0);
  for (std::size_t i = 1; i < offsets.size(); ++i)
    offsets[i] = offsets[i - 1] + degree;
  return AdjacencyList<std::vector<typename std::decay_t<U>::value_type>,
                       std::vector<std::int32_t>, V>(std::forward<U>(data),
                                                     std::move(offsets));
}

} // namespace dolfinx::graph
