// Copyright (C) 2009-2025 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPI.h"
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/// Generic tools
namespace dolfinx::common
{
///@brief Sort two arrays based on the values in array `indices`.
///
/// Any duplicate indices and the corresponding value are removed. In
/// the case of duplicates, the entry with the smallest value is
/// retained.
///
/// @param[in] indices Array of indices.
/// @param[in] values Array of values.
/// @return Sorted (indices, values), with sorting based on indices.
template <typename U, typename V>
std::pair<std::vector<typename U::value_type>,
          std::vector<typename V::value_type>>
sort_unique(const U& indices, const V& values)
{
  if (indices.size() != values.size())
    throw std::runtime_error("Cannot sort two arrays of different lengths");

  using T = typename std::pair<typename U::value_type, typename V::value_type>;
  std::vector<T> data(indices.size());
  std::ranges::transform(indices, values, data.begin(),
                         [](auto& idx, auto& v) -> T { return {idx, v}; });

  // Sort make unique
  std::ranges::sort(data);
  auto it = std::ranges::unique(data, [](auto& a, auto& b)
                                { return a.first == b.first; })
                .begin();

  std::vector<typename U::value_type> indices_new;
  std::vector<typename V::value_type> values_new;
  indices_new.reserve(data.size());
  values_new.reserve(data.size());
  std::transform(data.begin(), it, std::back_inserter(indices_new),
                 [](auto& d) { return d.first; });
  std::transform(data.begin(), it, std::back_inserter(values_new),
                 [](auto& d) { return d.second; });

  return {std::move(indices_new), std::move(values_new)};
}

/// @brief Compute a hash of a given object
///
/// The hash is computed using Boost container hash
/// (https://www.boost.org/doc/libs/release/libs/container_hash/).
///
/// @param[in] x The object to compute a hash of.
/// @return The hash values.
template <class T>
std::size_t hash_local(const T& x)
{
  boost::hash<T> hash;
  return hash(x);
}

/// @brief Compute a hash for a distributed (MPI) object.
///
/// A hash is computed on each process for the local part of the object.
/// Then, a hash of the std::vector containing each local hash key in
/// rank order is returned.
///
/// @note Collective
///
/// @param[in] comm The communicator on which to compute the hash.
/// @param[in] x The object to compute a hash of.
/// @return The hash values.
template <class T>
std::size_t hash_global(MPI_Comm comm, const T& x)
{
  // Compute local hash
  std::size_t local_hash = hash_local(x);

  // Gather hash keys on root process
  std::vector<std::size_t> all_hashes(dolfinx::MPI::size(comm));
  int err = MPI_Gather(&local_hash, 1, dolfinx::MPI::mpi_t<std::size_t>,
                       all_hashes.data(), 1, dolfinx::MPI::mpi_t<std::size_t>,
                       0, comm);
  dolfinx::MPI::check_error(comm, err);

  // Hash the received hash keys
  boost::hash<std::vector<std::size_t>> hash;
  std::size_t global_hash = hash(all_hashes);

  // Broadcast hash key to all processes
  err = MPI_Bcast(&global_hash, 1, dolfinx::MPI::mpi_t<std::size_t>, 0, comm);
  dolfinx::MPI::check_error(comm, err);

  return global_hash;
}

/// @brief Build communication graph data as a JSON string.
///
/// The data string can be decoded (loaded) to create a Python object
/// from which a [NetworkX](https://networkx.org/) graph can be
/// constructed.
///
/// See ::comm_graph for a description of the data.
///
/// @param[in] g Communication graph.
/// @return JSON string representing the communication graph. Edge
/// data is data volume (`weight`) and local/remote memory indicator
/// (`local==1` is an edge to an shared memory process/rank, other
/// wise the target node is a remote memory rank).
std::string comm_to_json(
    const graph::AdjacencyList<std::tuple<int, std::size_t, std::int8_t>,
                               std::pair<std::int32_t, std::int32_t>>& g);
} // namespace dolfinx::common
