// Copyright (C) 2020-2022 Michal Habera and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Topology.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/io/cells.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{

/// @brief MeshTags associate values with mesh topology entities.
///
/// The entity index (local to process) identifies the entity. MeshTags
/// is a *sparse* data storage class; it allows tags to be associated
/// with an arbitrary subset of mesh entities. An entity can have only
/// one associated tag.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// @brief Create a MeshTag from entities of given dimension on a
  /// mesh.
  ///
  /// @param[in] topology Mesh topology on which the tags are associated.
  /// @param[in] dim Topological dimension of mesh entities to tag.
  /// @param[in] indices List of entity indices (indices local to the
  /// process).
  /// @param[in] values List of values for each index in indices. The
  /// size must be equal to the size of `indices`.
  /// @pre `indices` must be sorted and unique.
  template <typename U, typename V>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
                 and std::is_convertible_v<std::remove_cvref_t<V>,
                                           std::vector<T>>
  MeshTags(std::shared_ptr<const Topology> topology, int dim, U&& indices,
           V&& values)
      : _topology(topology), _dim(dim), _indices(std::forward<U>(indices)),
        _values(std::forward<V>(values))
  {
    if (_indices.size() != _values.size())
    {
      throw std::runtime_error(
          "Indices and values arrays must have same size.");
    }
#ifndef NDEBUG
    if (!std::is_sorted(_indices.begin(), _indices.end()))
      throw std::runtime_error("MeshTag data is not sorted");
    if (std::adjacent_find(_indices.begin(), _indices.end()) != _indices.end())
      throw std::runtime_error("MeshTag data has duplicates");
#endif
  }

  /// Copy constructor
  MeshTags(const MeshTags& tags) = default;

  /// Move constructor
  MeshTags(MeshTags&& tags) = default;

  /// Destructor
  ~MeshTags() = default;

  /// Move assignment
  MeshTags& operator=(const MeshTags& tags) = default;

  /// Move assignment
  MeshTags& operator=(MeshTags&& tags) = default;

  /// @brief Find all entities with a given tag value
  /// @param[in] value The value
  /// @return Indices of tagged entities. The indices are sorted.
  std::vector<std::int32_t> find(const T value) const
  {
    std::size_t n = std::count(_values.begin(), _values.end(), value);
    std::vector<std::int32_t> indices;
    indices.reserve(n);
    for (std::int32_t i = 0; i < _values.size(); ++i)
    {
      if (_values[i] == value)
        indices.push_back(_indices[i]);
    }
    return indices;
  }

  /// Indices of tagged topology entities (local-to-process). The
  /// indices are sorted.
  std::span<const std::int32_t> indices() const { return _indices; }

  /// Values attached to topology entities
  std::span<const T> values() const { return _values; }

  /// Return topological dimension of tagged entities
  int dim() const { return _dim; }

  /// Return topology
  std::shared_ptr<const Topology> topology() const { return _topology; }

  /// Name
  std::string name = "mesh_tags";

private:
  // Associated topology
  std::shared_ptr<const Topology> _topology;

  // Topological dimension of tagged topology entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;
};

/// @brief Create MeshTags from arrays
/// @param[in] topology Mesh topology that the tags are associated with
/// @param[in] dim Topological dimension of tagged entities
/// @param[in] entities Local vertex indices for tagged entities.
/// @param[in] values Tag values for each entity in `entities`. The
/// length of `values` must be equal to number of rows in `entities`.
/// @note Entities that do not exist on this rank are ignored.
/// @warning `entities` must not contain duplicate entities.
template <typename T>
MeshTags<T> create_meshtags(std::shared_ptr<const Topology> topology, int dim,
                            const graph::AdjacencyList<std::int32_t>& entities,
                            std::span<const T> values)
{
  spdlog::info(
      "Building MeshTags object from tagged entities (defined by vertices).");

  // Compute the indices of the topology entities (index is set to -1 if
  // it can't be found)
  assert(topology);
  const std::vector<std::int32_t> indices
      = entities_to_index(*topology, dim, entities.array());
  if (indices.size() != values.size())
  {
    throw std::runtime_error(
        "Duplicate mesh entities when building MeshTags object.");
  }

  // Sort the indices and values by indices
  auto [indices_sorted, values_sorted] = common::sort_unique(indices, values);

  // Remove any entities that were not found (these have an index of -1)
  auto it0 = std::ranges::lower_bound(indices_sorted, 0);
  std::size_t pos0 = std::distance(indices_sorted.begin(), it0);
  indices_sorted.erase(indices_sorted.begin(), it0);
  values_sorted.erase(values_sorted.begin(),
                      std::next(values_sorted.begin(), pos0));

  return MeshTags<T>(topology, dim, std::move(indices_sorted),
                     std::move(values_sorted));
}
} // namespace dolfinx::mesh
