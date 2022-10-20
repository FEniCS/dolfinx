// Copyright (C) 2020-2022 Michal Habera and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/io/cells.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{

/// @brief MeshTags associate values with mesh entities.
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
  /// @brief Create a MeshTag from entities of given dimension on a mesh.
  ///
  /// @param[in] mesh The mesh on which the tags are associated
  /// @param[in] dim Topological dimension of mesh entities to tag
  /// @param[in] indices std::vector<std::int32> of sorted and unique
  /// entity indices (indices local to the process)
  /// @param[in] values std::vector<T> of values for each index in
  /// indices. The size must be equal to the size of @p indices.
  template <typename U, typename V>
  MeshTags(std::shared_ptr<const Mesh> mesh, int dim, U&& indices, V&& values)
      : _mesh(mesh), _dim(dim), _indices(std::forward<U>(indices)),
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
    int n = std::count(_values.begin(), _values.end(), value);
    std::vector<std::int32_t> indices(n);
    int counter = 0;
    for (std::int32_t i = 0; i < _values.size(); ++i)
    {
      if (_values[i] == value)
        indices[counter++] = _indices[i];
    }
    return indices;
  }

  /// Indices of tagged mesh entities (local-to-process). The indices
  /// are sorted.
  const std::vector<std::int32_t>& indices() const { return _indices; }

  /// Values attached to mesh entities
  const std::vector<T>& values() const { return _values; }

  /// Return topological dimension of tagged entities
  int dim() const { return _dim; }

  /// Return mesh
  std::shared_ptr<const Mesh> mesh() const { return _mesh; }

  /// Name
  std::string name = "mesh_tags";

private:
  // Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  // Topological dimension of tagged mesh entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;
};

/// @brief Create MeshTags from arrays
/// @param[in] mesh The Mesh that the tags are associated with
/// @param[in] dim Topological dimension of tagged entities
/// @param[in] entities Local vertex indices for tagged entities.
///
/// @param[in] values Tag values for each entity in `entities`. The
/// length of `values` must be equal to number of rows in `entities`.
/// @note Entities that do not exist on this rank are ignored.
/// @warning `entities` must not contain duplicate entities.
template <typename T>
MeshTags<T> create_meshtags(std::shared_ptr<const Mesh> mesh, int dim,
                            const graph::AdjacencyList<std::int32_t>& entities,
                            const std::span<const T>& values)
{
  LOG(INFO)
      << "Building MeshTgas object from tagged entities (defined by vertices).";

  assert(mesh);

  // Compute the indices of the mesh entities (index is set to -1 if it
  // can't be found)
  const std::vector<std::int32_t> indices
      = entities_to_index(mesh->topology(), dim, entities);
  if (indices.size() != values.size())
  {
    throw std::runtime_error(
        "Duplicate mesh entities when building MeshTags object.");
  }

  // Sort the indices and values by indices
  auto [indices_sorted, values_sorted] = common::sort_unique(indices, values);

  // Remove any entities that were not found (these have andindex of -1)
  auto it0 = std::lower_bound(indices_sorted.begin(), indices_sorted.end(), 0);
  std::size_t pos0 = std::distance(indices_sorted.begin(), it0);
  indices_sorted.erase(indices_sorted.begin(), it0);
  values_sorted.erase(values_sorted.begin(),
                      std::next(values_sorted.begin(), pos0));

  return MeshTags<T>(mesh, dim, std::move(indices_sorted),
                     std::move(values_sorted));
}
} // namespace dolfinx::mesh
