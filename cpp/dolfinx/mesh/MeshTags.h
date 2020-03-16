// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/IndexMap.h>
#include "Mesh.h"
#include "Topology.h"
#include <map>
#include <memory>
#include <utility>

namespace dolfinx
{
namespace mesh
{

/// A MeshTags is a class used to tag mesh entities using their
/// local-to-process index and an attached value.
/// MeshTags is a sparse data storage class, since it allows to
/// tag only few mesh entities.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
  /// @param[in] indices Array of indices, will be copied.
  ///    Local-to-process.
  /// @param[in] values Array of values attached to indices,
  ///    will be copied.
  MeshTags(std::shared_ptr<const Mesh> mesh, int dim,
           const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
           const Eigen::Array<T, Eigen::Dynamic, 1>& values);

  /// Destructor
  ~MeshTags() = default;

  /// Associated mesh
  const std::shared_ptr<const Mesh> mesh;

  /// Topological dimension of tagged mesh entities
  const int dim;

  /// Indices of tagged mesh entities, local-to-process
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices();

  /// Indices of tagged mesh entities, local-to-process (const.)
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices() const;

  /// Values attached to mesh entities
  Eigen::Array<T, Eigen::Dynamic, 1>& values();

  /// Values attached to mesh entities (const.)
  const Eigen::Array<T, Eigen::Dynamic, 1>& values() const;

  /// Append new indices with their values
  /// @param[in] indices
  /// @param[in] values
  void append(const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
              const Eigen::Array<T, Eigen::Dynamic, 1>& values);

  /// Append new indices with their values, appends only indices not already
  /// present
  /// @param[in] indices
  /// @param[in] values
  void
  append_unique(const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
                const Eigen::Array<T, Eigen::Dynamic, 1>& values);

  /// Name
  std::string name = "mesh_tags";

  /// Unique ID
  const std::size_t id = common::UniqueIdGenerator::id();

private:
  // Local-to-process indices of tagged entities
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _indices;

  // Values attached to entities
  Eigen::Array<T, Eigen::Dynamic, 1> _values;
};

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(
    std::shared_ptr<const Mesh> mesh, int dim,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
    const Eigen::Array<T, Eigen::Dynamic, 1>& values)
    : mesh(mesh), dim(dim), _indices(indices), _values(values)
{
  if (indices.rows() != values.rows())
    throw std::runtime_error("Indices and values arrays must match in size.");

  assert(mesh);
  const int D = mesh->topology().dim();
  mesh->create_connectivity(dim, D);
}
//---------------------------------------------------------------------------
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>& MeshTags<T>::values()
{
  return _values;
}
//---------------------------------------------------------------------------
template <typename T>
const Eigen::Array<T, Eigen::Dynamic, 1>& MeshTags<T>::values() const
{
  return _values;
}
//---------------------------------------------------------------------------
template <typename T>
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& MeshTags<T>::indices()
{
  return _indices;
}
//---------------------------------------------------------------------------
template <typename T>
const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
MeshTags<T>::indices() const
{
  return _indices;
}
//---------------------------------------------------------------------------
template <typename T>
void MeshTags<T>::append(
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
    const Eigen::Array<T, Eigen::Dynamic, 1>& values)
{
  if (indices.rows() != values.rows())
    throw std::runtime_error("Indices and values arrays must match in size.");

  const int new_size = _indices.rows() + indices.rows();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> new_indices(new_size);
  new_indices << _indices,
                 indices;
  _indices = new_indices;

  Eigen::Array<T, Eigen::Dynamic, 1> new_values(new_size);
  new_values << _values,
                values;
  _values = new_values;
}
//---------------------------------------------------------------------------
template <typename T>
void MeshTags<T>::append_unique(
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices,
    const Eigen::Array<T, Eigen::Dynamic, 1>& values)
{
  if (indices.rows() != values.rows())
    throw std::runtime_error("Indices and values arrays must match in size.");

  const int num_entities = mesh->topology().index_map(dim)->size_local()
                           + mesh->topology().index_map(dim)->num_ghosts();

  // Prepare a large vector which says if an enitty index was already inserted
  // Trading time complexity for memory cost
  std::vector<bool> inserted(num_entities, false);

  std::vector<std::int32_t> unique_indices;
  unique_indices.reserve(_indices.rows() + indices.rows());

  std::vector<T> unique_values;
  unique_values.reserve(_values.rows() + values.rows());

  for (int i = 0; i < _indices.rows(); ++i)
  {
    if (!inserted[_indices[i]])
    {
      unique_indices.push_back(_indices[i]);
      unique_values.push_back(_values[i]);
      inserted[_indices[i]] = true;
    }
  }

  for (int i = 0; i < indices.rows(); ++i)
  {
    if (!inserted[indices[i]])
    {
      unique_indices.push_back(indices[i]);
      unique_values.push_back(values[i]);
      inserted[indices[i]] = true;
    }
  }

  _indices = Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
      unique_indices.data(), unique_indices.size());
  _values = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(
      unique_values.data(), unique_values.size());
}
//---------------------------------------------------------------------------
} // namespace mesh
} // namespace dolfinx
