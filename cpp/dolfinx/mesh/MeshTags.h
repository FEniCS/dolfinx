// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
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
/// tag only few mesh entities. This class sorts and removes duplicates
/// in indices on construction.
/// @tparam Type
template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
  /// @param[in] indices Array of indices, will be copied, sorted
  ///    with duplicates removed. Local-to-process.
  /// @param[in] values Array of values attached to indices,
  ///    will be copied, sorted and duplicates removed according
  ///    to indices array.
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
           const std::vector<std::int32_t>& indices,
           const std::vector<T>& values);

  /// Move constructor
  MeshTags(MeshTags&& mt) = default;

  /// Destructor
  ~MeshTags() = default;

  /// Move assignment
  MeshTags& operator=(MeshTags&&) = default;

  /// Indices of tagged mesh entities, local-to-process (const.)
  const std::vector<std::int32_t>& indices() const;

  /// Values attached to mesh entities (const.)
  const std::vector<T>& values() const;

  /// Return topological dimension of tagged entities
  int dim() const;

  /// Return mesh
  std::shared_ptr<const Mesh> mesh() const;

  /// Name
  std::string name = "mesh_tags";

  /// Unique ID
  std::size_t id = common::UniqueIdGenerator::id();

private:
  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;

  /// Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  /// Topological dimension of tagged mesh entities
  int _dim;
};

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
                      const std::vector<std::int32_t>& indices,
                      const std::vector<T>& values)
    : _mesh(mesh), _dim(dim), _indices(indices.size()), _values(values.size())
{
  if (indices.size() != values.size())
    throw std::runtime_error("Indices and values arrays must match in size.");

  //
  // Sort indices and values according to indices and remove duplicates
  //

  // Prepare the sorting permutation
  std::vector<int> perm(indices.size());
  std::iota(perm.begin(), perm.end(), 0);

  std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
    return (indices[a] < indices[b]);
  });

  // Apply sorting and insert
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    _indices[i] = indices[perm[i]];
    _values[i] = values[perm[i]];
  }

  // Remove duplicates
  const auto it = std::unique(_indices.begin(), _indices.end());
  const int unique_size = std::distance(_indices.begin(), it);

  _indices.erase(it, _indices.end());
  _values.erase(_values.begin(), _values.begin() + unique_size);
}
//---------------------------------------------------------------------------
template <typename T>
const std::vector<T>& MeshTags<T>::values() const
{
  return _values;
}
//---------------------------------------------------------------------------
template <typename T>
const std::vector<std::int32_t>& MeshTags<T>::indices() const
{
  return _indices;
}
//---------------------------------------------------------------------------
template <typename T>
int MeshTags<T>::dim() const
{
  return _dim;
}
//---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const Mesh> MeshTags<T>::mesh() const
{
  return _mesh;
}
//---------------------------------------------------------------------------
} // namespace mesh
} // namespace dolfinx
