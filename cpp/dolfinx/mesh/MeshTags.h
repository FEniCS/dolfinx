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
/// tag only few mesh entities. This class always removes duplicates.
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
  /// @param[in] sorted True for already sorted indices.
  /// @param[in] unique True for unique indices.
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
           const std::vector<std::int32_t>& indices,
           const std::vector<T>& values, const bool sorted = false,
           const bool unique = false);

  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
  /// @param[in] indices Array of indices, will be copied, sorted
  ///    with duplicates removed. Local-to-process.
  /// @param[in] values Array of values attached to indices,
  ///    will be copied, sorted and duplicates removed according
  ///    to indices array.
  /// @param[in] sorted True for already sorted indices.
  /// @param[in] unique True for unique indices.
  MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
           std::vector<std::int32_t>&& indices, std::vector<T>&& values,
           const bool sorted = false, const bool unique = false);

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
  std::size_t id() const { return _unique_id; }

private:
  // Unique identifier
  std::size_t _unique_id;

  /// Associated mesh
  std::shared_ptr<const Mesh> _mesh;

  /// Topological dimension of tagged mesh entities
  int _dim;

  // Local-to-process indices of tagged entities
  std::vector<std::int32_t> _indices;

  // Values attached to entities
  std::vector<T> _values;

  // Sort indices and values according to indices
  void sort();

  // Remove duplicates in indices and values according to indices
  void remove_duplicates();
};

//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
                      const std::vector<std::int32_t>& indices,
                      const std::vector<T>& values, const bool sorted,
                      const bool unique)
    : _unique_id(common::UniqueIdGenerator::id()), _mesh(mesh), _dim(dim),
      _indices(indices), _values(values)
{
  if (indices.size() != values.size())
    throw std::runtime_error("Indices and values arrays must match in size.");

  if (!sorted)
    sort();

  if (!unique)
    remove_duplicates();
}
//---------------------------------------------------------------------------
template <typename T>
MeshTags<T>::MeshTags(const std::shared_ptr<const Mesh>& mesh, int dim,
                      std::vector<std::int32_t>&& indices,
                      std::vector<T>&& values, const bool sorted,
                      const bool unique)
    : _unique_id(common::UniqueIdGenerator::id()), _mesh(mesh), _dim(dim),
      _indices(std::move(indices)), _values(std::move(values))
{
  if (indices.size() != values.size())
    throw std::runtime_error("Indices and values arrays must match in size.");

  if (!sorted)
    sort();

  if (!unique)
    remove_duplicates();
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
template <typename T>
void MeshTags<T>::sort()
{
  // Prepare the sorting permutation
  std::vector<int> perm(_indices.size());
  std::iota(perm.begin(), perm.end(), 0);

  // Swap into a temporaries
  std::vector<std::int32_t> indices;
  indices.swap(_indices);
  std::vector<T> values;
  values.swap(_values);

  std::sort(perm.begin(), perm.end(), [&indices](const int a, const int b) {
    return (indices[a] < indices[b]);
  });

  // Make sure vectors are empty and preallocate space
  _indices.clear();
  _values.clear();
  _indices.reserve(indices.size());
  _values.reserve(values.size());

  // Apply sorting and insert
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    _indices.push_back(indices[perm[i]]);
    _values.push_back(values[perm[i]]);
  }
}
//---------------------------------------------------------------------------
template <typename T>
void MeshTags<T>::remove_duplicates()
{
  // Algorithm would fail for empty vector
  if (_indices.size() == 0)
    return;

  std::size_t last_unique = 0;
  for (std::size_t i = 0; i < _indices.size(); ++i)
  {
    if (_indices[i] > _indices[last_unique])
    {
      _indices[++last_unique] = _indices[i];
      _values[last_unique] = _values[i];
    }
  }
  _indices.erase(_indices.begin() + last_unique + 1, _indices.end());
  _values.erase(_values.begin() + last_unique + 1, _values.end());
}
} // namespace mesh
} // namespace dolfinx
