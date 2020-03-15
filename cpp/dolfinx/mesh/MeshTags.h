// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "Topology.h"
#include <map>
#include <memory>
#include <utility>

namespace dolfinx
{
namespace mesh
{

template <typename T>
class MeshTags
{
public:
  /// Create from entities of given dimension on a mesh
  /// @param[in] mesh The mesh associated with the tags
  /// @param[in] dim Topological dimension of mesh entities
  ///    to tag.
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

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& indices() const;

  /// Values attached to mesh entities
  Eigen::Array<T, Eigen::Dynamic, 1>& values();

  const Eigen::Array<T, Eigen::Dynamic, 1>& values() const;

  /// Name
  std::string name = "mesh_tags";

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
} // namespace mesh
} // namespace dolfinx
