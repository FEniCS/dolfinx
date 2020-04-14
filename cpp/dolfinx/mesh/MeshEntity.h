// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "Topology.h"
#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx::mesh
{

/// A MeshEntity represents a mesh entity associated with a specific
/// topological dimension of some Mesh. A MeshEntity object is left in
/// an undefined state if the Mesh that it is constructed with is
/// destroyed.

class MeshEntity
{
public:
  /// Constructor
  /// @param[in] mesh The mesh
  /// @param[in] dim The topological dimension
  /// @param[in] index The entity index
  MeshEntity(const Mesh& mesh, int dim, std::int32_t index)
      : _mesh(&mesh), _dim(dim), _local_index(index)
  {
    // Do nothing
  }

  /// Copy constructor
  MeshEntity(const MeshEntity& e) = default;

  /// Move constructor
  MeshEntity(MeshEntity&& e) = default;

  /// Destructor
  ~MeshEntity() = default;

  /// Assignment operator
  MeshEntity& operator=(const MeshEntity& e) = default;

  /// Move assignment operator
  MeshEntity& operator=(MeshEntity&& e) = default;

  /// Return mesh associated with mesh entity
  /// @return The mesh
  const Mesh& mesh() const { return *_mesh; }

  /// Return topological dimension
  /// @return The topological dimension
  int dim() const { return _dim; }

  /// Return index of mesh entity
  /// @return The index
  std::int32_t index() const { return _local_index; }

  /// Return array of indices for incident mesh entities of given
  /// topological dimension
  /// @param[in] dim The topological dimension
  /// @return The index for incident mesh entities of given dimension
  auto entities(int dim) const
  {
    assert(_mesh->topology().connectivity(_dim, dim));
    return _mesh->topology().connectivity(_dim, dim)->links(_local_index);
  }

private:
  // The mesh
  Mesh const* _mesh;

  // Topological dimension
  int _dim;

  // Local index of entity within topological dimension
  std::int32_t _local_index;
};

} // namespace dolfinx::mesh
