// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Connectivity.h"
#include "Mesh.h"
#include "Topology.h"

namespace dolfin
{

namespace mesh
{
/// A MeshEntity represents a mesh entity associated with a specific
/// topological dimension of some _Mesh_.

class MeshEntity
{
public:
  /// Constructor
  ///
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

  /// Comparison Operator
  ///
  /// @param e Another mesh entity
  ///  @return True if the two mesh entities are equal.
  bool operator==(const MeshEntity& e) const
  {
    return (_mesh == e._mesh and _dim == e._dim
            and _local_index == e._local_index);
  }

  /// Comparison Operator
  ///
  /// @param e Another mesh entity.
  /// @return True if the two mesh entities are NOT equal.
  bool operator!=(const MeshEntity& e) const { return !operator==(e); }

  /// Return mesh associated with mesh entity
  ///
  /// @return The mesh.
  const Mesh& mesh() const { return *_mesh; }

  /// Return topological dimension
  ///
  /// @return The dimension
  int dim() const { return _dim; }

  /// Return index of mesh entity
  ///
  /// @return The index
  std::int32_t index() const { return _local_index; }

  /// Return array of indices for incident mesh entities of given
  /// topological dimension
  ///
  /// @param dim The topological dimension
  /// @return The index for incident mesh entities of given dimension
  const std::int32_t* entities(int dim) const
  {
    if (dim == _dim)
      return &_local_index;
    else
    {
      assert(_mesh->topology().connectivity(_dim, dim));
      const std::int32_t* initialized_mesh_entities
          = _mesh->topology().connectivity(_dim, dim)->connections(
              _local_index);
      assert(initialized_mesh_entities);
      return initialized_mesh_entities;
    }
  }

  /// Compute local index of given incident entity (error if not
  /// found)
  ///
  /// @param[in] entity The mesh entity.
  /// @return The local index of given entity.
  int index(const MeshEntity& entity) const;

  /// Return informal string representation (pretty-print)
  ///
  /// @param[in] verbose Flag to turn on additional output
  /// @return An informal representation of the function space
  std::string str(bool verbose) const;

protected:
  friend class MeshRange;
  friend class EntityRange;
  friend class MeshIterator;
  friend class MeshEntityIterator;

  // The mesh
  Mesh const* _mesh;

  // Topological dimension
  int _dim;

  // Local index of entity within topological dimension
  std::int32_t _local_index;
};

} // namespace mesh
} // namespace dolfin
