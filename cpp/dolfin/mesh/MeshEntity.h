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
  /// @param   mesh (_Mesh_)
  ///         The mesh.
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  /// @param     index (std::size_t)
  ///         The index.
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

  /// Assignement operator
  MeshEntity& operator=(const MeshEntity& e) = default;

  /// Move assignement operator
  MeshEntity& operator=(MeshEntity&& e) = default;

  /// Comparison Operator
  ///
  /// @param e (MeshEntity)
  ///         Another mesh entity
  ///
  ///  @return    bool
  ///         True if the two mesh entities are equal.
  bool operator==(const MeshEntity& e) const
  {
    return (_mesh == e._mesh and _dim == e._dim
            and _local_index == e._local_index);
  }

  /// Comparison Operator
  ///
  /// @param e (MeshEntity)
  ///         Another mesh entity.
  ///
  /// @return     bool
  ///         True if the two mesh entities are NOT equal.
  bool operator!=(const MeshEntity& e) const { return !operator==(e); }

  /// Return mesh associated with mesh entity
  ///
  /// @return Mesh
  ///         The mesh.
  const Mesh& mesh() const { return *_mesh; }

  /// Return topological dimension
  ///
  /// @return     std::size_t
  ///         The dimension.
  inline int dim() const { return _dim; }

  /// Return index of mesh entity
  ///
  /// @return     std::size_t
  ///         The index.
  std::int32_t index() const { return _local_index; }

  /// Return global index of mesh entity
  ///
  /// @return     std::size_t
  ///         The global index. Set to -1  if global index
  ///         has not been computed
  std::int64_t global_index() const
  {
    const std::vector<std::int64_t>& global_indices
        = _mesh->topology().global_indices(_dim);
    if (global_indices.empty())
      return -1;
    return global_indices[_local_index];
  }

  /// Return local number of incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (int)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  /// The number of local incident MeshEntity objects of given
  /// dimension.
  inline std::size_t num_entities(int dim) const
  {
    if (dim == _dim)
      return 1;
    else
    {
      assert(_mesh->topology().connectivity(_dim, dim));
      return _mesh->topology().connectivity(_dim, dim)->size(_local_index);
    }
  }

  /// Return global number of incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (int)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  ///         The number of global incident MeshEntity objects of given
  ///         dimension.
  std::size_t num_global_entities(int dim) const
  {
    if (dim == _dim)
      return 1;
    else
    {
      assert(_mesh->topology().connectivity(_dim, dim));
      return _mesh->topology().connectivity(_dim, dim)->size_global(
          _local_index);
    }
  }

  /// Return array of indices for incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  ///         The index for incident mesh entities of given dimension.
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
  /// @param     entity (_MeshEntity_)
  ///         The mesh entity.
  ///
  /// @return     std::size_t
  ///         The local index of given entity.
  int index(const MeshEntity& entity) const;

  /// Determine if an entity is shared or not
  /// @return bool
  ///    True if entity is shared
  bool is_shared() const
  {
    if (_mesh->topology().have_shared_entities(_dim))
    {
      const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map
          = _mesh->topology().shared_entities(_dim);
      return (sharing_map.find(_local_index) != sharing_map.end());
    }
    return false;
  }

  /// Return informal string representation (pretty-print)
  ///
  /// @param      verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return      std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

protected:
  template <typename T>
  friend class MeshRange;
  template <typename T>
  friend class EntityRange;
  template <typename T>
  friend class MeshIterator;
  template <typename T>
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
