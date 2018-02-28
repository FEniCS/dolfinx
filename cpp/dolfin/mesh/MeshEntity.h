// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include <dolfin/geometry/Point.h>

namespace dolfin
{

namespace mesh
{
/// A MeshEntity represents a mesh entity associated with
/// a specific topological dimension of some _Mesh_.

class MeshEntity
{
public:
  /// Default Constructor
  MeshEntity() : _mesh(nullptr), _dim(0), _local_index(0) {}

  /// Constructor
  ///
  /// @param   mesh (_Mesh_)
  ///         The mesh.
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  /// @param     index (std::size_t)
  ///         The index.
  MeshEntity(const Mesh& mesh, std::size_t dim, std::size_t index)
      : _mesh(&mesh), _dim(dim), _local_index(index)
  {
    // Do nothing
  }

  /// Destructor
  ~MeshEntity();

  /// Initialize mesh entity with given data
  ///
  /// @param      mesh (_Mesh_)
  ///         The mesh.
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  /// @param     index (std::size_t)
  ///         The index.
  void init(const Mesh& mesh, std::size_t dim, std::size_t index);

  /// Comparison Operator
  ///
  /// @param e (MeshEntity)
  ///         Another mesh entity
  ///
  ///  @return    bool
  ///         True if the two mesh entities are equal.
  bool operator==(const MeshEntity& e) const
  {
    return (_mesh == e._mesh && _dim == e._dim
            && _local_index == e._local_index);
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
  inline std::size_t dim() const { return _dim; }

  /// Return index of mesh entity
  ///
  /// @return     std::size_t
  ///         The index.
  std::int32_t index() const { return _local_index; }

  /// Return global index of mesh entity
  ///
  /// @return     std::size_t
  ///         The global index. Set to
  ///         std::numerical_limits<std::size_t>::max() if global index
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
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  /// The number of local incident MeshEntity objects of given
  /// dimension.
  inline std::size_t num_entities(std::size_t dim) const
  {
    return _mesh->topology()(_dim, dim).size(_local_index);
  }

  /// Return global number of incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  ///         The number of global incident MeshEntity objects of given
  ///         dimension.
  std::size_t num_global_entities(std::size_t dim) const
  {
    return _mesh->topology()(_dim, dim).size_global(_local_index);
  }

  /// Return array of indices for incident mesh entities of given
  /// topological dimension
  ///
  /// @param     dim (std::size_t)
  ///         The topological dimension.
  ///
  /// @return     std::size_t
  ///         The index for incident mesh entities of given dimension.
  const std::int32_t* entities(std::size_t dim) const
  {
    const std::int32_t* initialized_mesh_entities
        = _mesh->topology()(_dim, dim)(_local_index);
    dolfin_assert(initialized_mesh_entities);
    return initialized_mesh_entities;
  }

  /// Check if given entity is incident
  ///
  /// @param     entity (_MeshEntity_)
  ///         The entity.
  ///
  ///  @return    bool
  ///         True if the given entity is incident
  bool incident(const MeshEntity& entity) const;

  /// Compute local index of given incident entity (error if not
  /// found)
  ///
  /// @param     entity (_MeshEntity_)
  ///         The mesh entity.
  ///
  /// @return     std::size_t
  ///         The local index of given entity.
  std::size_t index(const MeshEntity& entity) const;

  /// Compute midpoint of cell
  ///
  /// @return geometry::Point
  ///         The midpoint of the cell.
  geometry::Point midpoint() const;

  /// Determine whether an entity is a 'ghost' from another
  /// process
  /// @return bool
  ///    True if entity is a ghost entity
  bool is_ghost() const
  {
    return (_local_index >= (std::int32_t)_mesh->topology().ghost_offset(_dim));
  }

  /// Return set of sharing processes
  /// @return std::set<std::uint32_t>
  ///   List of sharing processes
  std::set<std::uint32_t> sharing_processes() const
  {
    const std::map<std::int32_t, std::set<std::uint32_t>>& sharing_map
        = _mesh->topology().shared_entities(_dim);
    const auto map_it = sharing_map.find(_local_index);
    if (map_it == sharing_map.end())
      return std::set<std::uint32_t>();
    else
      return map_it->second;
  }

  /// Determine if an entity is shared or not
  /// @return bool
  ///    True if entity is shared
  bool is_shared() const
  {
    if (_mesh->topology().have_shared_entities(_dim))
    {
      const std::map<std::int32_t, std::set<std::uint32_t>>& sharing_map
          = _mesh->topology().shared_entities(_dim);
      return (sharing_map.find(_local_index) != sharing_map.end());
    }
    return false;
  }

  /// Get ownership of this entity - only really valid for cells
  /// @return std::uint32_t
  ///    Owning process
  std::uint32_t owner() const;

  // Note: Not a subclass of Variable for efficiency!
  /// Return informal string representation (pretty-print)
  ///
  /// @param      verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return      std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

protected:
  // Friends
  template <typename T>
  friend class MeshRange;
  template <typename T>
  friend class EntityRange;
  template <typename T>
  friend class MeshIterator;
  template <typename T>
  friend class MeshEntityIteratorNew;

  // The mesh
  Mesh const* _mesh;

  // Topological dimension
  std::uint32_t _dim;

  // Local index of entity within topological dimension
  std::int32_t _local_index;
};
}
}