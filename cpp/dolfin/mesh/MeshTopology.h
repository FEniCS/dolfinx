// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "MeshConnectivity.h"
#include <dolfin/common/Variable.h>

namespace dolfin
{
namespace mesh
{

/// MeshTopology stores the topology of a mesh, consisting of mesh
/// entities and connectivity (incidence relations for the mesh
/// entities). Note that the mesh entities don't need to be stored,
/// only the number of entities and the connectivity. Any numbering
/// scheme for the mesh entities is stored separately in a
/// MeshFunction over the entities.
///
/// A mesh entity e may be identified globally as a pair e = (dim,
/// i), where dim is the topological dimension and i is the index of
/// the entity within that topological dimension.

class MeshTopology : public common::Variable
{
public:
  /// Create empty mesh topology
  MeshTopology();

  /// Copy constructor
  MeshTopology(const MeshTopology& topology) = default;

  /// Move constructor
  MeshTopology(MeshTopology&& topology) = default;

  /// Destructor
  ~MeshTopology() = default;

  /// Assignment
  MeshTopology& operator=(const MeshTopology& topology) = default;

  /// Return topological dimension
  std::uint32_t dim() const;

  /// Return number of entities for given dimension (local to process)
  std::uint32_t size(std::uint32_t dim) const;

  /// Return global number of entities for given dimension
  std::uint64_t size_global(std::uint32_t dim) const;

  /// Return number of regular (non-ghost) entities or equivalently,
  /// the offset of where ghost entities begin
  inline std::uint32_t ghost_offset(std::uint32_t dim) const
  {
    if (_ghost_offset_index.empty())
      return 0;

    dolfin_assert(dim < _ghost_offset_index.size());
    return _ghost_offset_index[dim];
  }

  /// Clear data for given pair of topological dimensions
  void clear(std::size_t d0, std::size_t d1);

  /// Initialize topology of given maximum dimension
  void init(std::size_t dim);

  /// Set number of local entities (local_size) and global entities
  /// (global_size) for given topological dimension dim
  void init(std::size_t dim, std::int32_t local_size, std::int64_t global_size);

  /// Initialize storage for global entity numbering for entities of
  /// dimension dim
  void init_global_indices(std::size_t dim, std::int64_t size);

  /// Initialise the offset index of ghost entities for this dimension
  void init_ghost(std::size_t dim, std::size_t index);

  /// Set global index for entity of dimension dim and with local
  /// index
  void set_global_index(std::size_t dim, std::int32_t local_index,
                        std::int64_t global_index)
  {
    dolfin_assert(dim < _global_indices.size());
    dolfin_assert(local_index < (std::int32_t)_global_indices[dim].size());
    _global_indices[dim][local_index] = global_index;
  }

  /// Get local-to-global index map for entities of topological
  /// dimension d
  const std::vector<std::int64_t>& global_indices(std::size_t d) const
  {
    dolfin_assert(d < _global_indices.size());
    return _global_indices[d];
  }

  /// Check if global indices are available for entities of
  /// dimension dim
  bool have_global_indices(std::size_t dim) const
  {
    dolfin_assert(dim < _global_indices.size());
    return !_global_indices[dim].empty();
  }

  /// Check whether there are any shared entities calculated
  /// of dimension dim
  bool have_shared_entities(std::uint32_t dim) const
  {
    return (_shared_entities.find(dim) != _shared_entities.end());
  }

  /// Return map from shared entities (local index) to processes
  /// that share the entity
  std::map<std::int32_t, std::set<std::uint32_t>>&
  shared_entities(std::uint32_t dim);

  /// Return map from shared entities (local index) to process that
  /// share the entity (const version)
  const std::map<std::int32_t, std::set<std::uint32_t>>&
  shared_entities(std::uint32_t dim) const;

  /// Return mapping from local ghost cell index to owning process
  /// Since ghost cells are at the end of the range, this is just
  /// a vector over those cells
  std::vector<std::uint32_t>& cell_owner() { return _cell_owner; }

  /// Return mapping from local ghost cell index to owning process (const
  /// version)
  /// Since ghost cells are at the end of the range, this is just
  /// a vector over those cells
  const std::vector<std::uint32_t>& cell_owner() const { return _cell_owner; }

  /// Return connectivity for given pair of topological dimensions
  MeshConnectivity& operator()(std::size_t d0, std::size_t d1)
  {
    dolfin_assert(d0 < _connectivity.size());
    dolfin_assert(d1 < _connectivity[d0].size());
    return _connectivity[d0][d1];
  }

  /// Return connectivity for given pair of topological dimensions
  const MeshConnectivity& operator()(std::size_t d0, std::size_t d1) const
  {
    dolfin_assert(d0 < _connectivity.size());
    dolfin_assert(d1 < _connectivity[d0].size());
    return _connectivity[d0][d1];
  }

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Number of mesh entities for each topological dimension
  std::vector<std::int32_t> _num_entities;

  // Number of ghost indices for each topological dimension (local
  // or global??)
  std::vector<std::size_t> _ghost_offset_index;

  // Global number of mesh entities for each topological dimension
  std::vector<std::int64_t> _global_num_entities;

  // Global indices for mesh entities (empty if not set)
  std::vector<std::vector<std::int64_t>> _global_indices;

  // For entities of a given dimension d, maps each shared entity
  // (local index) to a list of the processes sharing the vertex
  std::map<std::uint32_t, std::map<std::int32_t, std::set<std::uint32_t>>>
      _shared_entities;

  // For cells which are "ghosted", locate the owning process,
  // using a vector rather than a map,
  // since ghost cells are always at the end of the range.
  std::vector<std::uint32_t> _cell_owner;

  // Connectivity for pairs of topological dimensions
  std::vector<std::vector<MeshConnectivity>> _connectivity;
};
}
}