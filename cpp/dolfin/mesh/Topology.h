// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace dolfin
{
namespace mesh
{

class Connectivity;

/// Topology stores the topology of a mesh, consisting of mesh
/// entities and connectivity (incidence relations for the mesh
/// entities). Note that the mesh entities don't need to be stored,
/// only the number of entities and the connectivity. Any numbering
/// scheme for the mesh entities is stored separately in a
/// MeshFunction over the entities.
///
/// A mesh entity e may be identified globally as a pair e = (dim,
/// i), where dim is the topological dimension and i is the index of
/// the entity within that topological dimension.

class Topology
{
public:
  /// Create empty mesh topology
  /// @param dim
  ///   Topological dimension
  Topology(std::size_t dim, std::int32_t num_vertices,
           std::int64_t num_vertices_global);

  /// Copy constructor
  Topology(const Topology& topology) = default;

  /// Move constructor
  Topology(Topology&& topology) = default;

  /// Destructor
  ~Topology() = default;

  /// Assignment
  Topology& operator=(const Topology& topology) = default;

  /// Return topological dimension
  int dim() const;

  /// Return number of entities for given dimension (local to process)
  std::int32_t size(int dim) const;

  /// Return global number of entities for given dimension
  std::int64_t size_global(int dim) const;

  /// Return number of regular (non-ghost) entities or equivalently,
  /// the offset of where ghost entities begin
  std::int32_t ghost_offset(int dim) const;

  /// Clear data for given pair of topological dimensions
  void clear(int d0, int d1);

  /// Set number of global entities (global_size) for given topological
  /// dimension dim
  void set_num_entities_global(int dim, std::int64_t global_size);

  // Set the global indices for entities of dimension dim
  void set_global_indices(int dim,
                          const std::vector<std::int64_t>& global_indices);

  /// Initialise the offset index of ghost entities for this dimension
  void init_ghost(std::size_t dim, std::size_t index);

  /// Get local-to-global index map for entities of topological
  /// dimension d
  const std::vector<std::int64_t>& global_indices(std::size_t d) const;

  /// Check if global indices are available for entities of
  /// dimension dim
  bool have_global_indices(std::size_t dim) const;

  /// Check whether there are any shared entities calculated of
  /// dimension dim
  bool have_shared_entities(int dim) const;

  /// Return map from shared entities (local index) to processes
  /// that share the entity
  std::map<std::int32_t, std::set<std::int32_t>>& shared_entities(int dim);

  /// Return map from shared entities (local index) to process that
  /// share the entity (const version)
  const std::map<std::int32_t, std::set<std::int32_t>>&
  shared_entities(int dim) const;

  /// Return mapping from local ghost cell index to owning process Since
  /// ghost cells are at the end of the range, this is just a vector
  /// over those cells
  std::vector<std::int32_t>& cell_owner();

  /// Return mapping from local ghost cell index to owning process
  /// (const version). Since ghost cells are at the end of the range,
  /// this is just a vector over those cells
  const std::vector<std::int32_t>& cell_owner() const;

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<Connectivity> connectivity(std::size_t d0, std::size_t d1);

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<const Connectivity> connectivity(std::size_t d0,
                                                   std::size_t d1) const;

  /// Set connectivity for given pair of topological dimensions
  void set_connectivity(std::shared_ptr<Connectivity> c, std::size_t d0,
                        std::size_t d1);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Number of mesh vertices
  const std::int32_t _num_vertices;

  // Number of ghost indices for each topological dimension (local
  // or global??)
  std::vector<std::size_t> _ghost_offset_index;

  // Global number of mesh entities for each topological dimension
  std::vector<std::int64_t> _global_num_entities;

  // Global indices for mesh entities (empty if not set)
  std::vector<std::vector<std::int64_t>> _global_indices;

  // TODO: Could IndexMap be used here in place of std::map?
  // For entities of a given dimension d, maps each shared entity
  // (local index) to a list of the processes sharing the vertex
  std::map<std::int32_t, std::map<std::int32_t, std::set<std::int32_t>>>
      _shared_entities;

  // TODO: Could IndexMap be used here
  // For cells which are "ghosted", locate the owning process, using a
  // vector rather than a map, since ghost cells are always at the end
  // of the range.
  std::vector<std::int32_t> _cell_owner;

  // Connectivity for pairs of topological dimensions
  std::vector<std::vector<std::shared_ptr<Connectivity>>> _connectivity;
}; // namespace mesh
} // namespace mesh
} // namespace dolfin
