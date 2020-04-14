// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "TopologyStorage.h"
#include "cell_types.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace fem
{
class ElementDofLayout;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{
enum class GhostMode : int;

enum class CellType;

class Topology;

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities). Note
/// that the mesh entities don't need to be stored, only the number of
/// entities and the connectivity.
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.
///

// TODO: docs on caching
// TODO: shared_ptr<const> or non-const for stored data?
// Currently, everything is set to const (no clear pattern (?) before rework)
class Topology
{

public:
  /// Create mesh topology with ready data
  Topology(MPI_Comm comm, mesh::CellType type,
           storage::TopologyStorage remanent_storage)
      : _mpi_comm(comm),
        _cell_type(type), remanent_storage{check_and_save_storage(
                              std::move(remanent_storage), cell_dim(type))},
        cache(&remanent_storage)
  {
    // Acquire lock for explicitly stored data
    remanent_lock = std::make_shared<const storage::StorageLock>(
        remanent_storage.acquire_cache_lock());
  }

  /// Copy constructor
  Topology(const Topology& topology) = default;

  /// Move constructor
  Topology(Topology&& topology) = default;

  /// Destructor
  ~Topology() = default;

  /// Assignment
  Topology& operator=(const Topology& topology) = delete;

  /// Assignment
  Topology& operator=(Topology&& topology) = default;

  /// Return topological dimension
  int dim() const;

  /// Get the IndexMap that described the parallel distribution of the
  /// mesh entities
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] dim Topological dimension
  /// @param[in] discard_intermediate only has an effect in case of caching
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap>
  index_map(int dim, bool discard_intermediate = false) const;

  /// Computes and returns connectivity from entities of dimension d0 to
  /// entities of dimension d1.
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] d0
  /// @param[in] d1
  /// @param[in] discard_intermediate only has an effect in case of caching
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1, bool discard_intermediate = false) const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a
  /// boundary facet. It is not defined for codimension 0.
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  ///   must be less than the topological dimension.
  /// @param[in] discard_intermediate only has an effect in case of caching
  /// @return Vector of length equal to number of local entities, with
  ///   'true' for entities on the boundary and otherwise 'false'.
  std::vector<bool> on_boundary(int dim,
                                bool discard_intermediate = false) const;

  /// Returns the permutation information
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] discard_intermediate only has an effect in case of caching
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>&
  get_cell_permutation_info(bool discard_intermediate = false) const;

  /// Get the permutation number to apply to a facet. The permutations
  /// are numbered so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// Each column of the returned array represents a cell, and each row
  /// a facet of that cell.
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] discard_intermediate only has an effect in case of caching
  /// @return The permutation number
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_facet_permutations(bool discard_intermediate = false) const;

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process.
  /// Results will only be cached if the user has acquired a cache lock before.
  /// @param[in] discard_intermediate only has an effect in case of caching
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  const std::vector<bool>& interior_facets(bool discard_intermediate
                                           = false) const;

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Cell type
  /// @return Cell type that the topology is for
  mesh::CellType cell_type() const;

  // creation of entities
  /// Create entities of given topological dimension for later use
  /// @param[in] dim Topological dimension
  /// @return Number of newly created entities, returns -1 if entities
  ///   already existed
  std::int32_t create_entities(int dim);

  /// Precompute connectivity between given pair of dimensions, d0 -> d1
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(int d0, int d1);

  /// Precompute all entities and connectivity for later use
  void create_connectivity_all();

  /// Precompute entity permutations and reflections for later use
  void create_entity_permutations();

  /// Precompute and set markers for owned facets that are interior
  void create_interior_facets();

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  /// Enable caching for the lifetime of this lock. If a new layer is forced,
  /// then all new data will be associated to the lifetime of this lock,
  /// i.e., it will shadow any previous cache lock. Howver, data created
  /// explicitly is not affected. The cache only applies to data that is
  /// computed on-the-fly.
  storage::StorageLock acquire_cache_lock(bool force_new_layer = false) const {
    return cache.acquire_cache_lock(force_new_layer);
  }

  /// Discard all remanent storage except for essential information. Provides a
  /// new layer to add data which can be discarded again.
  void discard_remanent_storage()
  {
    discardable_remanent_lock = std::make_shared<const storage::StorageLock>(
        remanent_storage.acquire_cache_lock(true));
  }

  const storage::TopologyStorage& remanent_data() const {return remanent_storage;}

  const storage::TopologyStorage& cache_data() const {return cache;}


private:
  storage::TopologyStorage static check_and_save_storage(
      storage::TopologyStorage remanent_storage, int tdim);

  std::shared_ptr<const storage::StorageLock> remanent_lock;
  std::shared_ptr<const storage::StorageLock> discardable_remanent_lock;

  storage::TopologyStorage remanent_storage;
  mutable storage::TopologyStorage cache;

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // Cell type
  mesh::CellType _cell_type;
};

/// Create distributed topology
/// @param[in] comm MPI communicator across which the topology is
///   distributed
/// @param[in] cells The cell topology (list of cell vertices) using
///   global indices for the vertices. It contains cells that have been
///   distributed to this rank, e.g. via a graph partitioner.
/// @param[in] original_cell_index The original global index associated
///   with each cell.
/// @param[in] ghost_owners The ownership of any ghost cells (ghost
///   cells are always at the end of the list of cells, above)
/// @param[in] cell_type The cell shape
/// @param[in] ghost_mode How to partition the cell overlap: none,
/// shared_facet or shared_vertex
/// @return A distributed Topology.
Topology create_topology(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& cells,
                         const std::vector<std::int64_t>& original_cell_index,
                         const std::vector<int>& ghost_owners,
                         const CellType& cell_type, mesh::GhostMode ghost_mode);
} // namespace mesh
} // namespace dolfinx
