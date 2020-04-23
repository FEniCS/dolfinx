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
#include <dolfinx/common/Memory.h>
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
class FormIntegrals;
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

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities). Note
/// that the mesh entities don't need to be stored, only the number of
/// entities and the connectivity.
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.
///
/// All data beyond the defining data can either be provided at construction
/// time, created via create_XYZ or computed on-the-fly with optional caching.

// TODO: docs on caching
// TODO: shared_ptr<const> or non-const for stored data?
// Currently, everything is set to const (no clear pattern (?) before rework)
// This is in contrast with the previous design, where IndexMap was const but
// the connectivities not. Is there a reason for that? I did not find any place
// in the cpp layer where this was of relevance. This is also why setters were
// removed. See also the comment above the lock in the constructor. Wold it be
// desirable to keep old defining data once it is overwritten?
//
//
class Topology
{

public:
  using Storage = storage::TopologyStorage;

  /// Create mesh topology with prepared data.
  /// Note that everything beyond the data that defines the topology can be
  /// computed on-the-fly with possible caching.
  /// @param[in] comm MPI communicator
  /// @param[in] _remanent_storage storage with at least all essential data.
  /// Essential data are
  ///  * the connectivities for (d0, d1) = (tdim, 0) and (d0, d1) = (0, 0) as
  ///    well as
  ///  * the index maps for dim = tdim and dim = 0.
  /// When more than essential data is provided, the caller is responsible for
  /// correctness. In is safer to only provide the essential data at
  /// construction and create other data, if desired, via "create_XYZ" member
  /// functions.
  /// The copy from remanent_storage is shallow since storage internally work
  /// via shared_ptr<const T>.
  Topology(MPI_Comm comm, mesh::CellType type, const Storage& remanent_storage);
  // TODO: Storage&& instead Probably cleaner since Storage actually cannot be
  // copied? However, we do a "clean" copy afterwards, so from internal side, it
  // does not matter and with const& the caller does not have to give up on his.
  // I hesitate to implement the "clean" copy via the copy constructor
  // because it has some unclear side effects, e.g. difficult to get rid of the
  // memory which is not bound to Locks/handles any longer.

  /// Copy constructor. Loses the cache there currently is no automatic cleanup
  /// once handles are lost.
  Topology(const Topology& topology)
      : Topology{topology.mpi_comm(), topology.cell_type(),
                 topology.remanent_data()} {};

  /// Move constructor
  Topology(Topology&& topology) noexcept = default;

  /// Destructor
  ~Topology() = default;

  /// Assignment
  Topology& operator=(const Topology& topology) = delete;

  /// Assignment
  Topology& operator=(Topology&& topology) noexcept = default;

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
  /// Not that it is cheaper to compute first the connectivity for
  /// (max(d0, d1), min(d0, d1)) if both variants are required (requires cache)
  /// or explicit create_connectivity() calls.
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
  /// and does not discard facet permutations which are computed together
  /// but not as a precondition.
  /// @return The permutation numbers
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
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
  /// and does not discard cell permutations which are computed together
  /// but not as a precondition.
  /// @return The permutation numbers
  std::shared_ptr<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
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
  /// Not that it is cheaper to compute first the connectivity for
  /// (max(d0, d1), min(d0, d1)) if both variants are required.
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  /// @param[in] discard_intermediate results (only store the requested
  // connectivity)
  void create_connectivity(int d0, int d1, bool discard_intermediate = false);

  /// Precompute all entities and connectivity for later use
  void create_connectivity_all();

  /// Precompute entity permutations and reflections for later use
  /// @param[in] discard_intermediate results (only store permutations)
  void create_entity_permutations(bool discard_intermediate = false);

  /// Precompute and set markers for owned facets that are interior
  /// @param[in] discard_intermediate results (only store facet markers)
  void create_interior_facets(bool discard_intermediate = false);

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  /// Enable caching for the lifetime of this lock. If a new layer is forced,
  /// then all new data will be associated to the lifetime of this lock,
  /// i.e., it will shadow any previous cache lock. Howver, data created
  /// explicitly is not affected. The cache only applies to data that is
  /// computed on-the-fly.
  Storage::LayerLock_t acquire_cache_lock(bool force_new_layer = false) const;

  /// Discard all remanent storage except for essential information. Provides a
  /// new layer to add data which can be discarded again. Note that this only
  /// drops ownership but does not necessarily remove data from storage.
  /// Nevertheless, it is still guaranteed that there is no memory overhead.
  void discard_remanent_storage();

  /// Get the data that is either permanent or at least remanent, ie. explicitly
  /// created via the create_XYZ members.
  const Storage& remanent_data() const;

  /// Get the all stored data, that is permanent, remanent and in cache
  const Storage& data() const;

private:
  Topology create_scratch() const;

  static void check_storage(const Storage& remanent_storage, int tdim);

  std::optional<const Storage::LayerLock_t> _remanent_lock;

  // Storage for class invariant (permanent) and remanent (discardable
  // persistent) data
  Storage _remanent_storage;

  // Caching (only when the user acquired a lock)
  mutable Storage _cache;

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

/// Create a toplogy from essential data.
/// @param[in] comm MPI communicator across which the topology is
///   distributed
/// @param[in] cell_type The cell shape
/// @param[in] index_maps_tdim_0 The array keeps the index maps for dim = tdim
/// and dim = 0.
/// @param[in] connectivity_tdim_0 The cell-vertex connectivity,  that is for
/// (tdim, 0).
/// @return A distributed Topology based on given essential data.
/// The array keeps the index maps for dim = tdim and dim = 0.

Topology create_topology(MPI_Comm comm, const CellType& cell_type,
    std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps_tdim_0,
                                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connectivity_tdim_0);

// TODO:What we can't catch is the creation of a dofmap. This could be done in the
// Python layer where ufl element information is available. Probably, this
// function is anyway a convenience for the user of the Python interface.
/// Precomputes topological data based on the types of the integrals provided.
/// Currently, this happens in create_form on a const mesh via topology_mutable.
void create_topological_data(Topology& topology, const fem::FormIntegrals& integrals);
} // namespace mesh
} // namespace dolfinx
