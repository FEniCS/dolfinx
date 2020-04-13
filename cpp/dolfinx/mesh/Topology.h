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

/// Compute marker for owned facets that are interior, i.e. are
/// connected to two cells, one of which might be on a remote process.
/// @param[in] topology The topology.
/// @return Vector with length equal to the number of facets on this
///   this process. True if the ith facet (local index) is interior to
///   the domain.
std::vector<bool> compute_interior_facets(const Topology& topology);

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities). Note
/// that the mesh entities don't need to be stored, only the number of
/// entities and the connectivity.
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.
class Topology
{
public:
  /// Create empty mesh topology
  Topology(MPI_Comm comm, mesh::CellType type)
      : _mpi_comm(comm), _cell_type(type)
  {
    // Do nothing
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

  /// @todo Merge with set_connectivity
  ///
  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  /// Get the IndexMap that described the parallel distribution of the
  /// mesh entities
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a
  /// boundary facet. It is not defined for codimension 0.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  ///   must be less than the topological dimension.
  /// @return Vector of length equal to number of local entities, with
  ///   'true' for entities on the boundary and otherwise 'false'.
  std::vector<bool> on_boundary(int dim) const;

  /// Return connectivity from entities of dimension d0 to entities of
  /// dimension d1
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// @todo Merge with set_index_map
  /// Set connectivity for given pair of topological dimensions
  void set_connectivity(std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// Returns the permutation information
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>&
  get_cell_permutation_info() const;

  /// Get the permutation number to apply to a facet. The permutations
  /// are numbered so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// Each column of the returned array represents a cell, and each row
  /// a facet of that cell.
  /// @return The permutation number
  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
  get_facet_permutations() const;

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  const std::vector<bool>& interior_facets() const;

  /// Set markers for owned facets that are interior
  /// @param[in] interior_facets The marker vector
  void set_interior_facets(const std::vector<bool>& interior_facets);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Cell type
  /// @return Cell type that the topology is for
  mesh::CellType cell_type() const;

  // TODO: Rework memory management and associated API
  // Currently, there is no clear caching policy implemented and no way of
  // discarding cached data.

  // creation of entities
  /// Create entities of given topological dimension.
  /// @param[in] dim Topological dimension
  /// @return Number of newly created entities, returns -1 if entities
  ///   already existed
  std::int32_t create_entities(int dim);

  /// Create connectivity between given pair of dimensions, d0 -> d1
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(int d0, int d1);

  /// Compute entity permutations and reflections
  void create_entity_permutations();

  /// Compute all entities and connectivity
  void create_connectivity_all();

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  StorageLock acquire_cache_lock(bool force_new_layer = false) const
  {
    remove_expired_layers();
    if (cache.empty() or force_new_layer)
    {
      auto storage_layer_ptr = std::make_shared<TopologyStorageLayer>();
      cache.emplace_back(storage_layer_ptr);
      return StorageLock{storage_layer_ptr};
    }
    else
    {
      return StorageLock{cache.back().lock()};
    }
  }

private:

  template <typename func_t,
            std::enable_if_t<std::is_same_v<
                std::shared_ptr<typename func_t::return_type::element_type>,
                func_t::return_type>> = 0>
  typename func_t::return_type read_fom_storage(func_t read) const
  {
    // Search in remanent storage
    if (auto value = read(remanent_storage); value)
      return value;

    // Search in cache
    remove_expired_layers();
    for (const auto& w_layer : cache)
    {
      if (auto value = read(*w_layer.lock()); value)
        return value;
    }
    // Return default constructed (empty) object
    return {};
  }

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t retrieve_fom_storage(func_t read) const
  {
    // Search in remanent storage
    if (auto value = read(remanent_storage); value)
      return value;

    // Search in cache
    remove_expired_layers();
    for (const auto& w_layer : cache)
    {
      if (auto value = read(*w_layer.lock()); value)
        return value;
    }
    // Return default constructed (empty) object
    return {};
  }

  template <typename value_t,
      // Currently only for shared_ptr
      typename ret_t = std::shared_ptr<value_t>,
      typename func_t
      = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t retrieve_fom_storage(
      func_t read_func,
      std::function<void(TopologyStorageLayer&)> create_func, bool compute=true) const
  {
    // Acquire lock for ensuring some interal storage
    auto _lock = acquire_cache_lock();
    // Retrieve active layer
    auto active_layer = *cache.back().lock();

    if (compute)
      create_func(active_layer);

    return retrieve_fom_storage(read_func);
  }

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t compute(func_t compute_func) const
  {
    // Acquire lock for having interal storage
    auto _lock = acquire_cache_lock();
    // Retrieve active layer
    auto active_layer = *cache.back().lock();

    return compute_func(active_layer);
  }

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t copy_to_layer(func_t read_func, TopologyStorageLayer& target) const
  {
    if (auto value = (read_func(target) = retrieve_fom_storage(read_func));
        value)
    {
      return value;
    }
    // Return empty
    ret_t{};
  }

  std::shared_ptr<const std::vector<bool>> _interior_facets(bool compute
                                                            = false) const;

  TopologyStorageLayer remanent_storage;
  mutable std::list<std::weak_ptr<TopologyStorageLayer>> cache;

  void remove_expired_layers() const
  {
    cache.remove_if([](const std::weak_ptr<TopologyStorageLayer>& layer) {
      layer.expired();
    });
  }

  /// Create entities of given topological dimension in given storage.
  /// Works around constness by separating "Storage" from the owning "Topology".
  /// @param[in,out] storage Object where to store the created entities
  /// @param[in] dim Topological dimension
  /// @return Number of newly created entities, returns -1 if entities
  ///   already existed
  std::int32_t create_entities(TopologyStorageLayer& storage, int dim) const;

  /// Create connectivity between given pair of dimensions, d0 -> d1 in given
  /// storage
  /// @param[in,out] storage Object where to store the created entities
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(TopologyStorageLayer& storage, int d0, int d1) const;

  /// Set markers for owned facets that are interior in given storage
  /// @param[in,out] storage Object where to store the created entities
  /// @param[in] interior_facets The marker vector
  std::shared_ptr<const std::vector<bool>> set_interior_facets(TopologyStorageLayer& storage,
                           const std::vector<bool>& interior_facets) const;

  /// Set connectivity for given pair of topological dimensions in given storage
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> set_connectivity(TopologyStorageLayer& storage,
                        std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1) const;

  /// Set connectivity for given pair of topological dimensions in given storage
  std::shared_ptr<const common::IndexMap> set_index_map(TopologyStorageLayer& storage,
                     std::shared_ptr<const common::IndexMap> index_map, int dim) const;

  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  set_cell_permutations(TopologyStorageLayer& storage, std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>) const;

  std::shared_ptr<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
  set_facet_permutations(TopologyStorageLayer& storage, std::shared_ptr<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>) const;

  /// Sets connectivity for all entities and connectivity in given storage
  /// @param[in,out] storage Object where to store the created entities
  void create_connectivity_all(TopologyStorageLayer& storage) const;

  /// Set markers for owned facets that are interior in given storage
  /// @param[in,out] storage Object where to store the created entities
  /// @param[in] interior_facets The marker vector
  /// Compute entity permutations and reflections in given storage
  void create_entity_permutations(TopologyStorageLayer& storage) const;

  /// Computes and set markers for owned facets that are interior
  void create_interior_facets(TopologyStorageLayer& storage) const;

  typename decltype(TopologyStorageLayer::connectivity)::Scalar
  _connectivity(int d0, int d1, bool compute = false) const;
  decltype(remanent_storage.cell_permutations)
  _cell_permutations(bool compute = false) const;
  decltype(remanent_storage.facet_permutations)
  _facet_permutations(bool compute = false) const;

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
