//
// Created by mrambausek on 12.04.20.
//

#pragma once

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

namespace graph
{
template <typename T>
class AdjacencyList;
}
} // namespace dolfinx

namespace dolfinx::mesh
{

struct TopologyStorageLayer
{

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>, 3, 3,
               Eigen::RowMajor>
      connectivity;

  // The facet permutations
  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
      facet_permutations;

  // Cell permutation info. See the documentation for
  // get_cell_permutation_info for documentation of how this is encoded.
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
      cell_permutations;

  // Marker for owned facets, which evaluates to True for facets that
  // are interior to the domain
  std::shared_ptr<const std::vector<bool>> interior_facets;
};

/// Set markers for owned facets that are interior in given storage
/// @param[in,out] storage Object where to store the created entities
/// @param[in] interior_facets The marker vector
std::shared_ptr<const std::vector<bool>>
set_interior_facets(TopologyStorageLayer& storage,
                    const std::vector<bool>& interior_facets);

/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
set_connectivity(TopologyStorageLayer& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1);

/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const common::IndexMap>
set_index_map(TopologyStorageLayer& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
set_cell_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
set_facet_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>);


template <typename T>
using guarded_obj = std::pair<T, std::weak_ptr<const bool>>;

using lock_t = std::shared_ptr<const bool>;
using sentinel_t = std::weak_ptr<const bool>;

using storage_layers_t = std::list<guarded_obj<TopologyStorageLayer>>;


template <typename value_t,
    // Currently only for shared_ptr
    typename ret_t = std::shared_ptr<value_t>,
    typename func_t
    = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
ret_t read_from_layers(func_t read, const storage_layers_t& layers) {
  for (const auto& w_layer : layers)
  {
    assert(w_layer.second.lock());
    if (auto value = read(w_layer.first); value)
      return value;
  }
  return ret_t{};
}


class TopologyStorage;

class StorageLock
{

public:
  StorageLock(lock_t lock,
              guarded_obj<TopologyStorage*> storage)
      : lock{std::move(lock)}, storage{std::move(storage)}
  {
    // do nothing
  }

  StorageLock() = delete;
  StorageLock(const StorageLock&) = default;
  StorageLock(StorageLock&&) = default;

  StorageLock& operator=(const StorageLock&) = default;
  StorageLock& operator=(StorageLock&&) = default;

  ~StorageLock();

private:
  lock_t lock;
  guarded_obj<TopologyStorage*> storage;
};

class Topology;

class TopologyStorage
{
public:
  /// Create empty mesh topology
  TopologyStorage(MPI_Comm comm, std::weak_ptr<const Topology> owner,
                  guarded_obj<const TopologyStorage*> other_storage
                  = {nullptr, std::weak_ptr<const bool>{}})
      : _mpi_comm{comm}, _owner{owner}, _other_storage{other_storage}
  {
    // Do nothing
  }

  /// Copy constructor
  TopologyStorage(const TopologyStorage& topology) = default;

  /// Move constructor
  TopologyStorage(TopologyStorage&& topology) = default;

  /// Destructor
  ~TopologyStorage() = default;

  /// Assignment
  TopologyStorage& operator=(const TopologyStorage& topology) = delete;

  /// Assignment
  TopologyStorage& operator=(TopologyStorage&& topology) = default;

  /// The owning topology
  std::weak_ptr<const Topology> owner() const;

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
  /// dimension d1.
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// Computes and returns connectivity from entities of dimension d0 to
  /// entities of dimension d1.
  /// The non-const version allows caching if a cache is present.
  /// By default, all intermediate results are cached as well.
  /// @param[in] d0
  /// @param[in] d1
  /// @param[in] discard_intermediate
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  compute_connectivity(int d0, int d1, bool discard_intermediate = false);

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  compute_connectivity(int d0, int d1) const;

  /// @todo Merge with set_index_map
  /// Set connectivity for given pair of topological dimensions
  void
  set_connectivity(std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                   int d0, int d1);

  /// Returns the permutation information
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  cell_permutations() const;

  /// Computes and returns the permutation information.
  /// The non-const version allows caching if a cache is present.
  /// By default, all intermediate results are cached as well.
  /// @param[in] discard_intermediate
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  compute_cell_permutations(bool discard_intermediate = false);

  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  compute_cell_permutations() const;

  /// Get the permutations numbers to apply to a facet. The permutations
  /// are numbered so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// Each column of the returned array represents a cell, and each row
  /// a facet of that cell.
  /// @return The permutation number
  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
  facet_permutations() const;

  /// Computes the permutation numbers to apply to a facet.
  /// See the documentation of 'facet_permutations' for details.
  /// The non-const version allows caching if a cache is present.
  /// By default, all intermediate results are cached as well.
  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
  compute_facet_permutations(bool discard_intermediate = false);

  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
  compute_facet_permutations() const;

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  std::shared_ptr<const std::vector<bool>> interior_facets() const;

  /// Computes the markers for owned facets that are interior.
  /// The non-const version allows caching if a cache is present.
  /// By default, all intermediate results are cached as well.
  std::shared_ptr<const std::vector<bool>>
  compute_interior_facets(bool discard_intermediate = false);

  std::shared_ptr<const std::vector<bool>> compute_interior_facets();
  const

      /// Set markers for owned facets that are interior
      /// @param[in] interior_facets The marker vector
      void
      set_interior_facets(const std::vector<bool>& interior_facets);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

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

  /// Compute all entities and connectivity
  void create_connectivity_all();

  /// Compute entity permutations and reflections
  void create_entity_permutations();

  /// Compute entity permutations and reflections
  void create_interior_facets();

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  StorageLock acquire_cache_lock(bool force_new_layer = false)
  {
    if (layers.empty() or force_new_layer)
    {
      auto layer_lock = std::make_shared<const bool>();
      sentinel_t layer_sentinel = layer_lock;
      layers.emplace_back(TopologyStorageLayer{}, layer_sentinel);
      return StorageLock{layer_lock, {this, lifetime}};
    }
    else
    {
      return StorageLock{layers.back().second.lock(), {this, lifetime}};
    }
  }

  void remove_expired_layers()
  {
    layers.remove_if([](const std::weak_ptr<TopologyStorageLayer>& layer) {
      layer.expired();
    });
  }

private:

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t read_fom_storage(func_t read) const
  {
    // Search in not owned storage
    if (auto value = read_from_layers(read, _other_storage.first->layers); value)
        return value;

      // Search in owned storage
    else read_from_layers(read, layers);
  }

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
  ret_t compute(
      func_t read_func,
      std::function<std::shared_ptr<value_t>(TopologyStorageLayer&)> compute_func)
  {
    if (auto stored = read_fom_storage(read_func); stored)
      return stored;

    // Acquire lock for ensuring some internal storage
    auto _lock = acquire_cache_lock(true);
    return compute_func();
  }

  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(TopologyStorage&)>
  ret_t compute(func_t compute_func) const
  {
    TopologyStorage tmp_storage{mpi_comm(), _owner, {this, lifetime}};
    return compute_func(tmp_storage);
  }

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // The owning topology
  std::weak_ptr<const Topology> _owner;

  // The other (already existing storage) with read access
  guarded_obj<const TopologyStorage*> _other_storage;

  // The storage layers
  storage_layers_t layers;

  lock_t lifetime{std::make_shared<const bool>(true)};
};


} // namespace dolfinx::mesh
