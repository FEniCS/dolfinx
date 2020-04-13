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
  Eigen::Array<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>,
               3, 3, Eigen::RowMajor>
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


class StorageLock
{

public:
  StorageLock(std::shared_ptr<TopologyStorageLayer> lock)
      : lock{std::move(lock)}
  {
    // do nothing
  }

  StorageLock() = delete;
  StorageLock(const StorageLock&) = default;
  StorageLock(StorageLock&&) = default;

  StorageLock& operator=(const StorageLock&) = default;
  StorageLock& operator=(StorageLock&&) = default;

  ~StorageLock() = default;

private:
  std::shared_ptr<TopologyStorageLayer> lock;
};

class TopologyStorage
{
public:
  TopologyStorage() = default;

  TopologyStorage(const TopologyStorage& other) = delete;
  TopologyStorage(TopologyStorage&& other) = default;

  TopologyStorage& operator=(const TopologyStorage& other) = delete;
  TopologyStorage& operator=(TopologyStorage&& other) = default;

  ~TopologyStorage() = default;

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
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// Get the IndexMap that described the parallel distribution of the
  /// mesh entities
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @todo Merge with set_connectivity
  ///
  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  // TODO: make naming consistent with facet permutations
  /// Returns the permutation information
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  get_cell_permutation_info() const;

  /// Sets the cell permutation information
  void set_cell_permutations(
      Eigen::Array<std::uint32_t, Eigen::Dynamic, 1> cell_permutations);

  /// Get the permutation number to apply to a facet. The permutations
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
  get_facet_permutations() const;

  // TODO: how to ref another function in doc?
  /// Set the permutation number to apply to a facet. See
  /// get_facet_permutations() for numbering
  void set_facet_permutations(
      Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
          facet_permutations);

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  std::shared_ptr<const std::vector<bool>> interior_facets() const;

  /// Sets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @param[in] Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  void set_interior_facets(std::vector<bool> interior_facets);

  /// Acquire a storage lock. As long as it is alive, the associated storage
  /// (cache) layer will be as well.
  /// @param[in] force_new_layer A new storage layer is created and will be
  /// written to instead of already existing layers. Thus, this may interfere
  /// with caching requested at another place and should thus be handle with
  /// care. Hence, this option is deactivated by default, A possible application
  /// is to enforce removal of data stored for operations in a limited scope.
  StorageLock lock(bool force_new_layer = false) const
  {
    remove_expired_layers();
    if (storage_layers.empty() or force_new_layer)
    {
      TopologyStorageLayer storage_layer{};
      auto storage_lock_ptr = std::make_shared<const bool>(true);
      std::weak_ptr<const bool> storage_sentinel = storage_lock_ptr;
      storage_layers.emplace_back(storage_layer, storage_sentinel);
      return StorageLock{storage_lock_ptr, sentinel, this}
                         ;
    }
    else
    {
      return StorageLock{storage_layers.back().second.lock(), sentinel, this};
    }
  }

  // Obviously not const. This just
  void remove_expired_layers() const
  {
    storage_layers.remove_if(
        [](const std::pair<TopologyStorageLayer, std::weak_ptr<const bool>>&
               pair) { pair.second.expired(); });
  }

private:
  TopologyStorageLayer& active_layer()
  {
    assert(!storage_layers.empty());
    return storage_layers.back().first;
  }

  // inform the locks whether they lock something that is alive.
  std::shared_ptr<const bool> sentinel{std::make_shared<const bool>(true)};

  // The storage together with a sentinel.
  // Must be mutable if more than one layer should be supported.
  // Otherwise things are a bit tricky and probably also not very transparent
  mutable std::list<std::pair<TopologyStorageLayer, std::weak_ptr<const bool>>>
      storage_layers{};
};

} // namespace dolfinx::mesh