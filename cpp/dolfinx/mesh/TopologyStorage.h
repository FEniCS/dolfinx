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

namespace mesh
{
class Topology;
enum class CellType;
} // namespace mesh

} // namespace dolfinx

namespace dolfinx::mesh::storage
{

namespace internal
{

using lock_t = std::shared_ptr<const bool>;
using sentinel_t = std::weak_ptr<const bool>;

template <typename T>
using guarded_obj = std::pair<T, internal::sentinel_t>;

template <typename T>
guarded_obj<T> make_guarded(T obj, internal::sentinel_t sentinel)
{
  return std::make_pair<typename guarded_obj<T>::first_type,
                        typename guarded_obj<T>::second_type>(
      std::forward<T&&>(obj), std::move(sentinel));
}

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

/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
set_index_map(TopologyStorageLayer& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
set_cell_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
set_facet_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations);

std::shared_ptr<const std::vector<bool>>
set_interior_facets(TopologyStorageLayer& storage,
                    std::shared_ptr<const std::vector<bool>> interior_facets);

std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
connectivity(TopologyStorageLayer& storage, int d0, int d1);

std::shared_ptr<const common::IndexMap> index_map(TopologyStorageLayer& storage,
                                                  int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
cell_permutations(TopologyStorageLayer& storage);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
facet_permutations(TopologyStorageLayer& storage);

using storage_layers_t = std::list<guarded_obj<internal::TopologyStorageLayer>>;

template <typename value_t,
          // Currently only for shared_ptr
          typename ret_t = std::shared_ptr<value_t>,
          typename func_t
          = std::function<std::shared_ptr<value_t>>(TopologyStorageLayer&)>
ret_t read_from_layers(func_t read, const storage_layers_t& layers)
{
  for (const auto& w_layer : layers)
  {
    assert(w_layer.second.lock());
    if (auto value = read(w_layer.first); value)
      return value;
  }
  return ret_t{};
}

} // namespace internal

class TopologyStorage;

class StorageLock
{

public:
  StorageLock(internal::lock_t lock,
              internal::guarded_obj<TopologyStorage*> storage)
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
  internal::lock_t lock;
  internal::guarded_obj<TopologyStorage*> storage;
};


// TODO: Make lock sequential. This can be done in two ways: See acquire_lock...
class TopologyStorage
{

public:
  using StorageLayer = internal::TopologyStorageLayer;

  /// Create empty mesh topology
  explicit TopologyStorage(bool remanent, const TopologyStorage* other = nullptr)
      : _other_storage{make_other(other)}
  {
    if (remanent)
      make_remanent(true);
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

  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  std::shared_ptr<const common::IndexMap>
  set_index_map(std::shared_ptr<const common::IndexMap> index_map, int dim);

  /// Get the IndexMap that described the parallel distribution of the
  /// mesh entities
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Return connectivity from entities of dimension d0 to entities of
  /// dimension d1.
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  ///   gives the list of incident entities of dimension d1
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// Set connectivity for given pair of topological dimensions
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  set_connectivity(std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                   int d0, int d1);

  /// Returns the permutation information
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
  cell_permutations() const;

  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
      set_cell_permutations(
          std::shared_ptr<
              const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>);

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

  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
      set_facet_permutations(
          std::shared_ptr<const Eigen::Array<std::uint8_t, Eigen::Dynamic,
                                             Eigen::Dynamic>>);

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  std::shared_ptr<const std::vector<bool>> interior_facets() const;

  /// Set markers for owned facets that are interior
  /// @param[in] interior_facets The marker vector
  std::shared_ptr<const std::vector<bool>>
  set_interior_facets(std::shared_ptr<const std::vector<bool>> interior_facets);

  /// Provide another storage instance for read-only access.
  /// Lifetime is tracked of "other" is tracked. "other*" can be nullptr.
  void set_other(const TopologyStorage* other)
  {
    if (other)
      _other_storage = {other, other->lifetime};
    else
      _other_storage = {other, std::weak_ptr<const bool>{}};
  }

  /// Get the read-only instance.
  internal::guarded_obj<const TopologyStorage*> other() const
  {
    return _other_storage;
  }

  /// Drop the pointer on the read-only storage.
  internal::guarded_obj<const TopologyStorage*> drop_other()
  {
    internal::guarded_obj<const TopologyStorage*> ret;
    std::swap(ret, _other_storage);
    return ret;
  }

  // TODO: Make lock sequential. This can be done in two ways:
  // a) A lock must not be copied and moved. Then it cannot leave its scope.
  // b) Store a vector of locks and at the end of life, remove all new ones.
  // This should be optional since it increases storage (like a back button)
  /// Create a new lock. By default, it just is another handle for the current
  /// layer. The creation of  new write layer is optional.
  /// Note that it will share ownership with data from other layers. This means,
  /// it can also overwrite their data. Otherwise, the ownership would be to
  /// difficult to track, since the lifetimes of the locks do not end in any
  /// defined order.
  /// Let locks allow read access to "their" layer?
  StorageLock acquire_cache_lock(bool force_new_layer = false)
  {
    if (layers.empty() or force_new_layer)
    {
      auto layer_lock = std::make_shared<const bool>();
      internal::sentinel_t layer_sentinel = layer_lock;

      if (layers.empty())
        layers.emplace_front(StorageLayer{}, layer_sentinel);
      // Copy the last active layer for sharing ownership
      else
        layers.emplace_front(layers.front().first, layer_sentinel);

      return StorageLock(layer_lock, {this, lifetime});
    }
    else
    {
      return StorageLock(active_layer().second.lock(), {this, lifetime});
    }
  }

  void remove_expired_layers()
  {
    layers.remove_if([](const internal::guarded_obj<StorageLayer>& layer) {
      return layer.second.expired();
    });
  }

  std::size_t number_of_layers() const { return layers.size(); }

  void make_remanent(bool all_data)
  {
    remanent_lock = std::make_shared<const storage::StorageLock>(
        acquire_cache_lock(true));
    if (all_data)
    {
      for (int d0 = 0; d0 < active_layer().first.connectivity.rows(); ++d0)
      {
        set_index_map(index_map(d0), d0);
        for (int d1 = 0; d1 < active_layer().first.connectivity.cols(); ++d0)
        {
          set_connectivity(connectivity(d0, d1), d0, d1);
        }
      }
      set_interior_facets(interior_facets());
      set_facet_permutations(facet_permutations());
      set_cell_permutations(cell_permutations());
    }
  }

  /// The active layer
  internal::guarded_obj<StorageLayer>& active_layer() {
    return layers.front();
  }

  /// Discard is only guaranteed for what has been written since
  /// "make_remanent" and only if no new locks have been acquired.
  void discard() { remanent_lock.reset(); }

  /// Creates storage on top (ie. with read access) of this. Note that the
  /// it is still in valid state if the original instance has gone. The new
  /// instance however, does not have any ownership on the original data by
  /// default.
  /// @param[in] share_ownership whether ownership shall be shared. This then
  /// also automatically creates a writable layer.
  TopologyStorage create_on_top(bool share_ownership = false) const
  {
    TopologyStorage res(this);
    if (share_ownership)
      res.make_remanent(true);
    return std::move(res);
  }

private:

  static internal::guarded_obj<const TopologyStorage*>
  make_other(const TopologyStorage* other)
  {
    if (other)
      return internal::make_guarded(other, other->lifetime);
    else
      return internal::make_guarded(other, internal::sentinel_t{});
  }


  template <typename value_t,
            // Currently only for shared_ptr
            typename ret_t = std::shared_ptr<value_t>,
            typename func_t
            = std::function<std::shared_ptr<value_t>>(StorageLayer&)>
  ret_t read_from_storage(func_t read) const
  {
    // Search in owned storage
    if (auto value = internal::read_from_layers<value_t>(read, layers); value)
      return value;

    // Search in not owned storage. Assume that if first condition is not met,
    // the second is not evalutated. Otherwise, there can be a problem with
    // accessing "first".
    if ((!_other_storage.second.expired()) && _other_storage.first)
    {
      if (auto value = internal::read_from_layers<value_t>(
            read, _other_storage.first->layers);
          value)
        return value;
    }

    // Nothing found. Return empty pointer.
    return {};
  }

  std::shared_ptr<const storage::StorageLock> remanent_lock;

  // The other (already existing storage) with read access
  internal::guarded_obj<const TopologyStorage*> _other_storage;

  // The storage layers
  internal::storage_layers_t layers;

  internal::lock_t lifetime{std::make_shared<const bool>(true)};
};

} // namespace dolfinx::mesh::storage
