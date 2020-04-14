//
// Created by mrambausek on 12.04.20.
//

#include "TopologyStorage.h"
#include "Partitioning.h"
#include "PermutationComputation.h"
#include "TopologyComputation.h"

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::mesh;
using namespace dolfinx::mesh::storage;

//
namespace dolfinx::mesh::storage::internal
{
// TopologyStorageLayer free functions
// ===================================
//------------------------------------------------------------------------------
//
std::shared_ptr<const std::vector<bool>>
set_interior_facets(TopologyStorageLayer& storage,
                    const std::vector<bool>& interior_facets)
{
  return storage.interior_facets
         = std::make_shared<const std::vector<bool>>(interior_facets);
}
//------------------------------------------------------------------------------
/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
set_connectivity(TopologyStorageLayer& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1)
{
  assert(d0 < storage.connectivity.rows());
  assert(d1 < storage.connectivity.cols());
  return storage.connectivity(d0, d1) = c;
}
//------------------------------------------------------------------------------
/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
set_index_map(TopologyStorageLayer& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim)
{
  assert(dim < static_cast<int>(storage.index_map.size()));
  return storage.index_map[dim] = index_map;
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
set_cell_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations)
{
  return storage.cell_permutations = cell_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
set_facet_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations)
{
  return storage.facet_permutations = facet_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
interior_facets(const TopologyStorageLayer& storage)
{
  return storage.interior_facets;
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
connectivity(const TopologyStorageLayer& storage, int d0, int d1)
{
  assert(d0 < storage.connectivity.rows());
  assert(d1 < storage.connectivity.cols());
  return storage.connectivity(d0, d1);
}
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
index_map(const TopologyStorageLayer& storage, int dim)
{
  assert(dim < static_cast<int>(storage.index_map.size()));
  return storage.index_map[dim];
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
cell_permutations(const TopologyStorageLayer& storage)
{
  return storage.cell_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
facet_permutations(const TopologyStorageLayer& storage)
{
  return storage.facet_permutations;
}
//------------------------------------------------------------------------------
} // namespace dolfinx::mesh::storage::internal
//------------------------------------------------------------------------------
//
//
// StorageLock
// ========
//------------------------------------------------------------------------------
StorageLock::~StorageLock()
{
  if (!storage.second.expired())
    storage.first->remove_expired_layers();
}
//------------------------------------------------------------------------------
//
//
// TopologyStorage
// ===============
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> TopologyStorage::set_index_map(
    std::shared_ptr<const common::IndexMap> index_map, int dim)
{
  if (layers.empty())
    throw std::runtime_error(
        "No storage layer present. Acquire a cache lock before writing.");
  return internal::set_index_map(layers.back().first, index_map, dim);
}
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
TopologyStorage::index_map(int dim) const
{
  using func_t = std::function<std::shared_ptr<const common::IndexMap>(
      const StorageLayer&)>;
  func_t read = [=](const StorageLayer& storage) {
    return internal::index_map(storage, dim);
  };
  return read_from_storage<func_t::result_type::element_type>(std::move(read));
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
TopologyStorage::connectivity(int d0, int d1) const
{
  using func_t
      = std::function<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>(
          const StorageLayer&)>;
  func_t read = [=](const StorageLayer& storage) {
    return internal::connectivity(storage, d0, d1);
  };
  return read_from_storage<func_t::result_type::element_type>(std::move(read));
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
TopologyStorage::set_connectivity(
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  if (layers.empty())
    throw std::runtime_error(
        "No storage layer present. Acquire a cache lock before writing.");
  return internal::set_connectivity(layers.back().first, c, d0, d1);
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
TopologyStorage::cell_permutations() const
{
  using func_t = std::function<
      std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>(
          const StorageLayer&)>;
  func_t read = [=](const StorageLayer& storage) {
    return internal::cell_permutations(storage);
  };
  return read_from_storage<func_t::result_type::element_type>(std::move(read));
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
TopologyStorage::set_cell_permutations(
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations)
{
  if (layers.empty())
    throw std::runtime_error(
        "No storage layer present. Acquire a cache lock before writing.");
  return internal::set_cell_permutations(layers.back().first,
                                         cell_permutations);
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
TopologyStorage::facet_permutations() const
{
  using func_t = std::function<std::shared_ptr<const Eigen::Array<
      std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(const StorageLayer&)>;
  func_t read = [=](const StorageLayer& storage) {
    return internal::facet_permutations(storage);
  };
  return read_from_storage<func_t::result_type::element_type>(std::move(read));
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
TopologyStorage::set_facet_permutations(
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations)
{
  if (layers.empty())
    throw std::runtime_error(
        "No storage layer present. Acquire a cache lock before writing.");
  return internal::set_facet_permutations(layers.back().first,
                                          facet_permutations);
}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
TopologyStorage::interior_facets() const
{
  using func_t = std::function<std::shared_ptr<const std::vector<bool>>(
      const StorageLayer&)>;
  func_t read = [=](const StorageLayer& storage) {
    return internal::interior_facets(storage);
  };
  return read_from_storage<func_t::result_type::element_type>(std::move(read));
}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>> TopologyStorage::set_interior_facets(
    std::shared_ptr<const std::vector<bool>> interior_facets)
{
  if (layers.empty())
    throw std::runtime_error(
        "No storage layer present. Acquire a cache lock before writing.");
  return internal::set_interior_facets(layers.back().first, interior_facets);
}
//------------------------------------------------------------------------------
