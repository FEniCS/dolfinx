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
//
//
// Free function interface of TopologyStorage
// ==========================================
//------------------------------------------------------------------------------
std::int32_t dolfinx::mesh::storage::create_entities(TopologyStorage& storage,
                                                     int dim)
{
  if (auto conn
      = (storage.set_connectivity(storage.connectivity(dim, 0), dim, 0));
      conn)
  {
    storage.set_index_map(storage.index_map(dim), dim);
    storage.set_connectivity(storage.connectivity(storage.dim(), dim),
                             storage.dim(), dim);
    return -1;
  }

  // Create local entities
  const auto [cell_entity, entity_vertex, index_map]
      = TopologyComputation::compute_entities(storage, dim);

  // cell_entitity should not be empty because of the checks above
  assert(cell_entity);
  storage.set_connectivity(cell_entity, storage.dim(), dim);

  // entitiy_ should not be empty because of the checks above
  assert(entity_vertex);
  storage.set_connectivity(entity_vertex, dim, 0);

  // index_map should not be empty because of the checks above
  assert(index_map);
  storage.set_index_map(index_map, dim);

  return index_map->size_local();
}
//--------------------------------------------------------------------------
void dolfinx::mesh::storage::create_connectivity(TopologyStorage& storage,
                                                 int d0, int d1)
{
  // TODO: Replace the weird set(get()) by something more transparent?
  // The purpose is for obtaining ownership...

  // Are the parens required?
  if (auto conn
      = storage.set_connectivity(storage.connectivity(d0, d1), d0, d1);
      conn)
  {
    // Probably it's better to also copy the shared_ptr to index maps
    storage.set_index_map(storage.index_map(d0), d0);
    storage.set_index_map(storage.index_map(d1), d1);
    return;
  }

  // Make sure entities exist
  ::create_entities(storage, d0);
  ::create_entities(storage, d1);

  // Compute connectivity
  const auto [c_d0_d1, c_d1_d0]
      = TopologyComputation::compute_connectivity(storage, d0, d1);

  // NOTE: that to compute the (d0, d1) connections is it sometimes
  // necessary to compute the (d1, d0) connections. We store the (d1,
  // d0) for possible later use, but there is a memory overhead if they
  // are not required. It may be better to not automatically store
  // connectivity that was not requested, but advise in a docstring the
  // most efficient order in which to call this function if several
  // connectivities are needed.

  // Concerning the note above: Provide an overload
  // create_connectivity(std::vector<std::pair<int, int>>)?

  // Attach connectivities
  if (c_d0_d1)
    storage.set_connectivity(c_d0_d1, d0, d1);
  if (c_d1_d0)
    storage.set_connectivity(c_d1_d0, d1, d0);

  // Special facet handing
  if (d0 == (storage.dim() - 1) and d1 == storage.dim())
  {
    create_interior_facets(storage);
  }
}
//-----------------------------------------------------------------------------
void dolfinx::mesh::storage::create_connectivity_all(TopologyStorage& storage)
{
  // Compute all connectivity
  for (int d0 = 0; d0 <= storage.dim(); d0++)
    for (int d1 = 0; d1 <= storage.dim(); d1++)
      ::create_connectivity(storage, d0, d1);
}
//-----------------------------------------------------------------------------
void dolfinx::mesh::storage::create_interior_facets(TopologyStorage& storage)
{
  if (auto facets = storage.set_interior_facets(storage.interior_facets());
      facets)
    return;

  auto f = std::make_shared<std::vector<bool>>(
      TopologyComputation::compute_interior_facets(storage));
  storage.set_interior_facets(f);
}
//-----------------------------------------------------------------------------
void dolfinx::mesh::storage::create_entity_permutations(
    TopologyStorage& storage)
{
  if (auto permutations
      = storage.set_cell_permutations(storage.cell_permutations());
      permutations)
  {
    // Also copy facet permutations to storage
    storage.set_facet_permutations(storage.facet_permutations());
    return;
  }

  const int tdim = storage.dim();

  // FIXME: Is this always required? Could it be made cheaper by doing a
  // local version? This call does quite a lot of parallel work
  // Create all mesh entities
  for (int d = 0; d < tdim; ++d)
    ::create_entities(storage, d);

  auto [facet_permutations, cell_permutations]
      = PermutationComputation::compute_entity_permutations(storage);

  storage.set_facet_permutations(
      std::make_shared<
          const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(
          std::move(facet_permutations)));

  storage.set_cell_permutations(
      std::make_shared<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>(
          std::move(cell_permutations)));
}
//-----------------------------------------------------------------------------
