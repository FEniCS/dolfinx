//
// Created by mrambausek on 12.04.20.
//

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Memory.h>
#include <memory>
#include <shared_mutex>
#include <stack>
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

struct TopologyStorageLayer
{

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>, 4, 4,
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
std::shared_ptr<const std::vector<bool>>
set_interior_facets(TopologyStorageLayer& storage,
                    std::shared_ptr<const std::vector<bool>> interior_facets);

/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
set_connectivity(TopologyStorageLayer& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1);

/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
set_index_map(TopologyStorageLayer& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim);

/// Set cell permutation information
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
set_cell_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations);

/// Set facet permutation information
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
set_facet_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations);

/// Get the interior facet from a TopologyStorageLayer
std::shared_ptr<const std::vector<bool>>
interior_facets(const TopologyStorageLayer& storage);

/// Get the connectivity for dimensions (d0, d1) from a TopologyStorageLayer
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
connectivity(const TopologyStorageLayer& storage, int d0, int d1);

/// Get the index map for dimensions dim from a TopologyStorageLayer
std::shared_ptr<const common::IndexMap>
index_map(const TopologyStorageLayer& storage, int dim);

/// Get the cell permutation information from a TopologyStorageLayer
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
cell_permutations(const TopologyStorageLayer& storage);

/// Get the facet permutation information from a TopologyStorageLayer
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
facet_permutations(const TopologyStorageLayer& storage);

/// Assigns non-empty field from "from" to "to" and return number of assignments
/// performed. Data is not overwritten by default.
int assign(TopologyStorageLayer& to, const TopologyStorageLayer& from, bool override=false);

/// Assigns non-empty field from "from" to "to" only if the present field is
/// empty (holds a nullptr) and return number of assignments performed
int assign_where_empty(TopologyStorageLayer& to, const TopologyStorageLayer& from);

using TopologyStorage
    = common::memory::LayerManager<TopologyStorageLayer>;

using TopologyStorageLock = TopologyStorage::LayerLock_t;

/// Walk through the storage "from" and assigns its layers to "to" beginning
/// with the oldest layer, ending with the newest one, i.e. it copies over the
/// current state of "from".
void assign(TopologyStorageLayer& to, const TopologyStorage& from, bool override=false);

/// Walk through the storage "from" and assigns its layers to the current
/// writing layer of "to" beginning with the oldest layer, ending with the
/// newest one, i.e. it copies over the current state of "from".
void assign(TopologyStorage& to, const TopologyStorage& from,
               bool override=false);

/// Version of assign that does not throw when "to" is empty (does nothing)
void assign_if_not_empty(TopologyStorage& to, const TopologyStorage& from,
            bool override=false);


} // namespace dolfinx::mesh::storage
