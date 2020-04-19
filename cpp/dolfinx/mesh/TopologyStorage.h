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
interior_facets(const TopologyStorageLayer& storage);

std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
connectivity(const TopologyStorageLayer& storage, int d0, int d1);

std::shared_ptr<const common::IndexMap>
index_map(const TopologyStorageLayer& storage, int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
cell_permutations(const TopologyStorageLayer& storage);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
facet_permutations(const TopologyStorageLayer& storage);

using TopologyStorageManager
    = dolfinx::common::memory::LayerManager<TopologyStorageLayer>;

/// Set markers for owned facets that are interior in given storage
/// @param[in,out] storage Object where to store the created entities
/// @param[in] interior_facets The marker vector
std::shared_ptr<const std::vector<bool>>
set_interior_facets(TopologyStorageManager& storage,
                    std::shared_ptr<const std::vector<bool>> interior_facets);

/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
set_connectivity(TopologyStorageManager& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1);

/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
set_index_map(TopologyStorageManager& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
set_cell_permutations(
    TopologyStorageManager& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
set_facet_permutations(
    TopologyStorageManager& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations);

std::shared_ptr<const std::vector<bool>>
interior_facets(const TopologyStorageManager& storage);

std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
connectivity(const TopologyStorageManager& storage, int d0, int d1);

std::shared_ptr<const common::IndexMap>
index_map(const TopologyStorageManager& storage, int dim);

std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
cell_permutations(const TopologyStorageManager& storage);

std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
facet_permutations(const TopologyStorageManager& storage);

} // namespace dolfinx::mesh::storage
