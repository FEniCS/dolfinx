// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::mesh
{
enum class CellType;

/// @brief Topology stores the topology of a mesh, consisting of mesh
/// entities and connectivity (incidence relations for the mesh
/// entities).
///
/// A mesh entity e may be identified globally as a pair `e = (dim, i)`,
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.
///
/// @todo Rework memory management and associated API. Currently, there
/// is no clear caching policy implemented and no way of discarding
/// cached data.
class Topology
{
public:
  /// @brief Empty Topology constructor
  /// @param comm MPI communicator
  /// @param cell_type Type of cell
  Topology(MPI_Comm comm, CellType cell_type,
           std::shared_ptr<const common::IndexMap> vertex_map,
           std::shared_ptr<const common::IndexMap> cell_map,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>> cells,
           std::span<const std::size_t> original_cell_index);

  /// @brief Create empty mesh topology with multiple cell types
  /// @param comm MPI communicator
  /// @param cell_type List of cell types
  /// @warning Experimental
  Topology(
      MPI_Comm comm, const std::vector<CellType>& cell_type,
      std::shared_ptr<const common::IndexMap> vertex_map,
      std::vector<std::shared_ptr<const common::IndexMap>> cell_map,
      std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells,
      std::vector<std::span<const std::int64_t>> original_cell_index);

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

  /// @brief Return the topological dimension of the mesh.
  int dim() const noexcept;

  /// @todo Merge with set_connectivity
  ///
  /// @brief Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim, std::shared_ptr<const common::IndexMap> map);

  /// @todo Merge with set_connectivity
  ///
  /// @brief Set the IndexMap for the `i`th celltype of dimension dim
  /// @warning This is experimental and likely to change
  /// @param dim Topological dimension
  /// @param i Index of cell type within dimension `dim`. Cell types for each
  /// dimension can be obtained with `entity_types(dim)`.
  /// @param map Map to set
  void set_index_map(std::int8_t dim, std::int8_t i,
                     std::shared_ptr<const common::IndexMap> map);

  /// @brief Get the IndexMap that described the parallel distribution
  /// of the mesh entities.
  ///
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension `dim`. Returns
  /// `nullptr` if index map has not been set.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @param dim Topological dimension
  /// @warning Experimental
  /// @return List of index maps, one for each cell type
  std::vector<std::shared_ptr<const common::IndexMap>>
  index_maps(std::int8_t dim) const;

  /// @brief Return connectivity from entities of dimension d0 to
  /// entities of dimension d1. Assumes only one entity type per
  /// dimension.
  ///
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  /// gives the list of incident entities of dimension d1. Returns
  /// `nullptr` if connectivity has not been computed.
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// @brief Return the connectivity from entities of topological
  /// dimension d0 to dimension d1.
  ///
  /// The entity type, and incident entity type are each described by a
  /// pair (dim, index). The index within a topological dimension `dim`,
  /// is that of the cell type given in `entity_types(dim)`.
  ///
  /// @param d0 Pair of (topological dimension of entities, index of
  /// "entity type" within topological dimension)
  /// @param d1 Pair of (topological dimension of indicent entities,
  /// index of incident "entity type" within topological dimension)
  /// @return AdjacencyList of connectivity from entity type in d0 to
  /// entity types in d1, or nullptr if not yet computed.
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(std::pair<std::int8_t, std::int8_t> d0,
               std::pair<std::int8_t, std::int8_t> d1) const;

  /// @todo Merge with set_index_map
  /// @brief Set connectivity for given pair of topological dimensions.
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// @brief Set connectivity for given pair of entity types, defined by
  /// dimension and index, as listed in `entity_types()`.
  ///
  /// General version for mixed topology. Connectivity from d0 to d1.
  ///
  /// @param c Connectivity AdjacencyList.
  /// @param d0 Pair of (topological dimension of entities, index of
  /// "entity type" within topological dimension).
  /// @param d1 Pair of (topological dimension of indicent entities,
  /// index of incident "entity type" within topological dimension).
  /// @warning Experimental.
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        std::pair<std::int8_t, std::int8_t> d0,
                        std::pair<std::int8_t, std::int8_t> d1);

  /// @brief Returns the permutation information
  const std::vector<std::uint32_t>& get_cell_permutation_info() const;

  /// @brief Get the numbers that encode the number of permutations to apply to
  /// facets.
  ///
  /// The permutations are encoded so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// The data is stored in a flattened 2D array, so that `data[cell_index *
  /// facets_per_cell + facet_index]` contains the facet with index
  /// `facet_index` of the cell with index `cell_index`.
  /// @return The encoded permutation info
  /// @note An exception is raised if the permutations have not been
  /// computed
  const std::vector<std::uint8_t>& get_facet_permutations() const;

  /// @brief Cell type
  /// @return Cell type that the topology is for
  CellType cell_type() const;

  /// @brief Get the entity types in the topology for a given dimension
  /// @param dim Topological dimension
  /// @return List of entity types
  std::vector<CellType> entity_types(std::int8_t dim) const;

  /// @brief Create entities of given topological dimension.
  /// @param[in] dim Topological dimension
  /// @return Number of newly created entities, returns -1 if entities
  /// already existed
  std::int32_t create_entities(int dim);

  /// @brief Create connectivity between given pair of dimensions, `d0
  /// -> d1`.
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(int d0, int d1);

  /// @brief Compute entity permutations and reflections.
  void create_entity_permutations();

  /// @brief List of inter-process facets.
  ///
  /// "Inter-process" facets are facets that are connected (1) to a cell
  /// that is owned by the calling process (rank) and (2) to a cell that
  /// is owned by another process.
  ///
  /// @pre Inter-process facets are available only if facet topology has
  /// been computed.
  const std::vector<std::int32_t>& interprocess_facets() const;

  /// @brief List of inter-process facets, if facet topology has been
  /// computed, for the facet type in `Topology::entity_types`
  /// identified by index.
  /// @param index Index of facet type
  const std::vector<std::int32_t>& interprocess_facets(std::int8_t index) const;

  /// Original cell index for each cell type
  std::vector<std::vector<std::int64_t>> original_cell_index;

  /// Mesh MPI communicator
  /// @return The communicator on which the topology is distributed
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Cell types for entites in Topology, as follows:
  // [CellType::point, edge_types..., facet_types..., cell_types...]
  // Only one type is expected for vertices, (and usually edges), but facets
  // and cells can be a list of multiple types, e.g. [quadrilateral, triangle]
  // for facets.
  // Offsets are position in the list for each entity dimension, in
  // AdjacencyList style.
  std::vector<CellType> _entity_types;
  std::vector<std::int8_t> _entity_type_offsets;

  // Parallel layout of entities for each dimension and cell type
  // flattened in the same layout as _entity_types above.
  std::vector<std::shared_ptr<const common::IndexMap>> _index_map;

  // Connectivity between entity dimensions and cell types, arranged as
  // a 2D array. The indexing follows the order in _entity_types, i.e.
  // increasing in topological dimension. There may be multiple types in each
  // dimension, e.g. triangle and quadrilateral facets.
  // Connectivity between different entity types of same dimension will always
  // be nullptr.
  std::vector<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>>
      _connectivity;

  // The facet permutations (local facet, cell))
  // [cell0_0, cell0_1, ,cell0_2, cell1_0, cell1_1, ,cell1_2, ...,
  // celln_0, celln_1, ,celln_2,]
  std::vector<std::uint8_t> _facet_permutations;

  // Cell permutation info. See the documentation for
  // get_cell_permutation_info for documentation of how this is encoded.
  std::vector<std::uint32_t> _cell_permutations;

  // List of facets that are on the inter-process boundary for each facet type
  std::vector<std::vector<std::int32_t>> _interprocess_facets;
};

/// @brief Create a mesh topology.
///
/// This function creates a Topology from cells that have been
/// distributed to the processes that own or ghost the cell.
///
/// @param[in] comm Communicator across which the topology is
/// distributed.
/// @param[in] cells Cell topology (list of vertices for each cell)
/// using global indices for the vertices. It contains cells that have
/// been distributed to this rank, e.g. via a graph partitioner. It must
/// also contain all ghost cells via facet, i.e. cells that are on a
/// neighboring process and which share a facet with a local cell. Ghost
/// cells are the last `n` entries in `cells`, where `n` is given by the
/// length of `ghost_owners`.
/// @param[in] original_cell_index Original global index associated with
/// each cell.
/// @param[in] ghost_owners Owning rank of each ghost cell (ghost cells
/// are always at the end of the list of `cells`).
/// @param[in] cell_type A vector with cell shapes.
/// @param[in] boundary_vertices Vertices on the 'exterior' (boundary)
/// of the local topology. These vertices might appear on other
/// processes.
/// @return A distributed mesh topology
Topology create_topology(MPI_Comm comm, std::span<const std::int64_t> cells,
                         std::span<const std::int64_t> original_cell_index,
                         std::span<const int> ghost_owners, CellType cell_type,
                         std::span<const std::int64_t> boundary_vertices);

/// @brief Create a topology of mixed cell types.
///
/// @param comm MPI Communicator.
/// @param cell_type List of cell types.
/// @param cells Lists of cells, using vertex indices, flattened, for
/// each cell type.
/// @param original_cell_index Input cell index for each cell type.
/// @param ghost_owners Owning rank for ghost cells (at end of each list
/// of cells).
/// @param boundary_vertices Vertices of undetermined ownership on
/// external or inter-process boundary.
/// @return
Topology
create_topology(MPI_Comm comm, const std::vector<CellType>& cell_type,
                std::vector<std::span<const std::int64_t>> cells,
                std::vector<std::span<const std::int64_t>> original_cell_index,
                std::vector<std::span<const int>> ghost_owners,
                std::span<const std::int64_t> boundary_vertices);

/// @brief Create a topology for a subset of entities of a given
/// topological dimension.
///
/// @param topology Original (parent) topology.
/// @param dim Topological dimension of the entities in the new topology.
/// @param entities Indices of entities in `topology` to include in the
/// new topology.
/// @return New topology of dimension `dim` with all entities in
/// `entities`, map from entities of dimension `dim` in new sub-topology
/// to entities in `topology`, and map from vertices in new sub-topology
/// to vertices in `topology`.
std::tuple<Topology, std::vector<int32_t>, std::vector<int32_t>>
create_subtopology(const Topology& topology, int dim,
                   std::span<const std::int32_t> entities);

/// @brief Get entity indices for entities defined by their vertices.
///
/// @warning This function may be removed in the future.
///
/// @param[in] topology The mesh topology
/// @param[in] dim Topological dimension of the entities
/// @param[in] entities The mesh entities defined by their vertices
/// @return The index of the ith entity in `entities`
/// @note If an entity cannot be found on this rank, -1 is returned as
/// the index.
std::vector<std::int32_t>
entities_to_index(const Topology& topology, int dim,
                  std::span<const std::int32_t> entities);
} // namespace dolfinx::mesh
