// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
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
  /// @brief Create empty mesh topology with multiple cell types.
  ///
  /// @warning Experimental
  ///
  /// @param[in] comm MPI communicator.
  /// @param[in] cell_types Types of cells.
  /// @param[in] vertex_map Index map describing the distribution of
  /// mesh vertices.
  /// @param[in] cell_maps Index maps describing the distribution of
  /// mesh cells for each cell type in `cell_types`.
  /// @param[in] cells Cell-to-vertex connectivities for each cell type
  /// in `cell_types`.
  /// @param[in] original_cell_index Original indices for each cell in
  /// `cells`.
  Topology(
      MPI_Comm comm, std::vector<CellType> cell_types,
      std::shared_ptr<const common::IndexMap> vertex_map,
      std::vector<std::shared_ptr<const common::IndexMap>> cell_maps,
      std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells,
      const std::optional<std::vector<std::vector<std::int64_t>>>&
          original_cell_index
      = std::nullopt);

  /// @brief Topology constructor.
  /// @param[in] comm MPI communicator.
  /// @param[in] cell_type Type of cell.
  /// @param[in] vertex_map Index map describing the distribution of
  /// mesh vertices.
  /// @param[in] cell_map Index map describing the distribution of mesh
  /// cells.
  /// @param[in] cells Cell-to-vertex connectivity.
  /// @param[in] original_index Original index for each cell in `cells`.
  Topology(MPI_Comm comm, CellType cell_type,
           std::shared_ptr<const common::IndexMap> vertex_map,
           std::shared_ptr<const common::IndexMap> cell_map,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>> cells,
           const std::optional<std::vector<std::int64_t>>& original_index
           = std::nullopt);

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

  /// @brief Entity types in the topology for a given dimension.
  /// @param[in] dim Topological dimension.
  /// @return Entity types.
  const std::vector<CellType>& entity_types(int dim) const;

  /// @brief Cell type.
  ///
  /// This function is is for topologies with one cell type only.
  ///
  /// @return Cell type that the topology is for.
  CellType cell_type() const;

  /// @brief Get the index maps that described the parallel distribution
  /// of the mesh entities of a given topological dimension.
  ///
  /// @warning Experimental
  ///
  /// @param[in] dim Topological dimension.
  /// @return Index maps, one for each cell type.
  std::vector<std::shared_ptr<const common::IndexMap>>
  index_maps(int dim) const;

  /// @brief Get the IndexMap that described the parallel distribution
  /// of the mesh entities.
  ///
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension `dim`. Returns
  /// `nullptr` if index map has not been set.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @brief Get the connectivity from entities of topological
  /// dimension d0 to dimension d1.
  ///
  /// The entity type and incident entity type are each described by a
  /// pair (dim, index). The index within a topological dimension `dim`,
  /// is that of the cell type given in `entity_types(dim)`.
  ///
  /// @param[in] d0 Pair of (topological dimension of entities, index of
  /// "entity type" within topological dimension).
  /// @param[in] d1 Pair of (topological dimension of entities, index of
  /// incident "entity type" within topological dimension).
  /// @return AdjacencyList of connectivity from entity type in d0 to
  /// entity types in d1, or nullptr if not yet computed.
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(std::array<int, 2> d0, std::array<int, 2> d1) const;

  /// @brief Return connectivity from entities of dimension d0 to
  /// entities of dimension d1. Assumes only one entity type per dimension.
  ///
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  /// gives the list of incident entities of dimension d1. Returns
  /// `nullptr` if connectivity has not been computed.
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// @brief Returns the permutation information
  const std::vector<std::uint32_t>& get_cell_permutation_info() const;

  /// @brief Get the numbers that encode the number of permutations to
  /// apply to facets.
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

  /// @brief List of inter-process facets of a given type.
  ///
  /// "Inter-process" facets are facets that are connected (1) to a cell
  /// that is owned by the calling process (rank) and (2) to a cell that
  /// is owned by another process.
  ///
  /// Facets must have been computed for inter-process facet data to be
  /// available.
  ///
  ///  @param[in] index Index of facet type, following the order given
  ///  by ::entity_types.
  /// @return Indices of the inter-process facets.
  const std::vector<std::int32_t>& interprocess_facets(int index) const;

  /// @brief List of inter-process facets.
  ///
  /// "Inter-process" facets are facets that are connected (1) to a cell
  /// that is owned by the calling process (rank) and (2) to a cell that
  /// is owned by another process.
  ///
  /// @pre Inter-process facets are available only if facet topology has
  /// been computed.
  const std::vector<std::int32_t>& interprocess_facets() const;

private:
  /// @todo Merge with set_connectivity
  ///
  /// @brief Set the IndexMap for the `i`th `celltype` of dimension `dim`.
  ///
  /// @warning This is experimental and likely to change.
  ///
  /// @param[in] dim Topological dimension.
  /// @param[in] i Index of cell type within dimension `dim`. The cell types
  /// in the mesh for a given dimension are returned by ::entity_types.
  /// @param[in] map Index map to set.
  // void set_index_map(int dim, int i,
  //                    std::shared_ptr<const common::IndexMap> map);

  // /// @todo Merge with set_connectivity
  // ///
  // /// @brief Set the IndexMap for dimension dim
  // /// @warning This is experimental and likely to change
  // void set_index_map(int dim, std::shared_ptr<const common::IndexMap> map);

public:
  /// @brief Set connectivity for given pair of entity types, defined by
  /// dimension and index, as listed in ::entity_types.
  ///
  /// General version for mixed topology. Connectivity from d0 to d1.
  ///
  /// @warning Experimental.
  ///
  /// @param[in] c Connectivity.
  /// @param[in] d0 Pair of (topological dimension of entities, index of
  /// "entity type" within topological dimension).
  /// @param[in] d1 Pair of (topological dimension of incident entities,
  /// index of incident "entity type" within topological dimension).
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        std::array<int, 2> d0, std::array<int, 2> d1);

  /// @todo Merge with set_index_map
  /// @brief Set connectivity for given pair of topological dimensions.
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// @brief Create entities of given topological dimension.
  /// @param[in] dim Topological dimension of entities to compute.
  /// @return True if entities are created, false if entities already
  /// existed.
  bool create_entities(int dim);

  /// @brief Create connectivity between given pair of dimensions, `d0
  /// -> d1`.
  /// @param[in] d0 Topological dimension.
  /// @param[in] d1 Topological dimension.
  void create_connectivity(int d0, int d1);

  /// @brief Compute entity permutations and reflections.
  void create_entity_permutations();

  /// Original cell index for each cell type
  std::vector<std::vector<std::int64_t>> original_cell_index;

  /// @brief Mesh MPI communicator.
  /// @return Communicator on which the topology is distributed.
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Cell types for entities in Topology, where _entity_types_new[d][i]
  // is the ith entity type of dimension d
  std::vector<std::vector<CellType>> _entity_types;

  // Parallel layout of entities for each dimension and cell type
  // flattened in the same layout as _entity_types above.
  // std::vector<std::shared_ptr<const common::IndexMap>> _index_map;

  // _index_maps[d][i] is the index map for the ith entity type of
  // dimension d
  std::vector<std::vector<std::shared_ptr<const common::IndexMap>>> _index_maps;
  std::map<std::array<int, 2>, std::shared_ptr<const common::IndexMap>>
      _index_maps_new;

  // Connectivity between cell types _connectivity_new[(dim0, i0),
  // (dim1, i1)] is the connection from (dim0, i0) -> (dim1, i1),
  // where dim0 and dim1 are topological dimensions and i0 and i1
  // are the indices of cell types (following the order in _entity_types).
  std::map<std::pair<std::array<int, 2>, std::array<int, 2>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>>
      _connectivity;

  // The facet permutations (local facet, cell))
  // [cell0_0, cell0_1, ,cell0_2, cell1_0, cell1_1, ,cell1_2, ...,
  // celln_0, celln_1, ,celln_2,]
  std::vector<std::uint8_t> _facet_permutations;

  // Cell permutation info. See the documentation for
  // get_cell_permutation_info for documentation of how this is encoded.
  std::vector<std::uint32_t> _cell_permutations;

  // List of facets that are on the inter-process boundary for each
  // facet type. _interprocess_facets[i] is the inter-process facets of
  // facet type i.
  std::vector<std::vector<std::int32_t>> _interprocess_facets;
};

/// @brief Create a mesh topology.
///
/// This function creates a Topology from cells that have been already
/// distributed to the processes that own or ghost the cell.
///
/// @param[in] comm Communicator across which the topology is
/// distributed.
/// @param[in] cell_types List of cell types in the topology.
/// @param[in] cells Cell topology (list of vertices for each cell) for
/// each cell type using global indices for the vertices. The cell type
/// for `cells[i]` is `cell_types[i]`. Each `cells[i]` contains cells
/// that have been distributed to this rank, e.g. via a graph
/// partitioner. It must also contain all ghost cells via facet, i.e.
/// cells that are on a neighboring process and which share a facet with
/// a local cell. Ghost cells are the last `n` entries in `cells[i]`, where
/// `n` is given by the length of `ghost_owners[i]`.
/// @param[in] original_cell_index Input cell index for each cell type.
/// @param[in] ghost_owners Owning rank for ghost cells (at end of each list of
/// cells).
/// @param[in] boundary_vertices Vertices on the 'exterior' (boundary)
/// of the local topology. These vertices might appear on other
/// processes.
/// @return A distributed mesh topology
Topology
create_topology(MPI_Comm comm, const std::vector<CellType>& cell_types,
                std::vector<std::span<const std::int64_t>> cells,
                std::vector<std::span<const std::int64_t>> original_cell_index,
                std::vector<std::span<const int>> ghost_owners,
                std::span<const std::int64_t> boundary_vertices);

/// @brief Create a mesh topology for a single cell type.
///
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

/// @brief Create a topology for a subset of entities of a given
/// topological dimension.
///
/// @param[in] topology Original (parent) topology.
/// @param[in] dim Topological dimension of the entities in the new
/// topology.
/// @param[in] entities Indices of entities in `topology` to include in
/// the new topology.
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
