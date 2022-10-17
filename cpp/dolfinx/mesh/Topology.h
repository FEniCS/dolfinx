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
  /// Create empty mesh topology
  Topology(MPI_Comm comm, CellType type);

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
  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim, std::shared_ptr<const common::IndexMap> map);

  /// @brief Get the IndexMap that described the parallel distribution
  /// of the mesh entities.
  ///
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension `dim`. Returns
  /// `nullptr` if index map has not been set.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @brief Return connectivity from entities of dimension d0 to
  /// entities of dimension d1.
  ///
  /// @param[in] d0
  /// @param[in] d1
  /// @return The adjacency list that for each entity of dimension d0
  /// gives the list of incident entities of dimension d1. Returns
  /// `nullptr` if connectivity has not been computed.
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// @todo Merge with set_index_map
  /// @brief Set connectivity for given pair of topological dimensions.
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// Returns the permutation information
  const std::vector<std::uint32_t>& get_cell_permutation_info() const;

  /// @brief Get the permutation number to apply to a facet.
  ///
  /// The permutations are numbered so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// Each column of the returned array represents a cell, and each row
  /// a facet of that cell.
  /// @return The permutation number
  /// @note An exception is raised if the permutations have not been
  /// computed
  const std::vector<std::uint8_t>& get_facet_permutations() const;

  /// Cell type
  /// @return Cell type that the topology is for
  CellType cell_type() const noexcept;

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

  /// Compute entity permutations and reflections
  void create_entity_permutations();

  /// List of inter-process facets, if facet topology has been computed
  const std::vector<std::int32_t>& interprocess_facets() const;

  /// Original cell index
  std::vector<std::int64_t> original_cell_index;

  /// Mesh MPI communicator
  /// @return The communicator on which the topology is distributed
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Cell type
  CellType _cell_type;

  // Parallel layout of entities for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs [d0][d1] == d0 -> d1 connectivity
  std::vector<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>>
      _connectivity;

  // The facet permutations (local facet, cell))
  // [cell0_0, cell0_1, ,cell0_2, cell1_0, cell1_1, ,cell1_2, ...,
  // celln_0, celln_1, ,celln_2,]
  std::vector<std::uint8_t> _facet_permutations;

  // Cell permutation info. See the documentation for
  // get_cell_permutation_info for documentation of how this is encoded.
  std::vector<std::uint32_t> _cell_permutations;

  // List of facets that are on the inter-process boundary
  std::vector<std::int32_t> _interprocess_facets;
};

/// @brief Create a distributed mesh topology.
///
/// @param[in] comm MPI communicator across which the topology is
/// distributed
/// @param[in] cells The cell topology (list of vertices for each cell)
/// using global indices for the vertices. It contains cells that have
/// been distributed to this rank, e.g. via a graph partitioner. It must
/// also contain all ghost cells via facet, i.e. cells that are on a
/// neighboring process and share a facet with a local cell.
/// @param[in] original_cell_index The original global index associated
/// with each cell
/// @param[in] ghost_owners The owning rank of each ghost cell (ghost
/// cells are always at the end of the list of `cells`)
/// @param[in] cell_type The cell shape
/// @param[in] boundary_vertices List of vertices on the exterior of the
/// local mesh which may be shared with other processes.
/// @return A distributed mesh topology
Topology
create_topology(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                const std::span<const std::int64_t>& original_cell_index,
                const std::span<const int>& ghost_owners,
                const CellType& cell_type,
                const std::vector<std::int64_t>& boundary_vertices);

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
                  const graph::AdjacencyList<std::int32_t>& entities);
} // namespace dolfinx::mesh
