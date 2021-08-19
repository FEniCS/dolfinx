// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::fem
{
class ElementDofLayout;
}

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::mesh
{
enum class GhostMode : int;

enum class CellType;
class Topology;

/// Compute marker for owned facets that are on the exterior of the
/// domain, i.e. are connected to only one cell. The function does not
/// require parallel communication.
/// @param[in] topology The topology
/// @return Vector with length equal to the number of owned facets on
///   this this process. True if the ith facet (local index) is on the
///   exterior of the domain.
std::vector<bool> compute_boundary_facets(const Topology& topology);

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities).
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.
class Topology
{
public:
  /// Create empty mesh topology
  Topology(MPI_Comm comm, mesh::CellType type);

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

  /// Return the topological dimension of the mesh
  int dim() const;

  /// @todo Merge with set_connectivity
  ///
  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     const std::shared_ptr<const common::IndexMap>& map);

  /// Get the IndexMap that described the parallel distribution of the
  /// mesh entities
  /// @param[in] dim Topological dimension
  /// @return Index map for the entities of dimension @p dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

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

  /// Returns the permutation information
  const std::vector<std::uint32_t>& get_cell_permutation_info() const;

  /// Get the permutation number to apply to a facet. The permutations
  /// are numbered so that:
  ///
  ///   - `n % 2` gives the number of reflections to apply
  ///   - `n // 2` gives the number of rotations to apply
  ///
  /// Each column of the returned array represents a cell, and each row
  /// a facet of that cell.
  /// @return The permutation number
  /// @note An exception i raised if the permutations have not been
  /// computed
  const std::vector<std::uint8_t>& get_facet_permutations() const;

  /// Cell type
  /// @return Cell type that the topology is for
  mesh::CellType cell_type() const;

  // TODO: Rework memory management and associated API
  // Currently, there is no clear caching policy implemented and no way of
  // discarding cached data.

  /// Create entities of given topological dimension.
  /// @param[in] dim Topological dimension
  /// @return Number of newly created entities, returns -1 if entities
  /// already existed
  std::int32_t create_entities(int dim);

  /// Create connectivity between given pair of dimensions, d0 -> d1
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(int d0, int d1);

  /// Compute entity permutations and reflections
  void create_entity_permutations();

  /// Mesh MPI communicator
  /// @return The communicator on which the topology is distributed
  MPI_Comm mpi_comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // Cell type
  mesh::CellType _cell_type;

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs of topological dimensions
  std::vector<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>>
      _connectivity;

  // The facet permutations (local facet, cell))
  // [cell0_0, cell0_1, ,cell0_2, cell1_0, cell1_1, ,cell1_2, ...,
  // celln_0, celln_1, ,celln_2,]
  std::vector<std::uint8_t> _facet_permutations;

  // Cell permutation info. See the documentation for
  // get_cell_permutation_info for documentation of how this is encoded.
  std::vector<std::uint32_t> _cell_permutations;
};

/// Create distributed topology
///
/// @param[in] comm MPI communicator across which the topology is
/// distributed
/// @param[in] cells The cell topology (list of cell vertices) using
/// global indices for the vertices. It contains cells that have been
/// distributed to this rank, e.g. via a graph partitioner. It must also
/// contain all ghost cells via facet, i.e. cells that are on a
/// neighboring process and share a facet with a local cell.
/// @param[in] original_cell_index The original global index associated
/// with each cell
/// @param[in] ghost_owners The ownership of the ghost cells (ghost
/// cells are always at the end of the list of @p cells)
/// @param[in] cell_type The cell shape
/// @param[in] ghost_mode How to partition the cell overlap: none,
/// shared_facet or shared_vertex
/// @return A distributed Topology
Topology
create_topology(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                const xtl::span<const std::int64_t>& original_cell_index,
                const xtl::span<const int>& ghost_owners,
                const CellType& cell_type, mesh::GhostMode ghost_mode);
} // namespace dolfinx::mesh
