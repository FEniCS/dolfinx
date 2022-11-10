// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

/// @file DofMap.h
/// @brief Degree-of-freedom map representations and tools

#pragma once

#include "ElementDofLayout.h"
#include <concepts>
#include <cstdlib>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <functional>
#include <memory>
#include <mpi.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
}

namespace dolfinx::fem
{

/// @brief Create an adjacency list that maps a global index
/// (process-wise) to the 'unassembled' cell-wise contributions.
///
/// It is built from the usual (cell, local index) -> global index dof
/// map. An 'unassembled' vector is the stacked cell contributions,
/// ordered by cell index. If the usual dof map is:
///
///  `Cell:                0          1          2          3` \n
///  `Global index:  [ [0, 3, 5], [3, 2, 4], [4, 3, 2], [2, 1, 0]]`
///
/// the 'transpose' dof map will be:
///
///  `Global index:           0      1        2          3        4      5` \n
///  `Unassembled index: [ [0, 11], [10], [4, 8, 9], [1, 3, 7], [5, 6], [2] ]`
///
/// @param[in] dofmap The standard dof map that for each cell (node)
/// gives the global (process-wise) index of each local (cell-wise)
/// index.
/// @param[in] num_cells The number of cells (nodes) in @p dofmap to
/// consider. The first @p num_cells are used. This is argument is
/// typically used to exclude ghost cell contributions.
/// @return Map from global (process-wise) index to positions in an
/// unaassembled array. The links for each node are sorted.
graph::AdjacencyList<std::int32_t>
transpose_dofmap(const graph::AdjacencyList<std::int32_t>& dofmap,
                 std::int32_t num_cells);

/// @brief Degree-of-freedom map.
///
/// This class handles the mapping of degrees of freedom. It builds a
/// dof map based on an ElementDofLayout on a specific mesh topology. It
/// will reorder the dofs when running in parallel. Sub-dofmaps, both
/// views and copies, are supported.
class DofMap
{
public:
  /// @brief Create a DofMap from the layout of dofs on a reference
  /// element, an IndexMap defining the distribution of dofs across
  /// processes and a vector of indices.
  ///
  /// @param[in] element The layout of the degrees of freedom on an
  /// element
  /// @param[in] index_map The map describing the parallel distribution
  /// of the degrees of freedom.
  /// @param[in] index_map_bs The block size associated with the
  /// `index_map`.
  /// @param[in] dofmap Adjacency list with the degrees-of-freedom for
  /// each cell.
  /// @param[in] bs The block size of the `dofmap`.
  template <std::convertible_to<fem::ElementDofLayout> E,
            std::convertible_to<graph::AdjacencyList<std::int32_t>> U>
  DofMap(E&& element, std::shared_ptr<const common::IndexMap> index_map,
         int index_map_bs, U&& dofmap, int bs)
      : index_map(index_map), _index_map_bs(index_map_bs),
        _element_dof_layout(std::forward<E>(element)),
        _dofmap(std::forward<U>(dofmap)), _bs(bs)
  {
    // Do nothing
  }

  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  // Destructor
  virtual ~DofMap() = default;

  // Copy assignment
  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// @brief Equality operator
  /// @return Returns true if the data for the two dofmaps is equal
  bool operator==(const DofMap& map) const;

  /// @brief Local-to-global mapping of dofs on a cell
  /// @param[in] cell The cell index
  /// @return Local-global dof map for the cell (using process-local
  /// indices)
  std::span<const std::int32_t> cell_dofs(int cell) const
  {
    return _dofmap.links(cell);
  }

  /// @brief Return the block size for the dofmap
  int bs() const noexcept;

  /// @brief Extract subdofmap component
  /// @param[in] component The component indices
  /// @return The dofmap for the component
  DofMap extract_sub_dofmap(const std::vector<int>& component) const;

  /// @brief Create a "collapsed" dofmap (collapses a sub-dofmap)
  /// @param[in] comm MPI Communicator
  /// @param[in] topology The mesh topology that the dofmap is defined
  /// on
  /// @param[in] reorder_fn The graph re-ordering function to apply to
  /// the dof data
  /// @return The collapsed dofmap
  std::pair<DofMap, std::vector<std::int32_t>> collapse(
      MPI_Comm comm, const mesh::Topology& topology,
      const std::function<std::vector<int>(
          const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
      = [](const graph::AdjacencyList<std::int32_t>& g)
      { return graph::reorder_gps(g); }) const;

  /// @brief Get dofmap data
  /// @return The adjacency list with dof indices for each cell
  const graph::AdjacencyList<std::int32_t>& list() const;

  /// Layout of dofs on an element
  const ElementDofLayout& element_dof_layout() const
  {
    return _element_dof_layout;
  }

  /// @brief Index map that describes the parallel distribution of the
  /// dofmap
  std::shared_ptr<const common::IndexMap> index_map;

  /// @brief Block size associated with the index_map
  int index_map_bs() const;

private:
  // Block size for the IndexMap
  int _index_map_bs = -1;

  // Layout of dofs on a cell
  ElementDofLayout _element_dof_layout;

  // Cell-local-to-dof map (dofs for cell dofmap[cell])
  graph::AdjacencyList<std::int32_t> _dofmap;

  // Block size for the dofmap
  int _bs = -1;
};
} // namespace dolfinx::fem
