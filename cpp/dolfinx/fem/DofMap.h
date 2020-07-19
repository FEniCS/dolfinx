// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdlib>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
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
class ElementDofLayout;

/// Create an adjacency list that maps a global index (process-wise) to
/// the 'unassembled' cell-wise contributions. It is built from the
/// usual (cell, local index) -> global index dof map. An 'unassembled'
/// vector is the stacked cell contributions, ordered by cell index.
///
/// If the usual dof map is:
///
///  Cell:                0          1          2          3
///  Global index:  [ [0, 3, 5], [3, 2, 4], [4, 3, 2], [2, 1, 0]]
///
/// the 'transpose' dof map will be:
///
///  Global index:           0      1        2          3        4      5
///  Unassembled index: [ [0, 11], [10], [4, 8, 9], [1, 3, 7], [5, 6], [2] ]
///
/// @param[in] dofmap The standard dof map that for each cell (node)
///   gives the global (process-wise) index of each local (cell-wise)
///   index.
/// @param[in] num_cells The number of cells (nodes) in @p dofmap to
///   consider. The first @p num_cells are used. This is argument is
///   typically used to exclude ghost cell contributions.
/// @return Map from global (process-wise) index to positions in an
///   unaassembled array. The links for each node are sorted.
graph::AdjacencyList<std::int32_t>
transpose_dofmap(graph::AdjacencyList<std::int32_t>& dofmap,
                 std::int32_t num_cells);

/// Degree-of-freedom map
///
/// This class handles the mapping of degrees of freedom. It builds a
/// dof map based on an ElementDofLayout on a specific mesh topology. It
/// will reorder the dofs when running in parallel. Sub-dofmaps, both
/// views and copies, are supported.

class DofMap
{
public:
  /// Create a DofMap from the layout of dofs on a reference element, an
  /// IndexMap defining the distribution of dofs across processes and a
  /// vector of indices
  DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
         std::shared_ptr<const common::IndexMap> index_map,
         const graph::AdjacencyList<std::int32_t>& dofmap);

  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  virtual ~DofMap() = default;

  // Copy assignment
  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// Local-to-global mapping of dofs on a cell
  /// @param[in] cell The cell index
  /// @return Local-global dof map for the cell (using process-local
  ///   indices)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::ConstSegmentReturnType
  cell_dofs(int cell) const
  {
    return _dofmap.links(cell);
  }

  /// Extract subdofmap component
  /// @param[in] component The component indices
  /// @return The dofmap for the component
  DofMap extract_sub_dofmap(const std::vector<int>& component) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  /// @param[in] comm MPI Communicator
  /// @param[in] topology The mesh topology that the dofmap is defined
  ///   on
  /// @return The collapsed dofmap
  std::pair<std::unique_ptr<DofMap>, std::vector<std::int32_t>>
  collapse(MPI_Comm comm, const mesh::Topology& topology) const;

  /// Get dofmap data
  /// @return The adjacency list with dof indices for each cell
  const graph::AdjacencyList<std::int32_t>& list() const { return _dofmap; }

  /// Layout of dofs on an element
  std::shared_ptr<const ElementDofLayout> element_dof_layout;

  /// Index map that described the parallel distribution of the dofmap
  std::shared_ptr<const common::IndexMap> index_map;

private:
  // Cell-local-to-dof map (dofs for cell dofmap[cell])
  graph::AdjacencyList<std::int32_t> _dofmap;
};
} // namespace dolfinx::fem