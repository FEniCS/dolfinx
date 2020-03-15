// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include "petscsys.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Topology;
} // namespace mesh

namespace fem
{

/// Degree-of-freedom map

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
  /// @return Local-global dof map for the cell (using process-local indices)
  Eigen::Array<PetscInt, Eigen::Dynamic, 1>::ConstSegmentReturnType
  cell_dofs(int cell) const
  {
    return _dofmap.links(cell);
  }

  /// Extract subdofmap component
  /// @param[in] component The component indices
  /// @param[in] topology The mesh topology the the dofmap is defined on
  /// @return The dofmap for the component
  DofMap extract_sub_dofmap(const std::vector<int>& component,
                            const mesh::Topology& topology) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  /// @param[in] comm MPI Communicator
  /// @param[in] topology The meshtopology that the dofmap is defined on
  /// @return The collapsed dofmap
  std::pair<std::unique_ptr<DofMap>, std::vector<std::int32_t>>
  collapse(MPI_Comm comm, const mesh::Topology& topology) const;

  /// Get dofmap data
  /// @return The adjacency list with dof indices for each cell
  const graph::AdjacencyList<PetscInt>& list() const { return _dofmap; }

  /// Layout of dofs on an element
  std::shared_ptr<const ElementDofLayout> element_dof_layout;

  /// Index map that described the parallel distribution of the dofmap
  std::shared_ptr<const common::IndexMap> index_map;

private:
  // Cell-local-to-dof map (dofs for cell dofmap[i])
  graph::AdjacencyList<PetscInt> _dofmap;
};
} // namespace fem
} // namespace dolfinx
