// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <mpi.h>
#include <tuple>
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
class ElementDofLayout;

/// Builds a DofMap on a mesh::Mesh

class DofMapBuilder
{

public:
  /// Build dofmap
  /// @param[in] comm MPI communicator
  /// @param[in] topology The mesh topology
  /// @param[in] element_dof_layout The element dof layout for the function
  /// space
  /// @param[in] transpose_blocks Indicates whether or not the DOF blocks should
  /// be transposed from
  ///    XYZXYZ ordering to XXYYZZ ordering. This should be true for
  ///    VectorElements and TensorElements, and false for MixedElements. For
  ///    other elements, it's value should have no effect.
  static std::tuple<std::shared_ptr<const ElementDofLayout>,
                    std::shared_ptr<const common::IndexMap>,
                    graph::AdjacencyList<std::int32_t>>
  build(MPI_Comm comm, const mesh::Topology& topology,
        std::shared_ptr<const ElementDofLayout> element_dof_layout,
        const bool transpose_blocks = true);

  /// Build dofmap
  /// Build dofmap
  /// @param[in] comm MPI communicator
  /// @param[in] topology The mesh topology
  /// @param[in] element_dof_layout The element dof layout for the function
  /// space
  /// @param[in] block_size The number of DOF colocated at each point
  /// @param[in] transpose_blocks Indicates whether or not the DOF blocks should
  /// be transposed from
  ///    XYZXYZ ordering to XXYYZZ ordering. This should be true for
  ///    VectorElements and TensorElements, and false for MixedElements. For
  ///    other elements, it's value should have no effect.
  static std::pair<std::shared_ptr<common::IndexMap>,
                   graph::AdjacencyList<std::int32_t>>
  build(MPI_Comm comm, const mesh::Topology& topology,
        const ElementDofLayout& element_dof_layout, int block_size,
        const bool transpose_blocks = true);
};
} // namespace fem
} // namespace dolfinx
