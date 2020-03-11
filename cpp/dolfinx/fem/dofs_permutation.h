// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/mesh/cell_types.h>
#include <vector>

namespace dolfinx
{

namespace mesh
{
class Topology;
} // namespace mesh

namespace fem
{
class ElementDofLayout;

/// Return the dof permutations for all cells. Each row contains the
/// numbers from 0 to (number of dofs on reference - 1) permuted so that
/// edges are oriented towards the higher global vertex index
/// @param[in] topology The mesh topology
/// @param[in] dof_layout The layout of dofs on a each cell
/// @return The permutations
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_dof_permutations(const mesh::Topology& topology,
                         const fem::ElementDofLayout& dof_layout);

} // namespace fem
} // namespace dolfinx
