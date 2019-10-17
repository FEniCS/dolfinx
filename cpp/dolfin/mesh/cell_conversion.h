// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/mesh/cell_types.h>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// Map from DOLFIN node ordering to VTK/XDMF ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map from local DOLFIN node ordering to the VTK ordering
std::vector<std::uint8_t> dolfin_to_vtk(CellType type, int num_nodes);

/// Map from VTK ordering of a cell to tensor-product ordering. This map
/// returns the identity map for all other cells than quadrilaterals and
/// hexahedrons.
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> vtk_to_tp(CellType type, int num_nodes);

/// Map from the mapping of lexicographic nodes to a tensor product
/// ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> lex_to_tp(CellType type, int num_nodes);

/// Permutation for VTK ordering to DOLFIN ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> vtk_to_dolfin(CellType type, int num_nodes);

/// Convert gmsh cell ordering to FENICS cell ordering
/// @param cells array containing cell connectivities in VTK format
/// @param type Celltype to the permuter
/// @return Permuted cell connectivities
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
gmsh_to_dolfin_ordering(
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        cells,
    CellType type);

/// Default map of DOLFIN/UFC node ordering to the cell input ordering
/// @param type Celltype to map
/// @param degree Degree e of cell
/// @return Default dolfin permutation
std::vector<std::uint8_t> default_cell_permutation(CellType type,
                                                   std::int32_t degree);

} // namespace mesh
} // namespace dolfin
