// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <dolfin/mesh/cell_types.h>
#include <vector>

namespace dolfin
{
namespace io
{

namespace cells
{

/// Map from DOLFIN node ordering to VTK/XDMF ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map from local DOLFIN node ordering to the VTK ordering
std::vector<std::uint8_t> dolfin_to_vtk(mesh::CellType type, int num_nodes);

/// Map from VTK ordering of a cell to tensor-product ordering. This map
/// returns the identity map for all other cells than quadrilaterals and
/// hexahedra.
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> vtk_to_tp(mesh::CellType type, int num_nodes);

/// Map from the mapping of lexicographic nodes to a tensor product
/// ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> lex_to_tp(mesh::CellType type, int num_nodes);

/// Permutation for VTK ordering to DOLFIN ordering
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
std::vector<std::uint8_t> vtk_to_dolfin(mesh::CellType type, int num_nodes);

/// Convert VTK cell ordering to FEniCS cell ordering.
/// For simplices the FEniCS ordering is following the UFC-convention, see:
/// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf
/// For non-simplices (Quadrilaterals and Hexahedrons) a TensorProduct ordering,
/// as specified in FIAT.
/// @param[in] cells array containing cell connectivities in VTK format
/// @param[in] type Celltype to the permuter
/// @return Permuted cell connectivities
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
vtk_to_dolfin_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    mesh::CellType type);

/// Convert Lexicographic cell ordering to FEniCS cell ordering. For
/// simplices the FEniCS ordering is following the UFC-convention, see:
/// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
/// non-simplices (Quadrilaterals and Hexahedrons) a TensorProduct
/// ordering, as specified in FIAT. Lexicographical ordering is defined
/// as:
///
///   *--*--*   6--7--8  y
///   |     |   |     |  ^
///   *  *  *   3  4  5  |
///   |     |   |     |  |
///   *--*--*   0--1--2  ---> x
///
/// Tensor product:
///
///   *--*--*   1--7--4  y
///   |     |   |     |  ^
///   *  *  *   2  8  5  |
///   |     |   |     |  |
///   *--*--*   0--6--3  ---> x
///
/// @param[in] cells array containing cell connectivities in VTK format
/// @param[in] type Celltype to the permuter
/// @return Permuted cell connectivities
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
lex_to_dolfin_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    mesh::CellType type);

} // namespace cells
} // namespace io
} // namespace dolfin
