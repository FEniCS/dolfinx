// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <vector>

namespace dolfinx::io::cells
{
/// For simplices the FEniCS ordering follows the UFC convention, see:
/// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
/// non-simplices (quadrilaterals and hexahedrons) a tensor product
/// ordering, as specified in FIAT, is used.

/// Map from VTK node indices to DOLFINX node indicies
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map p from the position i in the VTK array to position p[i]
///   = j in the  DOLFINX array, i.e. a_dolfin[p[i]] = a_vtk[i].
///
///   If p = [0, 2, 1, 3] and a = [10, 3, 4, 7], then a_p = [10, 4, 3,
///   7]
std::vector<std::uint8_t> vtk_to_dolfin(mesh::CellType type, int num_nodes);

/// Map from DOLFINX local indices to VTK local indices. It is the
/// transpose of @pvtk_to_dolfin
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map p from the position i in the DOLFINX array to position
///   p[i] = j in the VTK array, i.e. a_vtk[p[i]] = a_dolfin[i].
std::vector<std::uint8_t> dolfin_to_vtk(mesh::CellType type, int num_nodes);

/// Re-order a collection of cell connections by applying a permutation
/// array
/// @param[in] cells Array of cell topologies, with each row
///     representing a cell
/// @param[in] permutation The permutation array to map to
/// @return Permted cell topology, where for a cell
///     v_new[permutation[i]] = v_old[i]
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
permute_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& permutation);

} // namespace dolfinx::io::cells
