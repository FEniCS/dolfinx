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
/*
    For simplices the FEniCS ordering follows the UFC convention, see:
    https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
    non-simplices (quadrilaterals and hexahedrons) a tensor product
    ordering, as specified in FIAT, is used.
    The dolfin-ordering is:
    Triangle:               Triangle6:          Triangle10:
    v
    ^
    |
    2                       2                    2
    |`\                     |`\                  | \
    |  `\                   |  `\                6   4
    |    `\                 4    `3              |     \
    |      `\               |      `\            5   9   3
    |        `\             |        `\          |         \
    0----------1 --> u      0-----5----1         0---7---8---1

    Quadrilateral:         Quadrilateral9:         Quadrilateral16:
    v
    ^
    |
    1-----------3          1-----7-----4           1---9--13---5
    |           |          |           |           |           |
    |           |          |           |           3  11  15   7
    |           |          2     8     5           |           |
    |           |          |           |           2  10  14   6
    |           |          |           |           |           |
    0-----------2 --> u    0-----6-----3           0---8--12---4

    Tetrahedron:                    Tetrahedron10:               Tetrahedron20
                v
               /
              2                            2                         2
            ,/|`\                        ,/|`\                     ,/|`\
          ,/  |  `\                    ,/  |  `\                 13  |  `9
         ,/    '.   `\                ,8    '.   `6           ,/     4   `\
       ,/       |     `\            ,/       5     `\       12    19 |     `8
     ,/         |       `\        ,/         |       `\   ,/         |       `\
    0-----------'.--------1 -> u 0--------9--'.--------1 0-----14----'.--15----1
     `\.         |      ,/        `\.         |      ,/   `\.  17     |  16 ,/
        `\.      |    ,/             `\.      |    ,4       10.   18 5    ,6
           `\.   '. ,/                  `7.   '. ,/            `\.   '.  7
              `\. |/                       `\. |/                 11. |/
                 `3                            `3                    `3
                    `\.
                       w

    Hexahedron:          Hexahedron27:
           v
    2----------6           3----21----12
    |\     ^   |\          |\         |\
    | \    |   | \         | 5    23  | 14
    |  \   |   |  \        6  \ 24    15 \
    |   3------+---7       |   4----22+---13
    |   |  +-- |-- | -> u  | 8 |  26  | 17|
    0---+---\--4   |       0---+18----9   |
     \  |    \  \  |        \  7     25\ 16
      \ |     \  \ |         2 |   20   11|
       \|      w  \|          \|         \|
       1----------5           1----19----10
*/
/// Determine the degree fo the cell given the type and number of nodes
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The degree of the cell
int cell_degree(mesh::CellType type, int num_nodes);

/// Map from VTK node indices to DOLFINX node indicies
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map from local VTK index to the DOLFINX local index, i.e.
/// map[i] is the position of the ith VTK index in the DOLFINX ordering
std::vector<std::uint8_t> vtk_to_dolfin(mesh::CellType type, int num_nodes);

/// Map from DOLFINX local indices to VTK local indices. It is the
/// transpose of vtk_to_dolfin
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map from local DOLFINX index to the VTK local index, i.e.
/// map[i] is the position of the ith DOLFINX index in the VTK ordering
std::vector<std::uint8_t> dolfin_to_vtk(mesh::CellType type, int num_nodes);

/// Map from GMSH local indices to DOLFINX local indices. It is the
/// transpose of dolfin_to_gmsh
/// @param[in] type The gmsh cell type
/// @return Map from local GMSH index to the DOLFINX local index, i.e.
/// map[i] is the position of the ith GMSH index in the DOLFINX ordering
std::vector<std::uint8_t> gmsh_to_dolfin(std::string type);

/// Map from DOLFINX local indices to GMSH local indices. It is the
/// transpose of gmsh_to_dolfin
/// @param[in] type The gmsh cell type
/// @return Map from local DOLFINX index to the GMSH local index, i.e.
/// map[i] is the position of the ith DOLFINX index in the GMSH ordering
std::vector<std::uint8_t> dolfin_to_gmsh(std::string type);

/// Re-order a collection of cell connections by applying a permutation
/// array
/// @param[in] cells Array of cell topologies, with each row
///     representing a cell
/// @param[in] permutation The permutation array to map to
/// @return Permuted cell topology, where for a cell
///     v_new[permutation[i]] = v_old[i]
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
permute_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& permutation);

} // namespace dolfinx::io::cells
