// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <span>
#include <vector>

/// @brief Functions for the re-ordering of input mesh topology to the
/// DOLFINx ordering, and transpose orderings for file output.
///
/// Basix ordering is used for the geometry nodes, and is shown below
/// for a range of cell types.
/*!
@verbatim
Triangle:               Triangle6:           Triangle10:
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
2-----------3          2-----7-----3           2--10--11---3
|           |          |           |           |           |
|           |          |           |           7  14  15   9
|           |          5     8     6           |           |
|           |          |           |           6  12  13   8
|           |          |           |           |           |
0-----------1 --> u    0-----4-----1           0---4---5---1


Tetrahedron:                 Tetrahedron10:          Tetrahedron20
            v
           /
          2                            2                         2
        ,/|`\                        ,/\`\                     ,/|`\
      ,/   \ `\                    ,/   \ `\.                13  |  `9
     ,/    '.   `\                ,8    '.   `6           ,/     4   `\
   ,/       |     `\            ,/       4     `\       12    19 |     `8
 ,/         |       `\        ,/         |       `\   ,/         |       `\
0-----------'.--------1 -> u 0--------9--'.--------1 0-----14----'.--15----1
 `\.         |      ,/        `\.         |      ,/   `\.  17     |  16 ,/
    `\.      |    ,/             `\.      |    ,5       10.   18  5   ,6
       `\.   '. ,/                  `7.   '. ,/            `\.    .  7
          `\. |/                       `\. |/                 11. |/
             `3                            `3                    `3
                `\.
                   w


Hexahedron:                Hexahedron27:
        w
    6----------7               6----19----7
   /|   ^   v /|              /|         /|
  / |   |  / / |            17 |  25    18|
 /  |   | / /  |            / 14    24 /  15
4----------5   |           4----16----5   |
|   |   +--|---|--> u      |22 |  26  | 23|
|   2------+---3           |   2----13+---3
|  /       |  /           10  / 21   12  /
| /        | /             | 9    20  | 11
|/         |/              |/         |/
0----------1               0-----8----1


Prism:                       Prism18:
            w
            ^
            |
            3                       3
          ,/|`\                   ,/|`\
        ,/  |  `\               12  |  13
      ,/    |    `\           ,/    |    `\
     4------+------5         4------14-----5
     |      |      |         |      8      |
     |    ,/|`\    |         | 15   |  16  |
     |  ,/  |  `\  |         |      |      |
     |,/    0    `\|        10      0  17  11
   ./|    ,/ `\    |\        |    ,/ `\    |
  /  |  ,/     `\  | `\      |  ,6     `7  |
u    |,/         `\|   v     |,/         `\|
     1-------------2         1------9------2


Pyramid:                      Pyramid14:
               4                             4
             ,/|\                          ,/|\
           ,/ .'|\                       ,/ .'|\
  v      ,/   | | \                    ,/   | | \
   \.  ,/    .' | `.                 ,/    .' | `.
     \.      |  '.  \              11      |  12  \
   ,/  \.  .'  w |   \           ,/       .'   |   \
 ,/      \. |  ^ |    \        ,/         7    |    9
2----------\'--|-3    `.      2-------10-.'----3    `.
 `\       |  \.|  `\    \      `\        |      `\    \
   `\     .'   +----`\ - \ -> u  `6     .'  13      8   \
     `\   |           `\  \        `\   |           `\  \
       `\.'             ` `\         `\.'             ` `\
          0-----------------1           0--------5--------1
@endverbatim
*/
namespace dolfinx::io::cells
{

/// @brief Get the Lagrange order of a given cell with a given number of
/// nodes.
///
/// @param[in] type Cell shape.
/// @param[in] num_nodes Number of cell 'nodes'
/// @return Lagrange order of the cell type.
int cell_degree(mesh::CellType type, int num_nodes);

/// @brief Permutation array to map from VTK to DOLFINx node ordering.
///
/// @param[in] type Cell shape.
/// @param[in] num_nodes Number of cell 'nodes'
/// @return Permutation array `p` for permuting from VTK ordering to
/// DOLFINx ordering, i.e. `a_dolfin[i] = a_vtk[p[i]]`.
///
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p =
/// [a[p[0]], a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`.
std::vector<std::uint16_t> perm_vtk(mesh::CellType type, int num_nodes);

/// @brief Permutation array to map from Gmsh to DOLFINx node ordering.
///
/// @param[in] type Cell shape.
/// @param[in] num_nodes Number of cell 'nodes'
/// @return Permutation array `p` for permuting from Gmsh ordering to
/// DOLFINx ordering, i.e. `a_dolfin[i] = a_gmsh[p[i]]`.
///
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p
/// =[a[p[0]], a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`.
std::vector<std::uint16_t> perm_gmsh(mesh::CellType type, int num_nodes);

/// @brief Compute the transpose of a re-ordering map.
///
/// @param[in] map A re-ordering map.
/// @return Transpose of the `map`. E.g., is `map = {1, 2, 3, 0}`, the
/// transpose will be `{3 , 0, 1, 2 }`.
std::vector<std::uint16_t> transpose(std::span<const std::uint16_t> map);

/// @brief Permute cell topology by applying a permutation array for
/// each cell.
///
/// @param[in] cells Array of cell topologies, with each row
/// representing a cell (row-major storage).
/// @param[in] shape Shape of the `cells` array.
/// @param[in] p Permutation array that maps `a_p[i] = a[p[i]]`, where
/// `a_p` is the permuted array.
/// @return Permuted cell topology, where for a cell `v_new[i] =
/// v_old[map[i]]`. The storage is row-major and the shape is the same
/// as `cells`.
std::vector<std::int64_t> apply_permutation(std::span<const std::int64_t> cells,
                                            std::array<std::size_t, 2> shape,
                                            std::span<const std::uint16_t> p);

/// @brief Get VTK cell identifier.
///
/// @param[in] cell Cell type.
/// @param[in] dim Topological dimension of the cell.
/// @return VTK cell identifier.
std::int8_t get_vtk_cell_type(mesh::CellType cell, int dim);

/// @brief Get DOLFINx cell type and degree from VTK cell type.
///
/// @param[in] vtk_cell_type VTK cell type identifier.
/// @return Return the cell type and degree. If arbitrary order Lagragian cell
/// from VTK is supplied, return -1 for the degree.
inline std::tuple<mesh::CellType, std::int8_t>
vtk_to_dolfinx(std::int8_t vtk_cell_type)
{
  {
    // For a complete overview of VTK cell types, see
    // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    switch (vtk_cell_type)
    {
      using enum mesh::CellType;
    case 1:
      return {point, -1};
    case 3:
      return {interval, 1};
    case 5:
      return {triangle, 1};
    case 9:
      return {quadrilateral, 1};
    case 10:
      return {tetrahedron, 1};
    case 12:
      return {hexahedron, 1};
    case 13:
      return {prism, 1};
    case 14:
      return {pyramid, 1};
    case 21:
      return {interval, 2};
    case 22:
      return {triangle, 2};
    case 23:
      return {quadrilateral, 2};
    case 24:
      return {tetrahedron, 2};
    case 25:
      return {hexahedron, 2};
    case 26:
      return {prism, 2};
    case 27:
      return {pyramid, 2};
    case 35:
      return {interval, 3};
    case 68:
      return {interval, -1};
    case 69:
      return {triangle, -1};
    case 70:
      return {quadrilateral, -1};
    case 71:
      return {tetrahedron, -1};
    case 72:
      return {hexahedron, -1};
    case 73:
      return {prism, -1};
    case 74:
      return {pyramid,
              -1}; // Not implemented in VTK yet, but added as placeholder.
    default:
      break;
    }
    throw std::runtime_error("Unknown VTK cell type");
  }
}

} // namespace dolfinx::io::cells
