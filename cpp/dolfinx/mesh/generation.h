// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "cell_types.h"
#include "utils.h"
#include <array>
#include <cstddef>
#include <mpi.h>

namespace dolfinx::mesh
{

/// Create a uniform mesh::Mesh over the rectangular prism spanned by
/// the two points @p p. The order of the two points is not important in
/// terms of minimum and maximum coordinates. The total number of
/// vertices will be `(n[0] + 1)*(n[1] + 1)*(n[2] + 1)`. For tetrahedra
/// there will be  will be `6*n[0]*n[1]*n[2]` cells. For hexahedra the
/// number of cells will be `n[0]*n[1]*n[2]`.
///
/// @param[in] comm MPI communicator to build mesh on
/// @param[in] p Points of box
/// @param[in] n Number of cells in each direction
/// @param[in] celltype Cell shape
/// @param[in] partitioner Partitioning function to use for
/// determining the parallel distribution of cells across MPI ranks
/// @return Mesh
Mesh create_box(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
                std::array<std::size_t, 3> n, CellType celltype,
                const mesh::CellPartitionFunction& partitioner
                = create_cell_partitioner());

/// Interval mesh of the 1D line `[a, b]`.  Given @p n cells in the
/// axial direction, the total number of intervals will be `n` and the
/// total number of vertices will be `n + 1`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] n The number of cells
/// @param[in] x The end points of the interval
/// @param[in] partitioner Partitioning function to use for determining
/// the parallel distribution of cells across MPI ranks
/// @return A mesh
Mesh create_interval(MPI_Comm comm, std::size_t n, std::array<double, 2> x,
                     const CellPartitionFunction& partitioner
                     = create_cell_partitioner());

/// Enum for different diagonal types
enum class DiagonalType
{
  left,
  right,
  crossed,
  shared_facet,
  left_right,
  right_left
};

/// Create a uniform mesh::Mesh over the rectangle spanned by the two
/// points @p p. The order of the two points is not important in terms
/// of minimum and maximum coordinates. The total number of vertices
/// will be `(n[0] + 1)*(n[1] + 1)`. For triangles there will be  will
/// be `2*n[0]*n[1]` cells. For quadrilaterals the number of cells will
/// be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] p Two corner points
/// @param[in] n Number of cells in each direction
/// @param[in] celltype Cell shape
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
Mesh create_rectangle(MPI_Comm comm,
                      const std::array<std::array<double, 2>, 2>& p,
                      std::array<std::size_t, 2> n, CellType celltype,
                      DiagonalType diagonal = DiagonalType::right);

/// Create a uniform mesh::Mesh over the rectangle spanned by the two
/// points @p p. The order of the two points is not important in terms
/// of minimum and maximum coordinates. The total number of vertices
/// will be `(n[0] + 1)*(n[1] + 1)`. For triangles there will be  will
/// be `2*n[0]*n[1]` cells. For quadrilaterals the number of cells will
/// be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] p Two corner points
/// @param[in] n Number of cells in each direction
/// @param[in] celltype Cell shape
/// @param[in] partitioner Partitioning function to use for determining
/// the parallel distribution of cells across MPI ranks
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
Mesh create_rectangle(MPI_Comm comm,
                      const std::array<std::array<double, 2>, 2>& p,
                      std::array<std::size_t, 2> n, CellType celltype,
                      const CellPartitionFunction& partitioner,
                      DiagonalType diagonal = DiagonalType::right);
} // namespace dolfinx::mesh
