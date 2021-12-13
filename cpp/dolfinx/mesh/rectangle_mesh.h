// Copyright (C) 2005-2021 Anders Logg, Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <mpi.h>

namespace dolfinx::mesh
{

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
} // namespace dolfinx::mesh

/// Rectangle mesh creation
namespace dolfinx::mesh::rectangle_mesh
{
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
/// @param[in] ghost_mode Mesh ghosting mode
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
mesh::Mesh create(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
                  std::array<std::size_t, 2> n, mesh::CellType celltype,
                  mesh::GhostMode ghost_mode,
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
/// @param[in] ghost_mode Mesh ghosting mode
/// @param[in] partitioner Partitioning function to use for determining
/// the parallel distribution of cells across MPI ranks
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
mesh::Mesh create(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
                  std::array<std::size_t, 2> n, mesh::CellType celltype,
                  mesh::GhostMode ghost_mode,
                  const mesh::CellPartitionFunction& partitioner,
                  DiagonalType diagonal = DiagonalType::right);
} // namespace dolfinx::generation::rectangle_mesh
