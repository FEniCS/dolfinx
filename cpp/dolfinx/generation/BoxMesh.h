// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

/// Right cuboid mesh creation
namespace dolfinx::generation::BoxMesh
{

/// Create a uniform mesh::Mesh over the rectangular prism spanned by the
/// two points @p p. The order of the two points is not important in
/// terms of minimum and maximum coordinates. The total number of
/// vertices will be `(n[0] + 1)*(n[1] + 1)*(n[2] + 1)`. For tetrahedra
/// there will be  will be `6*n[0]*n[1]*n[2]` cells. For hexahedra the
/// number of cells will be `n[0]*n[1]*n[2]`.
///
/// @param[in] comm MPI communicator to build mesh on
/// @param[in] p Points of box
/// @param[in] n Number of cells in each direction.
/// @param[in] celltype Cell shape
/// @param[in] ghost_mode Ghost mode
/// @param[in] partitioner Partitioning function to use for
/// determining the parallel distribution of cells across MPI ranks
/// @return Mesh
mesh::Mesh create(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
                  std::array<std::size_t, 3> n, mesh::CellType celltype,
                  mesh::GhostMode ghost_mode,
                  const mesh::CellPartitionFunction& partitioner
                  = mesh::create_cell_partitioner());
} // namespace dolfinx::generation::BoxMesh
