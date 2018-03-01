// Copyright (C) 2013 Chris N. Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfin/common/types.h"
#include <array>
#include <string>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScVector;
}

namespace fem
{
class GenericDofMap;
}

namespace mesh
{
class Mesh;
}

namespace io
{

/// This class contains some algorithms which do not explicitly
/// depend on the HDF5 file format, mostly to do with reorganising
/// Mesh entities with MPI

class HDF5Utility
{
public:
  /// Generate two vectors, in the range of "vector_range"
  /// of the global DOFs.
  /// global_cells is a list of cells which point to the DOF (non-unique)
  /// and remote_local_dofi is the pertinent local_dof of the cell.
  /// input_cells is a list of cells held on this process, and
  /// input_cell_dofs/x_cell_dofs list their local_dofs.
  static void
  map_gdof_to_cell(const MPI_Comm mpi_comm,
                   const std::vector<std::size_t>& input_cells,
                   const std::vector<dolfin::la_index_t>& input_cell_dofs,
                   const std::vector<std::int64_t>& x_cell_dofs,
                   const std::array<std::int64_t, 2> vector_range,
                   std::vector<std::size_t>& global_cells,
                   std::vector<std::size_t>& remote_local_dofi);

  /// Given the cell dof index specified
  /// as (process, local_cell_index, local_cell_dof_index)
  /// get the global_dof index from that location, and return it for all
  /// DOFs in the range of "vector_range"
  static void get_global_dof(
      MPI_Comm mpi_comm,
      const std::vector<std::pair<std::size_t, std::size_t>>& cell_ownership,
      const std::vector<std::size_t>& remote_local_dofi,
      std::array<std::int64_t, 2> vector_range,
      const fem::GenericDofMap& dofmap,
      std::vector<dolfin::la_index_t>& global_dof);

  /// Get cell owners for an arbitrary set of cells.
  /// Returns (process, local index) pairs
  static std::vector<std::pair<std::size_t, std::size_t>>
  cell_owners(const mesh::Mesh& mesh, const std::vector<std::size_t>& cells);

  /// Get mapping of cells in the assigned global range of the
  /// current process to remote process and remote local index.
  static void cell_owners_in_range(
      std::vector<std::pair<std::size_t, std::size_t>>& global_owner,
      const mesh::Mesh& mesh);

  /// Missing docstring
  static void
  set_local_vector_values(MPI_Comm mpi_comm, la::PETScVector& x,
                          const mesh::Mesh& mesh,
                          const std::vector<size_t>& cells,
                          const std::vector<dolfin::la_index_t>& cell_dofs,
                          const std::vector<std::int64_t>& x_cell_dofs,
                          const std::vector<double>& vector,
                          std::array<std::int64_t, 2> input_vector_range,
                          const fem::GenericDofMap& dofmap);
};
}
}