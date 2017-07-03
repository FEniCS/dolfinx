// Copyright (C) 2013 Chris N. Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
//
// First added:  2013-05-07
// Last changed: 2013-06-18

#ifndef __DOLFIN_HDF5UTILITY_H
#define __DOLFIN_HDF5UTILITY_H

#include <string>
#include <vector>

#include "dolfin/common/types.h"

namespace dolfin
{
  class LocalMeshData;
  class GenericDofMap;
  class Mesh;

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
                       const std::vector<dolfin::la_index>& input_cell_dofs,
                       const std::vector<std::size_t>& x_cell_dofs,
                       const std::pair<dolfin::la_index, dolfin::la_index>
                       vector_range,
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
      std::pair<std::size_t, std::size_t> vector_range,
      const GenericDofMap& dofmap,
      std::vector<dolfin::la_index>& global_dof);

    /// Get cell owners for an arbitrary set of cells.
    /// Returns (process, local index) pairs
    static std::vector<std::pair<std::size_t, std::size_t>>
      cell_owners(const Mesh& mesh, const std::vector<std::size_t>& cells);

    /// Get mapping of cells in the assigned global range of the
    /// current process to remote process and remote local index.
    static void cell_owners_in_range(
      std::vector<std::pair<std::size_t, std::size_t>>& global_owner,
      const Mesh& mesh);

    /// Convert LocalMeshData structure to a Mesh, used when running
    /// in serial
    static void build_local_mesh(Mesh& mesh, const LocalMeshData& mesh_data);

    /// Missing docstring
    static void set_local_vector_values(
      MPI_Comm mpi_comm,
      GenericVector& x,
      const Mesh& mesh,
      const std::vector<size_t>& cells,
      const std::vector<dolfin::la_index>& cell_dofs,
      const std::vector<std::size_t>& x_cell_dofs,
      const std::vector<double>& vector,
      std::pair<dolfin::la_index, dolfin::la_index> input_vector_range,
      const GenericDofMap& dofmap);
  };

}

#endif
