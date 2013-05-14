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
// Last changed: 2013-05-09

#ifndef __DOLFIN_HDF5UTILITY_H
#define __DOLFIN_HDF5UTILITY_H

#ifdef HAS_HDF5

#include <string>
#include <vector>

namespace dolfin
{
  class Mesh;

  /// This class contains some algorithms which do not explicitly
  /// depend on the HDF5 file format, mostly to do with reorganising
  /// Mesh entities with MPI

  class HDF5Utility
  {
  public:

    /// Get mapping of cells in the assigned global range of the
    /// current process to remote process and remote local index.
    static void compute_global_mapping(std::vector<std::pair<std::size_t,
                                       std::size_t> >& global_owner,
                                       const Mesh& mesh);

    /// Convert LocalMeshData structure to a Mesh, used when running
    /// in serial
    static void build_local_mesh(Mesh& mesh, const LocalMeshData& mesh_data);

    /// Reorder vertices into global index order, so they can be saved
    /// correctly for HDF5 mesh output
    static std::vector<double>
      reorder_vertices_by_global_indices(const Mesh& mesh);

    /// Reorder data values of type double into global index order
    /// Shape of 2D array is given in global_size
    static void reorder_values_by_global_indices(const Mesh& mesh,
                               std::vector<double>& data,
                               std::vector<std::size_t>& global_size);

  };

}

#endif
#endif
