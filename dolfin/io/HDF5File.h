// Copyright (C) 2012 Chris N. Richardson
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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-05-22
// Last changed: 2012-09-28

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/types.h"
#include "GenericFile.h"
#include "HDF5Interface.h"

namespace dolfin
{

  class Function;
  class GenericVector;
  class Mesh;

  class HDF5File: public GenericFile
  {
  public:

    /// Constructor
    HDF5File(const std::string filename);

    /// Destructor
    ~HDF5File();

    /// Write vector to file in HDF5 folder 'Vector'. Multiple calls
    /// will save in the same file with incrementing dataset names
    void operator<< (const GenericVector& x);

    /// Read vector from file in HDF5 folder 'Vector' for dataset 0
    void operator>> (GenericVector& x);

    /// Write Mesh to file
    void operator<< (const Mesh& mesh);

    /// Write Mesh to file. 'true_topology_indices' indicates
    /// whether the true vertex indices should be used for the connectivity
    /// or the position of the vertex in the list. The latter is required
    /// for visualisation and the former for reading a Mesh from file.
    void write_mesh(const Mesh& mesh, bool true_topology_indices=true);

    /// Read Mesh from file
    void operator>> (Mesh& mesh);

  private:

    // Friend
    friend class XDMFFile;

    // Create an empty HDF5 file (truncate if existing)
    void create();

    // Write data contiguously from each process in parallel into a 2D array
    // data contains local portion of data vector
    // width is the second dimension of the array (e.g. 3 for xyz data)
    // data in XYZXYZXYZ order
    template <typename T>
    void write_data(const std::string dataset_name, const std::vector<T>& data,
                    const uint width)
    {
      // Checks on width and size of data
      dolfin_assert(width != 0);
      dolfin_assert(data.size() % width == 0);
      const uint num_items = data.size()/width;

      const uint offset = MPI::global_offset(num_items, true);
      std::pair<uint, uint> range(offset, offset + num_items);
      HDF5Interface::write(filename, dataset_name, data, range, width);
    }

    // Search through list of dataset names for one beginning with
    // search_term
    std::string search_list(const std::vector<std::string>& list,
                            const std::string& search_term) const;

    // Generate HDF5 dataset names for mesh topology and coordinates
    std::string mesh_coords_dataset_name(const Mesh& mesh) const;
    std::string mesh_index_dataset_name(const Mesh& mesh) const;
    std::string mesh_topology_dataset_name(const Mesh& mesh) const;

    // Reorganise data into global order as defined by global_index
    // global_index contains the global index positions
    // local_vector contains the items to be redistributed
    // global_vector is the result: the local part of the new global vector created.
    template <typename T>
    void redistribute_by_global_index(const std::vector<uint>& global_index,
                                      const std::vector<T>& local_vector,
                                      std::vector<T>& global_vector);

  };

}
#endif
#endif
