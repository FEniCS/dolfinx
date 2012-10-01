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
// Last changed: 2012-10-01

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
    HDF5File(const std::string filename, const bool use_mpiio=true);

    /// Destructor
    ~HDF5File();

    /// Write vector to file in HDF5 folder 'Vector'. Multiple calls
    /// will save in the same file with incrementing dataset names
    void operator<< (const GenericVector& x);

    /// Read vector from file in HDF5 folder 'Vector' for dataset 0
    void operator>> (GenericVector& x);

    /// Read vector from HDF5 file
    void read(const std::string dataset_name, GenericVector& x,
              const bool use_partition_from_file=true);

    /// Write Mesh to file (using true topology indices)
    void operator<< (const Mesh& mesh);

    /// Write Mesh to file. 'true_topology_indices' indicates
    /// whether the true global vertex indices should be used when saving

    /// With true_topology_indices=true
    /// ===============================
    /// Vertex coordinates are reordered into global order before saving
    /// Topological connectivity uses global indices
    /// * may exhibit poor scaling due to MPI distribute of vertex
    /// coordinates
    /// * can be read back in by any number of processes

    /// With true_topology_indices=false
    /// ================================
    /// Vertex coordinates are in local order, with an offset
    /// Topological connectivity uses the local + offset values for indexing
    /// * some duplication of vertices => larger file size
    /// * reduced MPI communication when saving
    /// * more difficult to read back in, especially if nprocs > than
    ///   when writing
    /// * efficient to read back in if nprocs is the same, and
    ///   partitioning is the same
    void write_mesh(const Mesh& mesh, bool true_topology_indices=true);

    /// Read Mesh from file
    void operator>> (Mesh& mesh);

    /// Check is dataset with given name exists
    bool dataset_exists(const std::string dataset_name) const;

  private:

    // Friend
    friend class XDMFFile;

    // Open HDF5 file
    void open_hdf5_file(bool truncate);

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
      HDF5Interface::write_data(filename, dataset_name, data, range, width,
                                mpi_io);
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


    // HDF5 file descriptor/handle
    bool hdf5_file_open;
    hid_t hdf5_file_id;

    // Parallel mode
    const bool mpi_io;

  };

}
#endif
#endif
