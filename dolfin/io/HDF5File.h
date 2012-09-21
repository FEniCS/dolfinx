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
// Last changed: 2012-09-21

#ifndef __DOLFIN_HDF5FILE_H
#define __DOLFIN_HDF5FILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/types.h"
#include "GenericFile.h"

#define H5_USE_16_API
#include <hdf5.h>

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

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

    /// Write vector to file
    /// saves into HDF5 folder 'Vector'
    /// multiple calls will save in the same file
    /// with incrementing dataset names
    void operator<< (const GenericVector& x);

    /// Read vector from file
    /// looks in HDF5 folder 'Vector' for last dataset
    void operator>> (GenericVector& x);

    /// Write Mesh to file
    void operator<< (const Mesh& mesh);

    /// Write Mesh to file. 'true_topology_indices' indicares
    /// whether the true vertex indices should be used for the connectivity
    /// or the position of the vertex in the list. The latter is required
    /// for visualisation and the former for reading a Mesh from file.
    void write_mesh(const Mesh& mesh, bool true_topology_indices=true);

    /// Read Mesh from file
    void operator>> (Mesh& mesh);

  private:

    // Friend
    friend class XDMFFile;

    // Create an empty file (truncate if existing)
    void create();

    // Write data to existing HDF file contiguously from each process,
    // the range being set by the data size
    // width: is the width of the data item (dim 1, e.g. 3 for x, y, z data)
    void write(const std::vector<double>& data,
               const std::string dataset_name, const uint width);

    void write(const std::vector<uint>& data,
               const std::string dataset_name, const uint width);

    void write(const std::vector<int>& data,
               const std::string dataset_name, const uint width);

    // Write data to existing HDF file as 
    // defined by range blocks on each process
    // range: the local range on this processor (dim 0)
    // width: is the width of the data item (dim 1, e.g. 3 for x, y, z data)
    void write(const std::vector<double>& data,
               const std::pair<uint, uint> range,
               const std::string dataset_name, const uint width);

    void write(const std::vector<uint>& data,
               const std::pair<uint, uint> range,
               const std::string dataset_name, const uint width);

    void write(const std::vector<int>& data,
               const std::pair<uint, uint> range,
               const std::string dataset_name, const uint width);

    // Write data to existing HDF file as defined by range blocks on each
    // process
    // range: the local range on this processor (dim 0)
    // width: is the width of the data item (dim 1, e.g. 3 for x, y, z data)
    template <typename T>
    void write(const std::vector<T>& data,
               const std::pair<uint, uint> range,
               const std::string dataset_name,
               const int h5type, const uint width) const;

    // Read from HDF5 file into data as defined by range blocks
    // range: the local range on this processor (dim 0)
    // width: is the width of the data item (dim 1, e.g. 3 for x, y, z data)
    template <typename T>
    void read(std::vector<T>& data, const std::pair<uint, uint> range,
              const std::string dataset_name, const int h5type,
              const uint width) const;

    // Get dimensions of 2D dataset
    std::pair<uint, uint> dataset_dimensions(const std::string dataset_name) const;

    // List of all datasets in a group
    std::vector<std::string> dataset_list(const std::string group_name) const;

    // Check existence of dataset in file
    bool dataset_exists(const std::string dataset_name) const;

    // Add an unsigned integer attribute to a dataset
    void add_attribute(const std::string dataset_name,
                       const std::string attribute_name,
                       const uint attribute_value);
    
    // Add a string attribute to a dataset
    void add_attribute(const std::string dataset_name,
                       const std::string attribute_name,
                       const std::string attribute_value);
    
    // Get a string attribute of a dataset
    void get_attribute(const std::string dataset_name,
                       const std::string attribute_name,
                       std::string &attribute_value) const;

    // Get a uint attribute of a dataset
    void get_attribute(const std::string dataset_name,
                       const std::string attribute_name,
                       uint &attribute_value) const;

    // Generate HDF5 dataset names for mesh topology and coordinates
    std::string mesh_coords_dataset_name(const Mesh& mesh) const;
    std::string mesh_index_dataset_name(const Mesh& mesh) const;
    std::string mesh_topology_dataset_name(const Mesh& mesh) const;

    // Return a HDF5 file descriptor suitable for parallel access
    hid_t HDF5File::open_parallel_file() const;
    

  };

}
#endif
#endif
