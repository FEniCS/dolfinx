// Copyright (C) 2010 Garth N. Wells
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
// First added:  2012-09-21
// Last changed:

#ifndef __HDF5_INTERFACE_H
#define __HDF5_INTERFACE_H

// Use 1.6 API for stability
// Could update to latest version
// requires adding a few extra arguments to calls
// for little obvious benefit
#define H5_USE_16_API
#include <hdf5.h>

namespace dolfin
{

  class HDF5Interface
  {
  public:
    static void create(const std::string &filename);
    
    // Write double dataset
    static void write(const std::string &filename,
                      const std::string dataset_name,
                      const std::vector<double>& data,
                      const std::pair<uint, uint> range,
                      const uint width);

    // Write int dataset
    static void write(const std::string &filename,
                      const std::string dataset_name,
                      const std::vector<int>& data,
                      const std::pair<uint, uint> range,
                      const uint width);

    // Write uint dataset
    static void write(const std::string &filename,
                      const std::string dataset_name, 
                      const std::vector<uint>& data,
                      const std::pair<uint, uint> range,
                      const uint width);

    // Read double dataset
    static void read(const std::string &filename,
                     const std::string dataset_name,
                     std::vector<double>& data, 
                     const std::pair<uint, uint> range,
                     const uint width);

    // Read uint dataset
    static void read(const std::string &filename,
                     const std::string dataset_name,
                     std::vector<uint>& data,
                     const std::pair<uint, uint> range,
                     const uint width);
    
    // Check for existence of dataset in file
    static bool dataset_exists(const std::string &filename,const std::string &dataset_name);
    
    // List all datasets in named group of file
    static std::vector<std::string> dataset_list(const std::string &filename,
                                                 const std::string &group_name);
    
    // Get dimensions (NX,NY) of 2D dataset
    static std::pair<uint, uint> dataset_dimensions(const std::string &filename,
                                                    const std::string &dataset_name);
    
    // Get a named attribute of a dataset
    template <typename T>
    static void get_attribute(const std::string &filename,
                              const std::string &dataset_name,
                              const std::string &attribute_name,
                              T &attribute_value);

    // Generic add attribute to dataset
    template <typename T>
    static void add_attribute(const std::string &filename,
                              const std::string &dataset_name,
                              const std::string &attribute_name,
                              const T &attribute_value);
        
  private:

    // Open HDF5 file descriptor in parallel - a common operation
    static hid_t open_parallel_file(const std::string &filename);
    
    // Internal uint-specific code to add an attribute
    static void _add_attribute_value(const hid_t &dset_id,
                                    const std::string &attribute_name, 
                                    const uint &attribute_value);

    // Internal vector<uint>-specific code to add an attribute
    static void _add_attribute_value(const hid_t &dset_id,
                                     const std::string &attribute_name, 
                                     const std::vector<uint> &attribute_value);

    // Internal string-specific code to add an attribute
    static void _add_attribute_value(const hid_t &dset_id,
                                    const std::string &attribute_name, 
                                    const std::string &attribute_value);

    // Internal uint-specific code to get an attribute
    static void _get_attribute_value(const hid_t &attr_type,
                                     const hid_t &attr_id, 
                                     uint &attribute_value);

    // Internal vector<uint>-specific code to get an attribute
    static void _get_attribute_value(const hid_t &attr_type,
                                     const hid_t &attr_id, 
                                     std::vector<uint> &attribute_value);

    // Internal string-specific code to get an attribute
    static void _get_attribute_value(const hid_t &attr_type,
                                     const hid_t &attr_id, 
                                     std::string &attribute_value);

    // Generic read from a dataset - given range and width, type.
    template <typename T>
    static void read(const std::string &filename,
                     const std::string dataset_name,
                     std::vector<T>& data, const std::pair<uint, uint> range,
                     const int h5type, const uint width);
    
    // Generic write to a dataset - given range and width, type.
    template <typename T>
    static void write(const std::string &filename,
                      const std::string dataset_name,
                      const std::vector<T>& data,
                      const std::pair<uint, uint> range,
                      const int h5type, const uint width);

  };

}

#endif
