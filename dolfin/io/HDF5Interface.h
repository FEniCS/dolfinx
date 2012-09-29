// Copyright (C) 2012 Chris Richardson and Garth N. Wells
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

    /// FIXME: Add description
    static void create(const std::string filename);

    /// Write data to existing HDF file as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    static void write(const std::string filename,
                      const std::string dataset_name,
                      const std::vector<double>& data,
                      const std::pair<uint, uint> range,
                      const uint width);

    /// Write data to existing HDF file as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    static void write(const std::string filename,
                      const std::string dataset_name,
                      const std::vector<int>& data,
                      const std::pair<uint, uint> range, const uint width);

    /// Write data to existing HDF file as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    static void write(const std::string filename,
                      const std::string dataset_name,
                      const std::vector<uint>& data,
                      const std::pair<uint, uint> range, const uint width);

    /// Read data from a HDF5 dataset as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    static void read(const std::string filename,
                     const std::string dataset_name,
                     std::vector<double>& data,
                     const std::pair<uint, uint> range, const uint width);

    /// Read uint dataset
    static void read(const std::string filename,
                     const std::string dataset_name, std::vector<uint>& data,
                     const std::pair<uint, uint> range, const uint width);

    /// Check for existence of dataset in file
    static bool dataset_exists(const std::string filename,
                               const std::string dataset_name);

    /// List all datasets in named group of file
    static std::vector<std::string> dataset_list(const std::string filename,
                                                 const std::string group_name);

    /// Get dimensions (NX, NY) of 2D dataset
    static std::pair<uint, uint> dataset_dimensions(const std::string filename,
                                               const std::string dataset_name);

    /// Get a named attribute of a dataset
    template <typename T>
    static void get_attribute(const std::string filename,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              T &attribute_value);

    /// Add attribute to dataset
    template <typename T>
    static void add_attribute(const std::string filename,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              const T& attribute_value);

  private:

    // HDF5 calls to open a file descriptor on multiple processes
    // Common file opening sequence
    static hid_t open_parallel_file(const std::string filename);

    template <typename T>
    static void add_attribute_value(const hid_t dset_id,
                                    const std::string attribute_name,
                                    const T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "add attribute data",
                   "No specialised function fot this data type");
    }

    template <typename T>
    static void get_attribute_value(const hid_t attr_type,
                                    const hid_t attr_id,
                                    T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "gate attribute data",
                   "No specialised function fot this data type");
    }

    // Generic write to a dataset - given range and width, type.
    template <typename T>
    static void write(const std::string filename,
                      const std::string dataset_name,
                      const std::vector<T>& data,
                      const std::pair<uint, uint> range, const int h5type,
                      const uint width);

    // Generic read from a dataset - given range and width, type.
    template <typename T>
    static void read(const std::string filename,
                     const std::string dataset_name, std::vector<T>& data,
                     const std::pair<uint, uint> range, const int h5type,
                     const uint width);

  };

  //-----------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  #define HDF5_FAIL -1
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const uint& attribute_value)
  {
    // Add uint attribute to dataset

    // Create a scalar dataspace
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create attribute of type uint
    hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_UINT,
                                   dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_UINT, &attribute_value);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                        const std::string attribute_name,
                                        const std::vector<uint>& attribute_value)
  {
    // Create a vector dataspace
    hsize_t dimsf = attribute_value.size();
    hid_t dataspace_id = H5Screate_simple(1, &dimsf, NULL);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create an attribute of type uint in the dataspace
    hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_UINT, dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_UINT, &attribute_value[0]);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const std::string& attribute_value)
  {
    // Add string attribute to dataset

    // Create a scalar dataspace
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Copy basic string type from HDF5 types and set string length
    hid_t datatype_id = H5Tcopy(H5T_C_S1);
    herr_t status = H5Tset_size(datatype_id, attribute_value.size());
    dolfin_assert(status != HDF5_FAIL);

    // Create attribute in the dataspace with the given string
    hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(), datatype_id,
                                    dataspace_id, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    status = H5Awrite(attribute_id, datatype_id, attribute_value.c_str());
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          uint& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // Read value
    herr_t status = H5Aread(attr_id, H5T_NATIVE_UINT, &attribute_value);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          std::string& attribute_value)
  {
    // Check this attribute is a string
    dolfin_assert(H5Tget_class(attr_type) == H5T_STRING);

    // Copy string type from HDF5 types and set length accordingly
    hid_t memtype = H5Tcopy(H5T_C_S1);
    int string_length = H5Tget_size(attr_type) + 1;
    herr_t status = H5Tset_size(memtype,string_length);
    dolfin_assert(status != HDF5_FAIL);

    // FIXME: messy
    // Copy string value into temporary vector
    // std::vector::data can be copied into
    // (std::string::data cannot)
    std::vector<char> attribute_data(string_length);
    status = H5Aread(attr_id, memtype, attribute_data.data());
    dolfin_assert(status != HDF5_FAIL);

    attribute_value.assign(attribute_data.data());
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          std::vector<uint>& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // get dimensions of attribute array, check it is one-dimensional
    hid_t dataspace = H5Aget_space(attr_id);
    dolfin_assert(dataspace != HDF5_FAIL);
    hsize_t cur_size[10];
    hsize_t max_size[10];
    const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);
    dolfin_assert(ndims == 1);

    attribute_value.resize(cur_size[0]);

    // Read value to vector
    herr_t status = H5Aread(attr_id, H5T_NATIVE_UINT, attribute_value.data());
    dolfin_assert(status != HDF5_FAIL);
  }

}

#endif
