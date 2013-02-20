// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
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

#ifndef __DOLFIN_HDF5_INTERFACE_H
#define __DOLFIN_HDF5_INTERFACE_H

#ifdef HAS_HDF5

#include <vector>
#include <string>
#include <hdf5.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class HDF5File;

  /// This class wraps HDF5 function calls. HDF5 function calls should
  /// only appear in a member function of this class and not elsewhere in
  /// the library.

  class HDF5Interface
  {
  #define HDF5_FAIL -1
  public:

    /// Open HDF5 and return file descriptor
    static hid_t open_file(const std::string filename, const std::string mode,
                           const bool use_mpi_io);


    /// Flush data to file to improve data integrity after interruption
    static void flush_file(const hid_t hdf5_file_handle);

    /// Write data to existing HDF file as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    template <typename T>
    static void write_dataset(const hid_t file_handle,
                              const std::string dataset_name,
                              const std::vector<T>& data,
                              const std::pair<std::size_t, std::size_t> range,
                              const std::vector<std::size_t> global_size,
                              bool use_mpio, bool use_chunking);

    /// Read data from a HDF5 dataset as defined by range blocks on
    /// each process
    /// range: the local range on this processor
    /// width: is the width of the dataitem (e.g. 3 for x, y, z data)
    template <typename T>
    static void read_dataset(const hid_t file_handle,
                             const std::string dataset_name,
                             const std::pair<std::size_t, std::size_t> range,
                             std::vector<T>& data);

    /// Check for existence of group in HDF5 file
    static bool has_group(const hid_t hdf5_file_handle,
                          const std::string group_name);

    /// Check for existence of dataset in HDF5 file
    static bool has_dataset(const hid_t hdf5_file_handle,
                            const std::string dataset_name);

    /// Add group to HDF5 file
    static void add_group(const hid_t hdf5_file_handle,
                          const std::string dataset_name);

    /// Get dataset rank
    static std::size_t dataset_rank(const hid_t hdf5_file_handle,
                             const std::string dataset_name);

    /// Return number of data sets in a group
    static std::size_t num_datasets_in_group(const hid_t hdf5_file_handle,
                                      const std::string group_name);

    /// Get dataset size (size of each dimension)
    static std::vector<std::size_t> get_dataset_size(const hid_t hdf5_file_handle,
                                                      const std::string dataset_name);

    /// Return list all datasets in named group of file
    static std::vector<std::string> dataset_list(const hid_t hdf5_file_handle,
                                                 const std::string group_name);

    /// Get a named attribute of a dataset
    template <typename T>
    static void get_attribute(const hid_t hdf5_file_handle,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              T& attribute_value);

    /// Add attribute to dataset
    template <typename T>
    static void add_attribute(const hid_t hdf5_file_handle,
                              const std::string dataset_name,
                              const std::string attribute_name,
                              const T& attribute_value);

  private:

    template <typename T>
    static void add_attribute_value(const hid_t dset_id,
                                    const std::string attribute_name,
                                    const T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "add attribute data",
                   "No specialised function for this data type");
    }

    template <typename T>
    static void get_attribute_value(const hid_t attr_type,
                                    const hid_t attr_id,
                                    T& attribute_value)
    {
      dolfin_error("HDF5Interface.cpp",
                   "get attribute data",
                   "No specialised function for this data type");
    }

    // Return HDF5 data type
    template <typename T>
    static hid_t hdf5_type()
    {
      dolfin_error("HDF5Interface.cpp",
                   "get HDF5 primitive data type",
                   "No specialised function for this data type");
      return 0;
    }

  };

  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::write_dataset(const hid_t file_handle,
                                           const std::string dataset_name,
                                           const std::vector<T>& data,
                                           const std::pair<std::size_t, std::size_t> range,
                                           const std::vector<std::size_t> global_size,
                                           bool use_mpi_io, bool use_chunking)
  {
    // Data rank
    const std::size_t rank = global_size.size();
    dolfin_assert(rank != 0);

    if (rank > 2)
    {
      dolfin_error("HDF5Interface.cpp",
                   "write dataset to HDF5 file",
                   "Only rank 1 and rank 2 datsset are supported");
    }

    // Get HDF5 data type
    const int h5type = hdf5_type<T>();

    // Hyperslab selection parameters
    std::vector<hsize_t> count(global_size.begin(), global_size.end());
    count[0] = range.second - range.first;

    // Data offsets
    std::vector<hsize_t> offset(rank, 0);
    offset[0] = range.first;

    // Dataset dimensions
    const std::vector<hsize_t> dimsf(global_size.begin(), global_size.end());

    // Check sizes
    dolfin_assert(MPI::sum(count[0]) == global_size[0]);

    // Generic status report
    herr_t status;

    // Create a global data space
    const hid_t filespace0 = H5Screate_simple(rank, dimsf.data(), NULL);
    dolfin_assert(filespace0 != HDF5_FAIL);

    // Set chunking parameters
    hid_t chunking_properties;
    if (use_chunking)
    {
      // Set chunk size and limit to 1kB min/1MB max
      hsize_t chunk_size = dimsf[0]/2;
      if (chunk_size > 1048576)
        chunk_size = 1048576;
      if (chunk_size < 1024)
        chunk_size = 1024;

      hsize_t chunk_dims[2] = {chunk_size, dimsf[1]};
      chunking_properties = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(chunking_properties, rank, chunk_dims);
    }
    else
      chunking_properties = H5P_DEFAULT;

    // Create global dataset (using dataset_name)
    const hid_t dset_id = H5Dcreate2(file_handle, dataset_name.c_str(), h5type,
                                     filespace0, H5P_DEFAULT,
                                     chunking_properties, H5P_DEFAULT);
    dolfin_assert(dset_id != HDF5_FAIL);

    // Close global data space
    status = H5Sclose(filespace0);
    dolfin_assert(status != HDF5_FAIL);

    // Create a local data space
    const hid_t memspace = H5Screate_simple(rank, count.data(), NULL);
    dolfin_assert(memspace != HDF5_FAIL);

    // Create a file dataspace within the global space - a hyperslab
    const hid_t filespace1 = H5Dget_space(dset_id);
    status = H5Sselect_hyperslab(filespace1, H5S_SELECT_SET, offset.data(), NULL,
                                 count.data(), NULL);
    dolfin_assert(status != HDF5_FAIL);

    // Set parallel access
    const hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    if (use_mpi_io)
    {
      status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
      dolfin_assert(status != HDF5_FAIL);
    }

    // Write local dataset into selected hyperslab
    status = H5Dwrite(dset_id, h5type, memspace, filespace1, plist_id,
                      data.data());
    dolfin_assert(status != HDF5_FAIL);

    // Close dataset collectively
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);

    // Close hyperslab
    status = H5Sclose(filespace1);
    dolfin_assert(status != HDF5_FAIL);

    // Close local dataset
    status = H5Sclose(memspace);
    dolfin_assert(status != HDF5_FAIL);

    // Release file-access template
    status = H5Pclose(plist_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::read_dataset(const hid_t file_handle,
                                          const std::string dataset_name,
                                          const std::pair<std::size_t, std::size_t> range,
                                          std::vector<T>& data)
  {
    // Open the dataset
    const hid_t dset_id = H5Dopen2(file_handle, dataset_name.c_str(), H5P_DEFAULT);
    dolfin_assert(dset_id != HDF5_FAIL);

    // Open dataspace
    const hid_t dataspace = H5Dget_space(dset_id);
    dolfin_assert(dataspace != HDF5_FAIL);

    // Get rank of data set
    const int rank = H5Sget_simple_extent_ndims(dataspace);
    dolfin_assert(rank >= 0);

    if (rank > 2)
      warning("HDF5Interface::read_dataset untested for rank > 2.");

    // Allocate data for size of each dimension
    std::vector<hsize_t> dimensions_size(rank);

    // Get size in each dimension
    const int ndims = H5Sget_simple_extent_dims(dataspace,
                                                dimensions_size.data(), NULL);
    dolfin_assert(ndims == rank);

    // Hyperslab selection
    std::vector<hsize_t> offset(rank, 0);
    offset[0]= range.first;
    std::vector<hsize_t> count(dimensions_size);
    count[0] = range.second - range.first;

    // Select a block in the dataset beginning at offset[], with size=count[]
    herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
                                        offset.data(), NULL, count.data(),
                                        NULL);
    dolfin_assert(status != HDF5_FAIL);

    // Create a memory dataspace
    const hid_t memspace = H5Screate_simple(rank, count.data(), NULL);
    dolfin_assert (memspace != HDF5_FAIL);

    // Resize local data to read into
    std::size_t data_size = 1;
    for (std::size_t i = 0; i < count.size(); ++i)
      data_size *= count[i];
    data.resize(data_size);

    // Read data on each process
    const int h5type = hdf5_type<T>();
    status = H5Dread(dset_id, h5type, memspace, dataspace, H5P_DEFAULT,
                     data.data());
    dolfin_assert(status != HDF5_FAIL);

    // Close dataspace
    status = H5Sclose(dataspace);
    dolfin_assert(status != HDF5_FAIL);

    // Close dataset
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::get_attribute(hid_t hdf5_file_handle,
                                  const std::string dataset_name,
                                  const std::string attribute_name,
                                  T& attribute_value)
  {
    herr_t status;

    // Open dataset by name
    const hid_t dset_id = H5Dopen2(hdf5_file_handle, dataset_name.c_str(),
                                   H5P_DEFAULT);
    dolfin_assert(dset_id != HDF5_FAIL);

    // Open attribute by name and get its type
    const hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
    dolfin_assert(attr_id != HDF5_FAIL);
    const hid_t attr_type = H5Aget_type(attr_id);
    dolfin_assert(attr_type != HDF5_FAIL);

    // Specific code for each type of data template
    get_attribute_value(attr_type, attr_id, attribute_value);

    // Close attribute type
    status = H5Tclose(attr_type);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attr_id);
    dolfin_assert(status != HDF5_FAIL);

    // Close dataset
    status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  inline void HDF5Interface::add_attribute(const hid_t hdf5_file_handle,
                                           const std::string dataset_name,
                                           const std::string attribute_name,
                                           const T& attribute_value)
  {

    // Open named dataset
    hid_t dset_id = H5Dopen2(hdf5_file_handle, dataset_name.c_str(),
                             H5P_DEFAULT);
    dolfin_assert(dset_id != HDF5_FAIL);

    // Add attribute of appropriate type
    add_attribute_value(dset_id, attribute_name, attribute_value);

    // Close dataset
    herr_t status = H5Dclose(dset_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //---------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const long unsigned int& attribute_value)
  {
    // Create a scalar dataspace
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create attribute of type std::size_t
    hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_ULONG, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &attribute_value);
    dolfin_assert(status != HDF5_FAIL);

    // Close attribute
    status = H5Aclose(attribute_id);
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                          const std::string attribute_name,
                                          const unsigned int& attribute_value)
  {
    // Create a scalar dataspace
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create attribute of type uint
    hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(),
                                   H5T_NATIVE_UINT, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT);
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
                                        const std::vector<unsigned int>& attribute_value)
  {

    // Create a vector dataspace
    const hsize_t dimsf = attribute_value.size();
    const hid_t dataspace_id = H5Screate_simple(1, &dimsf, NULL);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create an attribute of type uint in the dataspace
    const hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(),
                                         H5T_NATIVE_UINT, dataspace_id,
                                         H5P_DEFAULT, H5P_DEFAULT);
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
                                        const std::vector<unsigned long int>& attribute_value)
  {

    // Create a vector dataspace
    const hsize_t dimsf = attribute_value.size();
    const hid_t dataspace_id = H5Screate_simple(1, &dimsf, NULL);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Create an attribute of type uint in the dataspace
    const hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(),
                                         H5T_NATIVE_ULONG, dataspace_id,
                                         H5P_DEFAULT, H5P_DEFAULT);
    dolfin_assert(attribute_id != HDF5_FAIL);

    // Write attribute to dataset
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_ULONG, &attribute_value[0]);
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
    // Create a scalar dataspace
    const hid_t dataspace_id = H5Screate(H5S_SCALAR);
    dolfin_assert(dataspace_id != HDF5_FAIL);

    // Copy basic string type from HDF5 types and set string length
    const hid_t datatype_id = H5Tcopy(H5T_C_S1);
    herr_t status = H5Tset_size(datatype_id, attribute_value.size());
    dolfin_assert(status != HDF5_FAIL);

    // Create attribute in the dataspace with the given string
    const hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(),
                                          datatype_id, dataspace_id,
                                          H5P_DEFAULT, H5P_DEFAULT);
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
                                                 unsigned int& attribute_value)
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
                                                 unsigned long int& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // Read value
    herr_t status = H5Aread(attr_id, H5T_NATIVE_ULONG, &attribute_value);
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
    const hid_t memtype = H5Tcopy(H5T_C_S1);
    const int string_length = H5Tget_size(attr_type) + 1;
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
                                          std::vector<unsigned int>& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // get dimensions of attribute array, check it is one-dimensional
    const hid_t dataspace = H5Aget_space(attr_id);
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
  //-----------------------------------------------------------------------------
  template<>
  inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                          const hid_t attr_id,
                                          std::vector<unsigned long int>& attribute_value)
  {
    // FIXME: more complete check of type
    dolfin_assert(H5Tget_class(attr_type) == H5T_INTEGER);

    // get dimensions of attribute array, check it is one-dimensional
    const hid_t dataspace = H5Aget_space(attr_id);
    dolfin_assert(dataspace != HDF5_FAIL);

    hsize_t cur_size[10];
    hsize_t max_size[10];
    const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);
    dolfin_assert(ndims == 1);

    attribute_value.resize(cur_size[0]);

    // Read value to vector
    herr_t status = H5Aread(attr_id, H5T_NATIVE_ULONG, attribute_value.data());
    dolfin_assert(status != HDF5_FAIL);
  }
  //-----------------------------------------------------------------------------
  template <>
  inline hid_t HDF5Interface::hdf5_type<double>()
  { return H5T_NATIVE_DOUBLE; }
  //-----------------------------------------------------------------------------
  template <>
  inline hid_t HDF5Interface::hdf5_type<int>()
  { return H5T_NATIVE_INT; }
  //-----------------------------------------------------------------------------
  template <>
  inline hid_t HDF5Interface::hdf5_type<unsigned int>()
  { return H5T_NATIVE_UINT; }
  //-----------------------------------------------------------------------------
  template <>
  inline hid_t HDF5Interface::hdf5_type<unsigned long int>()
  { return H5T_NATIVE_ULONG; }
  //-----------------------------------------------------------------------------

}

#endif

#endif
