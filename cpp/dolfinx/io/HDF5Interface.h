// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Note: dolfin/common/MPI.h is included before hdf5.h to avoid the
// MPICH_IGNORE_CXX_SEEK issue
#include <dolfinx/common/MPI.h>

#include <dolfinx/common/log.h>
#include <hdf5.h>

namespace dolfinx
{

namespace io
{
class HDF5File;

/// This class wraps HDF5 function calls. HDF5 function calls should
/// only appear in a member function of this class and not elsewhere
/// in the library.

class HDF5Interface
{
#define HDF5_FAIL -1
public:
  /// Open HDF5 and return file descriptor
  static hid_t open_file(MPI_Comm mpi_comm, const std::string filename,
                         const std::string mode, const bool use_mpi_io);

  /// Close HDF5 file
  static void close_file(const hid_t hdf5_file_handle);

  /// Flush data to file to improve data integrity after
  /// interruption
  static void flush_file(const hid_t hdf5_file_handle);

  /// Get filename
  static std::string get_filename(hid_t hdf5_file_handle);

  /// Write data to existing HDF file as defined by range blocks on
  /// each process
  /// data: data to be written, flattened into 1D vector
  /// range: the local range on this processor
  /// global_size: the global multidimensional shape of the array
  /// use_mpio: whether using MPI or not
  /// use_chunking: whether using chunking or not
  template <typename T>
  static void write_dataset(const hid_t file_handle,
                            const std::string dataset_path, const T* data,
                            const std::array<std::int64_t, 2> range,
                            const std::vector<std::int64_t> global_size,
                            bool use_mpio, bool use_chunking);

  /// Read data from a HDF5 dataset "dataset_path" as defined by
  /// range blocks on each process range: the local range on this
  /// processor data: a flattened 1D array of values. If range = {-1, -1},
  /// then all data is read on this process.
  template <typename T>
  static std::vector<T> read_dataset(const hid_t file_handle,
                                     const std::string dataset_path,
                                     const std::array<std::int64_t, 2> range);

  /// Check for existence of group in HDF5 file
  static bool has_group(const hid_t hdf5_file_handle,
                        const std::string group_name);

  /// Check for existence of dataset in HDF5 file
  static bool has_dataset(const hid_t hdf5_file_handle,
                          const std::string dataset_path);

  /// Add group to HDF5 file
  static void add_group(const hid_t hdf5_file_handle,
                        const std::string dataset_path);

  /// Get dataset rank
  static int dataset_rank(const hid_t hdf5_file_handle,
                          const std::string dataset_path);

  /// Return number of data sets in a group
  static int num_datasets_in_group(const hid_t hdf5_file_handle,
                                   const std::string group_name);

  /// Get dataset shape (size of each dimension)
  static std::vector<std::int64_t>
  get_dataset_shape(const hid_t hdf5_file_handle,
                    const std::string dataset_path);

  /// Return list all datasets in named group of file
  static std::vector<std::string> dataset_list(const hid_t hdf5_file_handle,
                                               const std::string group_name);

  /// Get type of attribute
  static const std::string get_attribute_type(const hid_t hdf5_file_handle,
                                              const std::string dataset_path,
                                              const std::string attribute_name);

  /// Get a named attribute of a dataset of known type
  template <typename T>
  static T get_attribute(const hid_t hdf5_file_handle,
                         const std::string dataset_path,
                         const std::string attribute_name);

  /// Add attribute to dataset or group
  template <typename T>
  static void
  add_attribute(const hid_t hdf5_file_handle, const std::string dataset_path,
                const std::string attribute_name, const T& attribute_value);

  /// Delete an attribute from a dataset or group
  static void delete_attribute(const hid_t hdf5_file_handle,
                               const std::string dataset_path,
                               const std::string attribute_name);

  /// Check if an attribute exists on a dataset or group
  static bool has_attribute(const hid_t hdf5_file_handle,
                            const std::string dataset_path,
                            const std::string attribute_name);

  /// List attributes of dataset or group
  static const std::vector<std::string>
  list_attributes(const hid_t hdf5_file_handle, const std::string dataset_path);

  /// Set MPI atomicity. See
  /// https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-SetMpiAtomicity
  /// and
  /// https://www.open-mpi.org/doc/v2.0/man3/MPI_File_set_atomicity.3.php
  /// Writes must be followed by an MPI_Barrier on the communicator before
  /// any subsequent reads are guaranteed to return the same data.
  static void set_mpi_atomicity(const hid_t hdf5_file_handle,
                                const bool atomic);

  /// Get MPI atomicity. See
  /// https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-GetMpiAtomicity
  /// and
  /// https://www.open-mpi.org/doc/v2.0/man3/MPI_File_get_atomicity.3.php
  static bool get_mpi_atomicity(const hid_t hdf5_file_handle);

private:
  static herr_t attribute_iteration_function(hid_t loc_id, const char* name,
                                             const H5A_info_t* info, void* str);

  template <typename T>
  static void add_attribute_value(const hid_t dset_id,
                                  const std::string attribute_name,
                                  const T& attribute_value);

  template <typename T>
  static void add_attribute_value(const hid_t dset_id,
                                  const std::string attribute_name,
                                  const std::vector<T>& attribute_value);

  template <typename T>
  static void get_attribute_value(const hid_t attr_type, const hid_t attr_id,
                                  T& attribute_value);

  template <typename T>
  static void get_attribute_value(const hid_t attr_type, const hid_t attr_id,
                                  std::vector<T>& attribute_value);

  // Return HDF5 data type
  template <typename T>
  static hid_t hdf5_type()
  {
    throw std::runtime_error("Cannot get HDF5 primitive data type. "
                             "No specialised function for this data type");
    return 0;
  }
};
/// @cond
//---------------------------------------------------------------------------
template <>
inline hid_t HDF5Interface::hdf5_type<float>()
{
  return H5T_NATIVE_FLOAT;
}
//---------------------------------------------------------------------------
template <>
inline hid_t HDF5Interface::hdf5_type<double>()
{
  return H5T_NATIVE_DOUBLE;
}
//---------------------------------------------------------------------------
template <>
inline hid_t HDF5Interface::hdf5_type<int>()
{
  return H5T_NATIVE_INT;
}
//---------------------------------------------------------------------------
template <>
inline hid_t HDF5Interface::hdf5_type<std::int64_t>()
{
  return H5T_NATIVE_INT64;
}
//---------------------------------------------------------------------------
template <>
inline hid_t HDF5Interface::hdf5_type<std::size_t>()
{
  if (sizeof(std::size_t) == sizeof(unsigned long))
    return H5T_NATIVE_ULONG;
  else if (sizeof(std::size_t) == sizeof(unsigned int))
    return H5T_NATIVE_UINT;
  throw std::runtime_error("Cannot determine size of std::size_t. "
                           "std::size_t is not the same size as long or int");
  return 0;
}
//---------------------------------------------------------------------------
template <typename T>
inline void HDF5Interface::write_dataset(
    const hid_t file_handle, const std::string dataset_path, const T* data,
    const std::array<std::int64_t, 2> range,
    const std::vector<int64_t> global_size, bool use_mpi_io, bool use_chunking)
{
  // Data rank
  const std::size_t rank = global_size.size();
  assert(rank != 0);

  if (rank > 2)
  {
    throw std::runtime_error("Cannot write dataset to HDF5 file"
                             "Only rank 1 and rank 2 dataset are supported");
  }

  // Get HDF5 data type
  const hid_t h5type = hdf5_type<T>();

  // Hyperslab selection parameters
  std::vector<hsize_t> count(global_size.begin(), global_size.end());
  count[0] = range[1] - range[0];

  // Data offsets
  std::vector<hsize_t> offset(rank, 0);
  offset[0] = range[0];

  // Dataset dimensions
  const std::vector<hsize_t> dimsf(global_size.begin(), global_size.end());

  // Generic status report
  herr_t status;

  // Create a global data space
  const hid_t filespace0 = H5Screate_simple(rank, dimsf.data(), nullptr);
  assert(filespace0 != HDF5_FAIL);

  // Set chunking parameters
  hid_t chunking_properties;
  if (use_chunking)
  {
    // Set chunk size and limit to 1kB min/1MB max
    hsize_t chunk_size = dimsf[0] / 2;
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

  // Check that group exists and recursively create if required
  const std::string group_name(dataset_path, 0, dataset_path.rfind('/'));
  add_group(file_handle, group_name);

  // Create global dataset (using dataset_path)
  const hid_t dset_id
      = H5Dcreate2(file_handle, dataset_path.c_str(), h5type, filespace0,
                   H5P_DEFAULT, chunking_properties, H5P_DEFAULT);
  assert(dset_id != HDF5_FAIL);

  // Close global data space
  status = H5Sclose(filespace0);
  assert(status != HDF5_FAIL);

  // Create a local data space
  const hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
  assert(memspace != HDF5_FAIL);

  // Create a file dataspace within the global space - a hyperslab
  const hid_t filespace1 = H5Dget_space(dset_id);
  status = H5Sselect_hyperslab(filespace1, H5S_SELECT_SET, offset.data(),
                               nullptr, count.data(), nullptr);
  assert(status != HDF5_FAIL);

  // Set parallel access
  const hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  if (use_mpi_io)
  {
#ifdef H5_HAVE_PARALLEL
    status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    assert(status != HDF5_FAIL);
#else
    throw std::runtime_error("HDF5 library has not been configured with MPI");
#endif
  }

  // Write local dataset into selected hyperslab
  status = H5Dwrite(dset_id, h5type, memspace, filespace1, plist_id, data);
  assert(status != HDF5_FAIL);

  if (use_chunking)
  {
    // Close chunking properties
    status = H5Pclose(chunking_properties);
    assert(status != HDF5_FAIL);
  }

  // Close dataset collectively
  status = H5Dclose(dset_id);
  assert(status != HDF5_FAIL);

  // Close hyperslab
  status = H5Sclose(filespace1);
  assert(status != HDF5_FAIL);

  // Close local dataset
  status = H5Sclose(memspace);
  assert(status != HDF5_FAIL);

  // Release file-access template
  status = H5Pclose(plist_id);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
template <typename T>
inline std::vector<T>
HDF5Interface::read_dataset(const hid_t file_handle,
                            const std::string dataset_path,
                            const std::array<std::int64_t, 2> range)
{
  // Open the dataset
  const hid_t dset_id
      = H5Dopen2(file_handle, dataset_path.c_str(), H5P_DEFAULT);
  assert(dset_id != HDF5_FAIL);

  // Open dataspace
  const hid_t dataspace = H5Dget_space(dset_id);
  assert(dataspace != HDF5_FAIL);

  // Get rank of data set
  const int rank = H5Sget_simple_extent_ndims(dataspace);
  assert(rank >= 0);

  if (rank > 2)
    LOG(WARNING) << "HDF5Interface::read_dataset untested for rank > 2.";

  // Allocate data for shape
  std::vector<hsize_t> shape(rank);

  // Get size in each dimension
  const int ndims = H5Sget_simple_extent_dims(dataspace, shape.data(), nullptr);
  assert(ndims == rank);

  // Hyperslab selection
  std::vector<hsize_t> offset(rank, 0);
  std::vector<hsize_t> count = shape;
  if (range[0] != -1 and range[1] != -1)
  {
    offset[0] = range[0];
    count[0] = range[1] - range[0];
  }
  else
    offset[0] = 0;

  // Select a block in the dataset beginning at offset[], with
  // size=count[]
  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset.data(),
                                      nullptr, count.data(), nullptr);
  assert(status != HDF5_FAIL);

  // Create a memory dataspace
  const hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
  assert(memspace != HDF5_FAIL);

  // Create local data to read into
  std::size_t data_size = 1;
  for (std::size_t i = 0; i < count.size(); ++i)
    data_size *= count[i];
  std::vector<T> data(data_size);

  // Read data on each process
  const hid_t h5type = hdf5_type<T>();
  status
      = H5Dread(dset_id, h5type, memspace, dataspace, H5P_DEFAULT, data.data());
  assert(status != HDF5_FAIL);

  // Close dataspace
  status = H5Sclose(dataspace);
  assert(status != HDF5_FAIL);

  // Close memspace
  status = H5Sclose(memspace);
  assert(status != HDF5_FAIL);

  // Close dataset
  status = H5Dclose(dset_id);
  assert(status != HDF5_FAIL);

  return data;
}
//---------------------------------------------------------------------------
template <typename T>
inline T HDF5Interface::get_attribute(hid_t hdf5_file_handle,
                                      const std::string dataset_path,
                                      const std::string attribute_name)
{
  herr_t status;

  // Open dataset or group by name
  const hid_t dset_id
      = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  assert(dset_id != HDF5_FAIL);

  // Open attribute by name and get its type
  const hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
  assert(attr_id != HDF5_FAIL);
  const hid_t attr_type = H5Aget_type(attr_id);
  assert(attr_type != HDF5_FAIL);

  // Specific code for each type of data template
  T attribute_value;
  get_attribute_value(attr_type, attr_id, attribute_value);

  // Close attribute type
  status = H5Tclose(attr_type);
  assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attr_id);
  assert(status != HDF5_FAIL);

  // Close dataset or group
  status = H5Oclose(dset_id);
  assert(status != HDF5_FAIL);

  return attribute_value;
}
//--------------------------------------------------------------------------
template <typename T>
inline void HDF5Interface::add_attribute(const hid_t hdf5_file_handle,
                                         const std::string dataset_path,
                                         const std::string attribute_name,
                                         const T& attribute_value)
{

  // Open named dataset or group
  hid_t dset_id = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  assert(dset_id != HDF5_FAIL);

  // Check if attribute already exists and delete if so
  htri_t has_attr = H5Aexists(dset_id, attribute_name.c_str());
  assert(has_attr != HDF5_FAIL);
  if (has_attr > 0)
  {
    herr_t status = H5Adelete(dset_id, attribute_name.c_str());
    assert(status != HDF5_FAIL);
  }

  // Add attribute of appropriate type
  add_attribute_value(dset_id, attribute_name, attribute_value);

  // Close dataset or group
  herr_t status = H5Oclose(dset_id);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
// Specialised member functions (must be inlined to avoid link errors)
//---------------------------------------------------------------------------

// Template for simple types (e.g. size_t, double, int etc.) and
// vectors of these
// Specialization below for string
template <typename T>
inline void HDF5Interface::add_attribute_value(const hid_t dset_id,
                                               const std::string attribute_name,
                                               const T& attribute_value)
{
  // Create a scalar dataspace
  hid_t dataspace_id = H5Screate(H5S_SCALAR);
  assert(dataspace_id != HDF5_FAIL);

  const hid_t h5type = hdf5_type<T>();

  // Create attribute of type std::size_t
  hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(), h5type,
                                  dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  assert(attribute_id != HDF5_FAIL);

  // Write attribute to dataset
  herr_t status = H5Awrite(attribute_id, h5type, &attribute_value);
  assert(status != HDF5_FAIL);

  // Close dataspace
  status = H5Sclose(dataspace_id);
  assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attribute_id);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
template <typename T>
inline void
HDF5Interface::add_attribute_value(const hid_t dset_id,
                                   const std::string attribute_name,
                                   const std::vector<T>& attribute_value)
{

  const hid_t h5type = hdf5_type<T>();

  // Create a vector dataspace
  const hsize_t dimsf = attribute_value.size();
  const hid_t dataspace_id = H5Screate_simple(1, &dimsf, nullptr);
  assert(dataspace_id != HDF5_FAIL);

  // Create an attribute of type size_t in the dataspace
  const hid_t attribute_id = H5Acreate2(dset_id, attribute_name.c_str(), h5type,
                                        dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  assert(attribute_id != HDF5_FAIL);

  // Write attribute to dataset
  herr_t status = H5Awrite(attribute_id, h5type, attribute_value.data());
  assert(status != HDF5_FAIL);

  // Close dataspace
  status = H5Sclose(dataspace_id);
  assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attribute_id);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
template <>
inline void
HDF5Interface::add_attribute_value(const hid_t dset_id,
                                   const std::string attribute_name,
                                   const std::string& attribute_value)
{
  // Create a scalar dataspace
  const hid_t dataspace_id = H5Screate(H5S_SCALAR);
  assert(dataspace_id != HDF5_FAIL);

  // Copy basic string type from HDF5 types and set string length
  const hid_t datatype_id = H5Tcopy(H5T_C_S1);
  herr_t status = H5Tset_size(datatype_id, attribute_value.size());
  assert(status != HDF5_FAIL);

  // Create attribute in the dataspace with the given string
  const hid_t attribute_id
      = H5Acreate2(dset_id, attribute_name.c_str(), datatype_id, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT);
  assert(attribute_id != HDF5_FAIL);

  // Write attribute to dataset
  status = H5Awrite(attribute_id, datatype_id, attribute_value.c_str());
  assert(status != HDF5_FAIL);

  // Close dataspace
  status = H5Sclose(dataspace_id);
  assert(status != HDF5_FAIL);

  // Close string type
  status = H5Tclose(datatype_id);
  assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attribute_id);
  assert(status != HDF5_FAIL);
}
//--------------------------------------------------------------------------
template <typename T>
inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                               const hid_t attr_id,
                                               T& attribute_value)
{
  const hid_t h5type = hdf5_type<T>();

  // FIXME: more complete check of type
  assert(H5Tget_class(attr_type) == H5Tget_class(h5type));

  // Read value
  herr_t status = H5Aread(attr_id, h5type, &attribute_value);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
template <typename T>
inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                               const hid_t attr_id,
                                               std::vector<T>& attribute_value)
{
  const hid_t h5type = hdf5_type<T>();

  // FIXME: more complete check of type
  assert(H5Tget_class(attr_type) == H5Tget_class(h5type));

  // get dimensions of attribute array, check it is one-dimensional
  const hid_t dataspace = H5Aget_space(attr_id);
  assert(dataspace != HDF5_FAIL);

  hsize_t cur_size[10];
  hsize_t max_size[10];
  const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);
  assert(ndims == 1);

  attribute_value.resize(cur_size[0]);

  // Read value to vector
  herr_t status = H5Aread(attr_id, h5type, attribute_value.data());
  assert(status != HDF5_FAIL);

  // Close dataspace
  status = H5Sclose(dataspace);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
template <>
inline void HDF5Interface::get_attribute_value(const hid_t attr_type,
                                               const hid_t attr_id,
                                               std::string& attribute_value)
{
  // Check this attribute is a string
  assert(H5Tget_class(attr_type) == H5T_STRING);

  // Copy string type from HDF5 types and set length accordingly
  const hid_t memtype = H5Tcopy(H5T_C_S1);
  const int string_length = H5Tget_size(attr_type) + 1;
  herr_t status = H5Tset_size(memtype, string_length);
  assert(status != HDF5_FAIL);

  // FIXME: messy
  // Copy string value into temporary vector std::vector::data can
  // be copied into (std::string::data cannot)
  std::vector<char> attribute_data(string_length);
  status = H5Aread(attr_id, memtype, attribute_data.data());
  assert(status != HDF5_FAIL);

  attribute_value.assign(attribute_data.data());

  // Close memory type
  status = H5Tclose(memtype);
  assert(status != HDF5_FAIL);
}
//---------------------------------------------------------------------------
/// @endcond
} // namespace io
} // namespace dolfinx
