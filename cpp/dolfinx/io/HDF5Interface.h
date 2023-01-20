// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <dolfinx/common/log.h>
#include <filesystem>
#include <hdf5.h>
#include <mpi.h>
#include <numeric>
#include <string>
#include <vector>

namespace dolfinx::io
{

/// This class provides an interface to some HDF5 functionality

class HDF5Interface
{
#define HDF5_FAIL -1
public:
  /// Open HDF5 and return file descriptor
  /// @param[in] comm MPI communicator
  /// @param[in] filename Name of the HDF5 file to open
  /// @param[in] mode Mode in which to open the file (w, r, a)
  /// @param[in] use_mpi_io True if MPI-IO should be used
  static hid_t open_file(MPI_Comm comm, const std::filesystem::path& filename,
                         const std::string& mode, const bool use_mpi_io);

  /// Close HDF5 file
  /// @param[in] handle HDF5 file handle
  static void close_file(const hid_t handle);

  /// Flush data to file to improve data integrity after interruption
  /// @param[in] handle HDF5 file handle
  static void flush_file(const hid_t handle);

  /// Get filename
  /// @param[in] handle HDF5 file handle
  /// return The filename
  static std::filesystem::path get_filename(hid_t handle);

  /// Write data to existing HDF file as defined by range blocks on each
  /// process
  /// @param[in] handle HDF5 file handle
  /// @param[in] dataset_path Path for the dataset in the HDF5 file
  /// @param[in] data Data to be written, flattened into 1D vector
  ///   (row-major storage)
  /// @param[in] range The local range on this processor
  /// @param[in] global_size The global shape shape of the array
  /// @param[in] use_mpi_io True if MPI-IO should be used
  /// @param[in] use_chunking True if chunking should be used
  template <typename T>
  static void write_dataset(const hid_t handle, const std::string& dataset_path,
                            const T* data,
                            const std::array<std::int64_t, 2>& range,
                            const std::vector<std::int64_t>& global_size,
                            bool use_mpi_io, bool use_chunking);

  /// Read data from a HDF5 dataset "dataset_path" as defined by range
  /// blocks on each process.
  ///
  /// @param[in] handle HDF5 file handle
  /// @param[in] dataset_path Path for the dataset in the HDF5 file
  /// @param[in] range The local range on this processor
  /// @return Flattened 1D array of values. If range = {-1, -1}, then
  ///   all data is read on this process.
  template <typename T>
  static std::vector<T> read_dataset(const hid_t handle,
                                     const std::string& dataset_path,
                                     const std::array<std::int64_t, 2>& range);

  /// Check for existence of dataset in HDF5 file
  /// @param[in] handle HDF5 file handle
  /// @param[in] dataset_path Data set path
  /// @return True if @p dataset_path is in the file
  static bool has_dataset(const hid_t handle, const std::string& dataset_path);

  /// Get dataset shape (size of each dimension)
  /// @param[in] handle HDF5 file handle
  /// @param[in] dataset_path Dataset path
  /// @return The shape of the dataset (row-major)
  static std::vector<std::int64_t>
  get_dataset_shape(const hid_t handle, const std::string& dataset_path);

  /// Set MPI atomicity. See
  /// https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-SetMpiAtomicity
  /// and
  /// https://www.open-mpi.org/doc/v2.0/man3/MPI_File_set_atomicity.3.php
  /// Writes must be followed by an MPI_Barrier on the communicator before
  /// any subsequent reads are guaranteed to return the same data.
  static void set_mpi_atomicity(const hid_t handle, const bool atomic);

  /// Get MPI atomicity. See
  /// https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-GetMpiAtomicity
  /// and
  /// https://www.open-mpi.org/doc/v2.0/man3/MPI_File_get_atomicity.3.php
  static bool get_mpi_atomicity(const hid_t handle);

private:
  /// Add group to HDF5 file
  /// @param[in] handle HDF5 file handle
  /// @param[in] dataset_path Data set path to add
  static void add_group(const hid_t handle, const std::string& dataset_path);

  // Return HDF5 data type
  template <typename T>
  static hid_t hdf5_type()
  {
    throw DolfinXException("Cannot get HDF5 primitive data type. "
                           "No specialised function for this data type");
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

  throw DolfinXException("Cannot determine size of std::size_t. "
                         "std::size_t is not the same size as long or int");
}
//---------------------------------------------------------------------------
template <typename T>
inline void HDF5Interface::write_dataset(
    const hid_t file_handle, const std::string& dataset_path, const T* data,
    const std::array<std::int64_t, 2>& range,
    const std::vector<int64_t>& global_size, bool use_mpi_io, bool use_chunking)
{
  // Data rank
  const int rank = global_size.size();
  assert(rank != 0);
  if (rank > 2)
  {
    throw DolfinXException("Cannot write dataset to HDF5 file"
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

  // Create a global data space
  const hid_t filespace0 = H5Screate_simple(rank, dimsf.data(), nullptr);
  if (filespace0 == HDF5_FAIL)
    throw DolfinXException("Failed to create HDF5 data space");

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
  if (dset_id == HDF5_FAIL)
    throw DolfinXException("Failed to create HDF5 global dataset.");

  // Generic status report
  herr_t status;

  // Close global data space
  status = H5Sclose(filespace0);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 global data space.");

  // Create a local data space
  const hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
  if (memspace == HDF5_FAIL)
    throw DolfinXException("Failed to create HDF5 local data space.");

  // Create a file dataspace within the global space - a hyperslab
  const hid_t filespace1 = H5Dget_space(dset_id);
  status = H5Sselect_hyperslab(filespace1, H5S_SELECT_SET, offset.data(),
                               nullptr, count.data(), nullptr);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to create HDF5 dataspace.");

  // Set parallel access
  const hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  if (use_mpi_io)
  {
#ifdef H5_HAVE_PARALLEL
    status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    if (status == HDF5_FAIL)
    {
      throw DolfinXException("Failed to set HDF5 data transfer property list.");
    }

#else
    throw DolfinXException("HDF5 library has not been configured with MPI");
#endif
  }

  // Write local dataset into selected hyperslab
  status = H5Dwrite(dset_id, h5type, memspace, filespace1, plist_id, data);
  if (status == HDF5_FAIL)
  {
    throw DolfinXException(
        "Failed to write HDF5 local dataset into hyperslab.");
  }

  if (use_chunking)
  {
    // Close chunking properties
    status = H5Pclose(chunking_properties);
    if (status == HDF5_FAIL)
      throw DolfinXException("Failed to close HDF5 chunking properties.");
  }

  // Close dataset collectively
  status = H5Dclose(dset_id);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 dataset.");

  // Close hyperslab
  status = H5Sclose(filespace1);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 hyperslab.");

  // Close local dataset
  status = H5Sclose(memspace);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close local HDF5 dataset.");

  // Release file-access template
  status = H5Pclose(plist_id);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to release HDF5 file-access template.");
}
//---------------------------------------------------------------------------
template <typename T>
inline std::vector<T>
HDF5Interface::read_dataset(const hid_t file_handle,
                            const std::string& dataset_path,
                            const std::array<std::int64_t, 2>& range)
{
  auto timer_start = std::chrono::system_clock::now();

  // Open the dataset
  const hid_t dset_id
      = H5Dopen2(file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id == HDF5_FAIL)
    throw DolfinXException("Failed to open HDF5 global dataset.");

  // Open dataspace
  const hid_t dataspace = H5Dget_space(dset_id);
  if (dataspace == HDF5_FAIL)
    throw DolfinXException("Failed to open HDF5 data space.");

  // Get rank of data set
  const int rank = H5Sget_simple_extent_ndims(dataspace);
  assert(rank >= 0);
  if (rank > 2)
    LOG(WARNING) << "HDF5Interface::read_dataset untested for rank > 2.";

  // Allocate data for shape
  std::vector<hsize_t> shape(rank);

  // Get size in each dimension
  const int ndims = H5Sget_simple_extent_dims(dataspace, shape.data(), nullptr);
  if (ndims != rank)
    throw DolfinXException("Failed to get dimensionality of dataspace");

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
  [[maybe_unused]] herr_t status = H5Sselect_hyperslab(
      dataspace, H5S_SELECT_SET, offset.data(), nullptr, count.data(), nullptr);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to select HDF5 hyperslab.");

  // Create a memory dataspace
  const hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
  if (memspace == HDF5_FAIL)
    throw DolfinXException("Failed to create HDF5 dataspace.");

  // Create local data to read into
  std::vector<T> data(
      std::reduce(count.begin(), count.end(), 1, std::multiplies{}));

  // Read data on each process
  const hid_t h5type = hdf5_type<T>();
  status
      = H5Dread(dset_id, h5type, memspace, dataspace, H5P_DEFAULT, data.data());
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to read HDF5 data.");

  // Close dataspace
  status = H5Sclose(dataspace);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 dataspace.");

  // Close memspace
  status = H5Sclose(memspace);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 memory space.");

  // Close dataset
  status = H5Dclose(dset_id);
  if (status == HDF5_FAIL)
    throw DolfinXException("Failed to close HDF5 dataset.");

  auto timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt = (timer_end - timer_start);
  double data_rate = data.size() * sizeof(T) / (1e6 * dt.count());

  LOG(INFO) << "HDF5 Read data rate: " << data_rate << "MB/s";

  return data;
}
//---------------------------------------------------------------------------
/// @endcond
} // namespace dolfinx::io
