// Copyright (C) 2012-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/log.h>
#include <filesystem>
#include <functional>
#include <hdf5.h>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dolfinx::io::hdf5
{
/// @brief Scope guard for an HDF5 identifier.
///
/// Closes the identifier on destruction, including when the enclosing
/// scope is left by an exception. Non-positive identifiers (invalid
/// handles, and the `H5P_DEFAULT`-style constants) are never closed.
///
/// @note A close failure during destruction is logged rather than
/// thrown. Use ::release and close explicitly where the failure must be
/// reported.
class Handle
{
public:
  /// @brief Take ownership of an HDF5 identifier.
  /// @param[in] id Identifier to manage.
  /// @param[in] close Function releasing `id`, e.g. `H5Dclose`.
  Handle(hid_t id, herr_t (*close)(hid_t)) noexcept : _id(id), _close(close) {}

  // Copy constructor (deleted)
  Handle(const Handle&) = delete;

  /// Move constructor
  Handle(Handle&& h) noexcept
      : _id(std::exchange(h._id, H5I_INVALID_HID)), _close(h._close)
  {
  }

  /// Destructor
  ~Handle() { close(); }

  // Copy assignment (deleted)
  Handle& operator=(const Handle&) = delete;

  /// Move assignment
  Handle& operator=(Handle&& h) noexcept
  {
    if (this != &h)
    {
      close();
      _id = std::exchange(h._id, H5I_INVALID_HID);
      _close = h._close;
    }
    return *this;
  }

  /// Managed identifier
  operator hid_t() const noexcept { return _id; }

  /// @brief Relinquish ownership of the identifier.
  /// @return The identifier, which the caller must close.
  hid_t release() noexcept { return std::exchange(_id, H5I_INVALID_HID); }

private:
  void close() noexcept
  {
    if (_id > 0 and _close(_id) < 0)
      spdlog::warn("Failed to close HDF5 identifier {}.", _id);
  }

  hid_t _id;
  herr_t (*_close)(hid_t);
};

/// @brief Scalar types that can be read from and written to an HDF5
/// file.
template <typename T>
concept native_scalar
    = std::is_same_v<T, float> or std::is_same_v<T, double>
      or std::is_same_v<T, std::int32_t> or std::is_same_v<T, std::uint32_t>
      or std::is_same_v<T, std::int64_t> or std::is_same_v<T, std::uint64_t>
      or std::is_same_v<T, std::uint8_t>;

/// @brief HDF5 native data type corresponding to the C++ type `T`.
template <native_scalar T>
hid_t hdf5_type()
{
  if constexpr (std::is_same_v<T, float>)
    return H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return H5T_NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return H5T_NATIVE_INT32;
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return H5T_NATIVE_UINT32;
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return H5T_NATIVE_INT64;
  else if constexpr (std::is_same_v<T, std::uint64_t>)
    return H5T_NATIVE_UINT64;
  else
    return H5T_NATIVE_UINT8;
}

/// Open HDF5 and return file descriptor
/// @param[in] comm MPI communicator
/// @param[in] filename Name of the HDF5 file to open
/// @param[in] mode Mode in which to open the file (w, r, a)
/// @param[in] use_mpi_io True if MPI-IO should be used
hid_t open_file(MPI_Comm comm, const std::filesystem::path& filename,
                std::string_view mode, bool use_mpi_io);

/// Close HDF5 file
/// @param[in] handle HDF5 file handle
void close_file(hid_t handle);

/// Flush data to file to improve data integrity after interruption
/// @param[in] handle HDF5 file handle
void flush_file(hid_t handle);

/// Get filename
/// @param[in] handle HDF5 file handle
/// return The filename
std::filesystem::path get_filename(hid_t handle);

/// @brief Check for existence of a dataset in an HDF5 file.
/// @param[in] handle HDF5 file handle.
/// @param[in] dataset_path Data set path.
/// @return True if @p dataset_path is in the file and is a dataset. A
/// path that holds an object of another type, e.g. a group, returns
/// false.
bool has_dataset(hid_t handle, std::string_view dataset_path);

/// @brief Set an attribute on a dataset or group.
///
/// An attribute that already exists is replaced, since the datatype and
/// dataspace of an existing attribute are fixed and so cannot
/// accommodate a value of a different size.
///
/// @param[in] handle Dataset or group handle.
/// @param[in] attr_name Name of attribute.
/// @param[in] value Value to set.
void set_attribute(hid_t handle, std::string_view attr_name,
                   std::string_view value);

/// @brief Set an array-valued attribute on a dataset or group.
/// @param[in] handle Dataset or group handle.
/// @param[in] attr_name Name of attribute.
/// @param[in] value Value to set.
void set_attribute(hid_t handle, std::string_view attr_name,
                   const std::vector<std::int32_t>& value);

/// @brief Set a scalar attribute on a dataset or group.
/// @param[in] handle Dataset or group handle.
/// @param[in] attr_name Name of attribute.
/// @param[in] value Value to set.
void set_attribute(hid_t handle, std::string_view attr_name,
                   std::int32_t value);

/// Open dataset
/// @param[in] handle HDF5 file handle.
/// @param[in] path Data set path.
/// @return Data set handle. Should be closed by caller using `H5Dclose`.
/// @throws std::runtime_error if the dataset cannot be opened.
hid_t open_dataset(hid_t handle, std::string_view path);

/// Get dataset shape (size of each dimension)
/// @param[in] handle HDF5 file handle
/// @param[in] dataset_path Dataset path
/// @return The shape of the dataset (row-major)
std::vector<std::int64_t> get_dataset_shape(hid_t handle,
                                            std::string_view dataset_path);

/// Add group to HDF5 file
/// @param[in] handle HDF5 file handle
/// @param[in] dataset_path Data set path to add
void add_group(hid_t handle, std::string_view dataset_path);

/// @brief Write data to existing HDF file as defined by range blocks on
/// each process.
///
/// @param[in] file_handle HDF5 file handle.
/// @param[in] dataset_path Path for the dataset in the HDF5 file.
/// @param[in] data Data to be written, flattened into 1D vector
/// (row-major storage).
/// @param[in] range Local range on this processor.
/// @param[in] global_size Global shape of the array.
/// @param[in] use_mpi_io `true` if MPI-IO should be used.
/// @param[in] use_chunking `true` if chunking should be used, required
/// for extensible datasets.
/// @note Can be used to resize and write into an existing dataset.
/// @note Chunking is required for extensible datasets.
template <native_scalar T>
void write_dataset(hid_t file_handle, std::string_view dataset_path,
                   const T* data, std::array<std::int64_t, 2> range,
                   const std::vector<int64_t>& global_size, bool use_mpi_io,
                   bool use_chunking)
{
  // Data rank. Validate before any collective call, so that all ranks
  // throw together.
  const int rank = global_size.size();
  if (rank < 1 or rank > 2)
  {
    throw std::invalid_argument(
        "Cannot write dataset to HDF5 file. Only rank 1 and rank 2 datasets "
        "are supported.");
  }

  // Check that group exists and recursively create if required
  const std::string group_name(dataset_path, 0, dataset_path.rfind('/'));
  add_group(file_handle, group_name);

  // Null-terminated copy for C API calls
  const std::string path(dataset_path);

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

  Handle dset_id(H5I_INVALID_HID, H5Dclose);
  if (has_dataset(file_handle, dataset_path))
  {
    // Resize existing dataset to the new global size
    dset_id = Handle(open_dataset(file_handle, dataset_path), H5Dclose);
    if (H5Dset_extent(dset_id, dimsf.data()) < 0)
      throw std::runtime_error("Failed to resize HDF5 dataset.");
  }
  else
  {
    std::vector<hsize_t> maxdims(dimsf.begin(), dimsf.end());

    // Set chunking parameters
    Handle chunking_properties(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (chunking_properties < 0)
      throw std::runtime_error("Failed to create HDF5 dataset property list.");
    if (use_chunking)
    {
      // Make array extensible, if chunking is set
      std::ranges::fill(maxdims, H5S_UNLIMITED);

      // Set chunk size and limit to 1kB min/1MB max
      hsize_t chunk_size
          = std::clamp(dimsf[0] / 2, hsize_t(1024), hsize_t(1048576));
      std::vector<hsize_t> chunk_dims(dimsf.begin(), dimsf.end());
      chunk_dims[0] = chunk_size;
      if (H5Pset_chunk(chunking_properties, rank, chunk_dims.data()) < 0)
        throw std::runtime_error("Failed to set HDF5 chunk size.");
    }

    // Create a global data space
    const Handle filespace0(
        H5Screate_simple(rank, dimsf.data(), maxdims.data()), H5Sclose);
    if (filespace0 < 0)
      throw std::runtime_error("Failed to create HDF5 data space");

    // Create global dataset (using dataset_path)
    dset_id = Handle(H5Dcreate2(file_handle, path.c_str(), h5type, filespace0,
                                H5P_DEFAULT, chunking_properties, H5P_DEFAULT),
                     H5Dclose);
    if (dset_id < 0)
      throw std::runtime_error("Failed to create HDF5 global dataset.");
  }

  const Handle dataspace(H5Dget_space(dset_id), H5Sclose);
  if (dataspace < 0)
    throw std::runtime_error("Failed to open HDF5 data space.");

  if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset.data(), nullptr,
                          count.data(), nullptr)
      < 0)
  {
    throw std::runtime_error("Failed to select HDF5 hyperslab.");
  }

  // Set parallel access
  const Handle plist_id(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
  if (plist_id < 0)
    throw std::runtime_error("Failed to create HDF5 data transfer list.");
  if (use_mpi_io)
  {
    if (H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE) < 0)
    {
      throw std::runtime_error(
          "Failed to set HDF5 data transfer property list.");
    }
  }

  // Create a local data space
  const Handle memspace(H5Screate_simple(rank, count.data(), nullptr),
                        H5Sclose);
  if (memspace < 0)
    throw std::runtime_error("Failed to create HDF5 local data space.");

  // Write local dataset
  if (H5Dwrite(dset_id, h5type, memspace, dataspace, plist_id, data) < 0)
    throw std::runtime_error("Failed to write HDF5 local dataset.");
}

/// Read data from a HDF5 dataset "dataset_path" as defined by range blocks on
/// each process.
///
/// @tparam T The data type to read into.
/// @param[in] dset_id HDF5 file handle.
/// @param[in] range The local range on this processor.
/// @param[in] allow_cast If true, allow casting from HDF5 type to type `T`.
/// @return Flattened 1D array of values. If range = {-1, -1}, then all data
/// is read on this process.
template <native_scalar T>
std::vector<T> read_dataset(hid_t dset_id, std::array<std::int64_t, 2> range,
                            bool allow_cast)
{
  auto timer_start = std::chrono::system_clock::now();

  if (!allow_cast)
  {
    // Check that HDF5 dataset type and the type T are the same
    const Handle dtype(H5Dget_type(dset_id), H5Tclose);
    if (dtype < 0)
      throw std::runtime_error("Failed to get HDF5 data type.");
    if (htri_t eq = H5Tequal(dtype, hdf5_type<T>()); eq < 0)
      throw std::runtime_error("HDF5 datatype equality test failed.");
    else if (eq == 0)
    {
      throw std::runtime_error("Wrong type for reading from HDF5. Use \"h5ls "
                               "-v\" to inspect the types in the HDF5 file.");
    }
  }

  // Open dataspace
  const Handle dataspace(H5Dget_space(dset_id), H5Sclose);
  if (dataspace < 0)
    throw std::runtime_error("Failed to open HDF5 data space.");

  // Get rank of data set
  int rank = H5Sget_simple_extent_ndims(dataspace);
  if (rank < 1)
    throw std::runtime_error("Failed to get rank of data space.");
  else if (rank > 2)
    spdlog::warn("io::hdf5::read_dataset untested for rank > 2.");

  // Get size in each dimension
  std::vector<hsize_t> shape(rank);
  if (int ndims = H5Sget_simple_extent_dims(dataspace, shape.data(), nullptr);
      ndims != rank)
  {
    throw std::runtime_error("Failed to get dimensionality of dataspace.");
  }

  // Hyperslab selection
  std::vector<hsize_t> offset(rank, 0);
  std::vector<hsize_t> count = shape;
  if (range[0] != -1 and range[1] != -1)
  {
    offset[0] = range[0];
    count[0] = range[1] - range[0];
  }

  // Select a block in the dataset beginning at offset[], with
  // size=count[]
  if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset.data(), nullptr,
                          count.data(), nullptr)
      < 0)
  {
    throw std::runtime_error("Failed to select HDF5 hyperslab.");
  }

  // Create a memory dataspace
  const Handle memspace(H5Screate_simple(rank, count.data(), nullptr),
                        H5Sclose);
  if (memspace < 0)
    throw std::runtime_error("Failed to create HDF5 dataspace.");

  // Create local data to read into. The extents are hsize_t, so the
  // product is accumulated in std::size_t to avoid narrowing.
  std::vector<T> data(std::reduce(count.begin(), count.end(), std::size_t(1),
                                  std::multiplies{}));

  // Read data on each process
  if (H5Dread(dset_id, hdf5_type<T>(), memspace, dataspace, H5P_DEFAULT,
              data.data())
      < 0)
  {
    throw std::runtime_error("Failed to read HDF5 data.");
  }

  auto timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt = (timer_end - timer_start);
  double data_rate = data.size() * sizeof(T) / (1e6 * dt.count());
  spdlog::info("HDF5 Read data rate: {} MB/s", data_rate);

  return data;
}

/// @brief Read data from a named HDF5 dataset, as defined by range
/// blocks on each process.
///
/// Opens the dataset, reads it, and closes it again.
///
/// @tparam T The data type to read into.
/// @param[in] file_handle HDF5 file handle.
/// @param[in] dataset_path Path for the dataset in the HDF5 file.
/// @param[in] range The local range on this processor.
/// @param[in] allow_cast If true, allow casting from HDF5 type to type
/// `T`.
/// @return Flattened 1D array of values. If range = {-1, -1}, then all
/// data is read on this process.
template <native_scalar T>
std::vector<T> read_dataset(hid_t file_handle, std::string_view dataset_path,
                            std::array<std::int64_t, 2> range, bool allow_cast)
{
  const Handle dset_id(open_dataset(file_handle, dataset_path), H5Dclose);
  return read_dataset<T>(dset_id, range, allow_cast);
}
} // namespace dolfinx::io::hdf5
