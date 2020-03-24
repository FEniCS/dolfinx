// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace io
{

/// Interface to HDF5 files

class HDF5File
{

public:
  /// Constructor. file_mode should be "a" (append), "w" (write) or "r"
  /// (read).
  HDF5File(MPI_Comm comm, const std::string filename,
           const std::string file_mode);

  /// Destructor
  ~HDF5File();

  /// Close file
  void close();

  /// Flush buffered I/O to disk
  void flush();

  /// Check if dataset exists in HDF5 file
  bool has_dataset(const std::string dataset_name) const;

  /// Set the MPI atomicity
  void set_mpi_atomicity(bool atomic);

  /// Get the MPI atomicity
  bool get_mpi_atomicity() const;

  /// Get the file ID
  hid_t h5_id() const { return _hdf5_file_id; }

  /// Chunking parameter - partition data into fixed size blocks for efficiency
  bool chunking = false;

  /// Write contiguous data to HDF5 data set. Data is flattened into a
  /// 1D array, e.g. [x0, y0, z0, x1, y1, z1] for a vector in 3D
  template <typename T>
  void write_data(const std::string dataset_name, const std::vector<T>& data,
                  const std::vector<std::int64_t>& global_size,
                  bool use_mpi_io);

  /// Write 2D dataset to HDF5. Eigen::Arrays on each process must have
  /// the same number of columns.
  template <typename T>
  void write_data(
      const std::string dataset_name,
      Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data,
      bool use_mpi_io);

private:
  // HDF5 file descriptor/handle
  hid_t _hdf5_file_id;

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;
};

//---------------------------------------------------------------------------
// Needs to go here, because of use in XDMFFile.cpp
template <typename T>
void HDF5File::write_data(const std::string dataset_name,
                          const std::vector<T>& data,
                          const std::vector<std::int64_t>& global_size,
                          bool use_mpi_io)
{
  assert(_hdf5_file_id > 0);
  assert(global_size.size() > 0);

  // Get number of 'items'
  std::int64_t num_local_items = 1;
  for (std::size_t i = 1; i < global_size.size(); ++i)
    num_local_items *= global_size[i];
  num_local_items = data.size() / num_local_items;

  // Compute offset
  const std::int64_t offset
      = MPI::global_offset(_mpi_comm.comm(), num_local_items, true);
  std::array<std::int64_t, 2> range = {{offset, offset + num_local_items}};

  // Write data to HDF5 file. Ensure dataset starts with '/'.
  std::string dset_name(dataset_name);
  if (dset_name[0] != '/')
    dset_name = "/" + dataset_name;

  HDF5Interface::write_dataset(_hdf5_file_id, dset_name, data.data(), range,
                               global_size, use_mpi_io, chunking);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write_data(
    const std::string dataset_name,
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data,
    bool use_mpi_io)
{
  assert(_hdf5_file_id > 0);

  // Compute offset
  const std::int64_t offset
      = MPI::global_offset(_mpi_comm.comm(), data.rows(), true);
  std::array<std::int64_t, 2> range = {{offset, offset + data.rows()}};

  // Write data to HDF5 file. Ensure dataset starts with '/'.
  std::string dset_name(dataset_name);
  if (dset_name[0] != '/')
    dset_name = "/" + dataset_name;

  std::int64_t global_rows = MPI::sum(_mpi_comm.comm(), data.rows());
  std::vector<std::int64_t> global_size = {global_rows, data.cols()};
  if (data.cols() == 1)
    global_size = {global_rows};

  HDF5Interface::write_dataset(_hdf5_file_id, dset_name, data.data(), range,
                               global_size, use_mpi_io, chunking);
}
//---------------------------------------------------------------------------
} // namespace io
} // namespace dolfinx
