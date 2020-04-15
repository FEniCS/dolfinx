// Copyright (C) 2012 Chris N Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5File.h"
#include "HDF5Interface.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(MPI_Comm comm, const std::string& filename,
                   const std::string& file_mode)
    : _hdf5_file_id(0), _mpi_comm(comm)
{
  // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
  // _hdf5_file_id(0)

  // Create directory, if required (create on rank 0)
  if (MPI::rank(_mpi_comm.comm()) == 0)
  {
    const boost::filesystem::path path(filename);
    if (path.has_parent_path()
        && !boost::filesystem::is_directory(path.parent_path()))
    {
      boost::filesystem::create_directories(path.parent_path());
      if (!boost::filesystem::is_directory(path.parent_path()))
      {
        throw std::runtime_error("Could not create directory \""
                                 + path.parent_path().string() + "\"");
      }
    }
  }

  // Wait until directory has been created
  MPI_Barrier(_mpi_comm.comm());

  // Open HDF5 file
  const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
#ifndef H5_HAVE_PARALLEL
  if (mpi_io)
  {
    throw std::runtime_error(
        "Cannot open file. HDF5 has not been compiled with support for MPI");
  }
#endif
  _hdf5_file_id
      = HDF5Interface::open_file(_mpi_comm.comm(), filename, file_mode, mpi_io);
  assert(_hdf5_file_id > 0);
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File() { close(); }
//-----------------------------------------------------------------------------
void HDF5File::close()
{
  // Close HDF5 file
  if (_hdf5_file_id > 0)
    HDF5Interface::close_file(_hdf5_file_id);
  _hdf5_file_id = 0;
}
//-----------------------------------------------------------------------------
void HDF5File::flush()
{
  assert(_hdf5_file_id > 0);
  HDF5Interface::flush_file(_hdf5_file_id);
}
//-----------------------------------------------------------------------------
bool HDF5File::has_dataset(const std::string& dataset_name) const
{
  assert(_hdf5_file_id > 0);
  return HDF5Interface::has_dataset(_hdf5_file_id, dataset_name);
}
//-----------------------------------------------------------------------------
void HDF5File::set_mpi_atomicity(bool atomic)
{
  assert(_hdf5_file_id > 0);
  HDF5Interface::set_mpi_atomicity(_hdf5_file_id, atomic);
}
//-----------------------------------------------------------------------------
bool HDF5File::get_mpi_atomicity() const
{
  assert(_hdf5_file_id > 0);
  return HDF5Interface::get_mpi_atomicity(_hdf5_file_id);
}
//-----------------------------------------------------------------------------
