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
// Modified by Johannes Ring, 2012
//
// First Added: 2012-09-21
// Last Changed: 2012-12-04

#include <boost/filesystem.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "HDF5File.h"
#include "HDF5Interface.h"

#ifdef HAS_HDF5

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

using namespace dolfin;

//-----------------------------------------------------------------------------
hid_t HDF5Interface::open_file(const std::string filename, const std::string mode,
                               const bool use_mpi_io)
{
  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  if (use_mpi_io)
  {
    #ifdef HAS_MPI
    MPICommunicator comm;
    MPIInfo info;
    herr_t status = H5Pset_fapl_mpio(plist_id, *comm, *info);
    dolfin_assert(status != HDF5_FAIL);
    #else
    dolfin_error("HDF5Interface.cpp",
                 "create HDF5 file",
                 "Cannot use MPI-IO output if DOLFIN is not configured with MPI");
    #endif
  }

  hid_t file_id;
  if (mode=="w")
  {
    // Create file for write, (overwriting existing file, if present)
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                        plist_id);
  }
  else
  {
    // Check that file exists
    if (!boost::filesystem::is_regular_file(filename))
    {
      dolfin_error("HDF5Interface.cpp",
                   "open HDF5 file",
                   "File does not exist");
    }

    if(mode=="a")
    {
      // Open file existing file for append
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
    }
    else if(mode=="r")
    {
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
    }
    else
    {
      dolfin_error("HDF5Interface.cpp",
                   "open HDF5 file",
                   "Unknown file mode \"%s\"", mode.c_str());
    }
    
  }
  dolfin_assert(file_id != HDF5_FAIL);

  // Release file-access template
  herr_t status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  return file_id;
}
//-----------------------------------------------------------------------------
void HDF5Interface::flush_file(const hid_t hdf5_file_handle)
{
  herr_t status = H5Fflush(hdf5_file_handle, H5F_SCOPE_GLOBAL);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  return has_dataset(hdf5_file_handle, group_name);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_dataset(const hid_t hdf5_file_handle,
                                const std::string dataset_name)
{
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  htri_t status = H5Lexists(hdf5_file_handle, dataset_name.c_str(), lapl_id);
  dolfin_assert(status >= 0);
  return status;
}
//-----------------------------------------------------------------------------
void HDF5Interface::add_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  if (has_group(hdf5_file_handle, group_name))
    return;

  hid_t group_id_vis = H5Gcreate2(hdf5_file_handle, group_name.c_str(),
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  dolfin_assert(group_id_vis != HDF5_FAIL);

  herr_t status = H5Gclose(group_id_vis);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
dolfin::uint HDF5Interface::dataset_rank(const hid_t hdf5_file_handle,
					 const std::string dataset_name)
{
  // Open dataset
  const hid_t dset_id = H5Dopen2(hdf5_file_handle, dataset_name.c_str(),
                                 H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Get the dataspace of the dataset
  const hid_t space = H5Dget_space(dset_id);
  dolfin_assert(space != HDF5_FAIL);

  // Get dataset rank
  const int rank = H5Sget_simple_extent_ndims(space);
  dolfin_assert(rank >= 0);

  // Close dataspace and dataset
  herr_t status = H5Sclose(space);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  return rank;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
      HDF5Interface::get_dataset_size(const hid_t hdf5_file_handle,
                                      const std::string dataset_name)
{
  // Open named dataset
  const hid_t dset_id = H5Dopen2(hdf5_file_handle, dataset_name.c_str(),
                                 H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Get the dataspace of the dataset
  const hid_t space = H5Dget_space(dset_id);
  dolfin_assert(space != HDF5_FAIL);

  // Get rank
  const int rank = H5Sget_simple_extent_ndims(space);

  // Allocate data
  std::vector<hsize_t> size(rank);

  // Get size in each dimension
  const int ndims = H5Sget_simple_extent_dims(space, size.data(), NULL);
  dolfin_assert(ndims == rank);

  // Close dataspace and dataset
  herr_t status = H5Sclose(space);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  return std::vector<std::size_t>(size.begin(), size.end());
}
//-----------------------------------------------------------------------------
dolfin::uint HDF5Interface::num_datasets_in_group(const hid_t hdf5_file_handle,
                                                  const std::string group_name)
{
  // Get group info by name
  H5G_info_t group_info;
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  herr_t status = H5Gget_info_by_name(hdf5_file_handle, group_name.c_str(),
                                      &group_info, lapl_id);
  dolfin_assert(status != HDF5_FAIL);
  return group_info.nlinks;
}
//-----------------------------------------------------------------------------
std::vector<std::string> HDF5Interface::dataset_list(const hid_t hdf5_file_handle,
                                                     const std::string group_name)
{
  // List all member datasets of a group by name
  char namebuf[HDF5_MAXSTRLEN];

  herr_t status;

  // Open group by name group_name
  hid_t group_id = H5Gopen2(hdf5_file_handle, group_name.c_str(), H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);

  // Count how many datasets in the group
  hsize_t num_datasets;
  status = H5Gget_num_objs(group_id, &num_datasets);
  dolfin_assert(status != HDF5_FAIL);

  // Iterate through group collecting all dataset names
  std::vector<std::string> list_of_datasets;
  for(hsize_t i = 0; i < num_datasets; i++)
  {
    H5Gget_objname_by_idx(group_id, i, namebuf, HDF5_MAXSTRLEN);
    list_of_datasets.push_back(std::string(namebuf));
  }

  // Close group
  status = H5Gclose(group_id);
  dolfin_assert(status != HDF5_FAIL);

  return list_of_datasets;
}
//-----------------------------------------------------------------------------

#endif
