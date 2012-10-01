// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
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
//
// First Added: 2012-09-21
// Last Changed: 2012-10-01

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "HDF5File.h"
#include "HDF5Interface.h"

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

using namespace dolfin;

//-----------------------------------------------------------------------------
hid_t HDF5Interface::open_file(const std::string filename, const bool truncate,
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
                 "create file",
                 "Cannot use MPI-IO output if DOLFIN is not configured with MPI");
    #endif
  }

  // Create file (overwriting existing file, if present)
  hid_t file_id;
  if (truncate)
  {
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                        plist_id);
  }
  else
  {
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  }
  dolfin_assert(file_id != HDF5_FAIL);

  // Release file-access template
  herr_t status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  return file_id;
}
//-----------------------------------------------------------------------------
void HDF5Interface::create(const std::string filename, bool use_mpiio)
{
  // make empty HDF5 file
  // overwriting any existing file
  // create some default 'folders' for storing different datasets

  // Return satus
  herr_t status;

  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  if (use_mpiio)
  {
    #ifdef HAS_MPI
    MPICommunicator comm;
    MPIInfo info;
    status = H5Pset_fapl_mpio(plist_id, *comm, *info);
    dolfin_assert(status != HDF5_FAIL);
    #else
    dolfin_error("HDF5Interface.cpp",
                 "create file",
                 "Cannot use MPI-IO output if DOLFIN is not configured with MPI");
    #endif
  }

  // Create file (overwriting existing file, if present)
  const hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                                  H5P_DEFAULT, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // Create subgroups suitable for storing different types of data.
  // VisualisationVector - values for visualisation
  const hid_t group_id_vis = H5Gcreate(file_id, "/VisualisationVector", H5P_DEFAULT);
  dolfin_assert(group_id_vis != HDF5_FAIL);
  status = H5Gclose(group_id_vis);
  dolfin_assert(status != HDF5_FAIL);

  // Vector - for checkpointing, etc
  const hid_t group_id_vector = H5Gcreate(file_id, "/Vector", H5P_DEFAULT);
  dolfin_assert(group_id_vector != HDF5_FAIL);
  status = H5Gclose(group_id_vector);
  dolfin_assert(status != HDF5_FAIL);

  // Mesh
  const hid_t group_id_mesh = H5Gcreate(file_id, "/Mesh", H5P_DEFAULT);
  dolfin_assert(group_id_mesh != HDF5_FAIL);
  status = H5Gclose(group_id_mesh);
  dolfin_assert(status != HDF5_FAIL);

  // Release file-access template
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_group(const hid_t hdf5_file_handle,
                                    const std::string group_name)
{
   herr_t status = H5Eset_auto(NULL, NULL);
   status = H5Gget_objinfo (hdf5_file_handle, group_name.c_str(), 0, NULL);
   if (status == 0)
    return true;
   else
    return false;
}
//-----------------------------------------------------------------------------
void HDF5Interface::add_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  if(has_group(hdf5_file_handle, group_name)) return;

  hid_t group_id_vis = H5Gcreate(hdf5_file_handle, group_name.c_str(), H5P_DEFAULT);
  dolfin_assert(group_id_vis != HDF5_FAIL);

  herr_t status = H5Gclose(group_id_vis);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
uint HDF5Interface::dataset_rank(const hid_t hdf5_file_handle,
                         const std::string dataset_name)
{
  // Open dataset
  const hid_t dset_id = H5Dopen(hdf5_file_handle, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Get the dataspace of the dataset
  const hid_t space = H5Dget_space(dset_id);
  dolfin_assert(space != HDF5_FAIL);

  // Get dataset rank
  const int rank = H5Sget_simple_extent_ndims(space);
  dolfin_assert(rank >= 0);
  return rank;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>
      HDF5Interface::get_dataset_size(const hid_t hdf5_file_handle,
                                      const std::string dataset_name)
{
  // Open named dataset
  const hid_t dset_id = H5Dopen(hdf5_file_handle, dataset_name.c_str());
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

  return std::vector<uint>(size.begin(), size.end());
}
//-----------------------------------------------------------------------------
bool HDF5Interface::dataset_exists(const HDF5File& hdf5_file,
                                   const std::string dataset_name,
                                   const bool use_mpi_io)
{
  // HDF5 filename
  const std::string filename = hdf5_file.name();

  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_file(filename, use_mpi_io);

  // Disable error reporting
  herr_t (*old_func)(void*);
  void *old_client_data;
  H5Eget_auto(&old_func, &old_client_data);

  // Redirect error reporting (to none)
  status = H5Eset_auto(NULL, NULL);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open dataset - returns HDF5_FAIL if non-existent
  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  if(dset_id != HDF5_FAIL)
    H5Dclose(dset_id);

  // Re-enable error reporting
  status = H5Eset_auto(old_func, old_client_data);
  dolfin_assert(status != HDF5_FAIL);

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  // Return true if dataset exists
  return (dset_id != HDF5_FAIL);
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
  hid_t group_id = H5Gopen(hdf5_file_handle, group_name.c_str());
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
std::vector<std::string> HDF5Interface::dataset_list(const std::string filename,
                                                 const std::string group_name,
                                                 const bool use_mpi_io)
{
  // List all member datasets of a group by name
  char namebuf[HDF5_MAXSTRLEN];

  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_file(filename, use_mpi_io);

  // Open group by name group_name
  hid_t group_id = H5Gopen(file_id,group_name.c_str());
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

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return list_of_datasets;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint>
  HDF5Interface::dataset_dimensions(const std::string filename,
                                    const std::string dataset_name,
                                    const bool use_mpi_io)
{
  // Current dataset dimensions
  hsize_t cur_size[10];

  // Maximum dataset dimensions
  hsize_t max_size[10];

  herr_t status;

  // Try to open existing HDF5 file
  const hid_t file_id = open_file(filename, use_mpi_io);

  // Open named dataset
  const hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Get the dataspace of the dataset
  const hid_t space = H5Dget_space(dset_id);
  dolfin_assert(space != HDF5_FAIL);

  // Enquire dimensions of the dataspace
  const int ndims = H5Sget_simple_extent_dims(space, cur_size, max_size);
  dolfin_assert(ndims == 2); // ensure it is a 2D dataset

  // Close dataset collectively
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return std::pair<uint, uint>(cur_size[0],cur_size[1]);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5Interface::get_attribute(const std::string filename,
                                  const std::string dataset_name,
                                  const std::string attribute_name,
                                  T& attribute_value,
                                  const bool use_mpio)
{
  herr_t status;

  // Try to open existing HDF5 file
  const hid_t file_id = open_file(filename, use_mpio);

  // Open dataset by name
  const hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
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

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
hid_t HDF5Interface::open_file(const std::string filename,
                               const bool use_mpiio)
{
  herr_t status;

  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  dolfin_assert(plist_id != HDF5_FAIL);
  if (use_mpiio)
  {
    #ifdef HAS_MPI
    MPICommunicator comm;
    MPIInfo info;
    status = H5Pset_fapl_mpio(plist_id, *comm, *info);
    dolfin_assert(status != HDF5_FAIL);
    #else
    dolfin_error("HDF5Interface.cpp",
                 "create file",
                 "Cannot use MPI-IO output if DOLFIN is not configured with MPI");
    #endif
  }

  // Try to open existing HDF5 file
  const hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // Release file-access template
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  return file_id;
}
//-----------------------------------------------------------------------------
