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
// Last Changed: 2013-10-24

#include <boost/filesystem.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "HDF5File.h"
#include "HDF5Interface.h"

#ifdef HAS_HDF5

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

using namespace dolfin;

//-----------------------------------------------------------------------------
hid_t HDF5Interface::open_file(MPI_Comm mpi_comm, const std::string filename,
                               const std::string mode,
                               const bool use_mpi_io)
{
  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  if (use_mpi_io)
  {
    #ifdef HAS_MPI
    MPI_Info info;
    MPI_Info_create(&info);
    herr_t status = H5Pset_fapl_mpio(plist_id, mpi_comm, info);
    dolfin_assert(status != HDF5_FAIL);
    MPI_Info_free(&info);
    #else
    dolfin_error("HDF5Interface.cpp",
                 "create HDF5 file",
                 "Cannot use MPI-IO output if DOLFIN is not configured with MPI");
    #endif
  }

  hid_t file_id = HDF5_FAIL;
  if (mode == "w")
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
                   "File \"%s\" does not exist", filename.c_str());
    }

    if (mode == "a")
    {
      // Open file existing file for append
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
    }
    else if (mode == "r")
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
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
void HDF5Interface::close_file(const hid_t hdf5_file_handle)
{
  herr_t status = H5Fclose(hdf5_file_handle);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
void HDF5Interface::flush_file(const hid_t hdf5_file_handle)
{
  herr_t status = H5Fflush(hdf5_file_handle, H5F_SCOPE_GLOBAL);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
std::string HDF5Interface::get_filename(hid_t hdf5_file_handle)
{
  // Get length of filename
  ssize_t length = H5Fget_name(hdf5_file_handle, NULL, 0);
  dolfin_assert(length > 0);

  // Allocate memory
  std::vector<char> name(length +1);

  // Retrive filename
  length = H5Fget_name(hdf5_file_handle, name.data(), length + 1);
  dolfin_assert(length > 0);

  return std::string(name.begin(), name.end());
}
//-----------------------------------------------------------------------------
const std::string HDF5Interface::get_attribute_type(
                  const hid_t hdf5_file_handle,
                  const std::string dataset_name,
                  const std::string attribute_name)
{
  herr_t status;

  // Open dataset or group by name
  const hid_t dset_id = H5Oopen(hdf5_file_handle, dataset_name.c_str(),
                                H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Open attribute by name and get its type
  const hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
  dolfin_assert(attr_id != HDF5_FAIL);
  const hid_t attr_type = H5Aget_type(attr_id);
  dolfin_assert(attr_type != HDF5_FAIL);

  // Determine type of attribute
  const hid_t h5class = H5Tget_class(attr_type);

  // Get size of space, will determine if it is a vector or not
  const hid_t dataspace = H5Aget_space(attr_id);
  dolfin_assert(dataspace != HDF5_FAIL);
  hsize_t cur_size[10];
  hsize_t max_size[10];
  const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);

  std::string attribute_type_description;

  if (h5class == H5T_FLOAT && ndims == 0)
    attribute_type_description = "float";
  else if (h5class == H5T_INTEGER && ndims == 0)
    attribute_type_description = "int";
  else if (h5class == H5T_FLOAT)
    attribute_type_description = "vectorfloat";
  else if (h5class == H5T_INTEGER)
    attribute_type_description = "vectorint";
  else if (h5class == H5T_STRING)
    attribute_type_description = "string";
  else
    attribute_type_description = "unsupported";

  // Close dataspace
  status = H5Sclose(dataspace);
  dolfin_assert(status != HDF5_FAIL);

  // Close attribute type
  status = H5Tclose(attr_type);
  dolfin_assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attr_id);
  dolfin_assert(status != HDF5_FAIL);

  // Close dataset or group
  status = H5Oclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  return attribute_type_description;
}
//-----------------------------------------------------------------------------
void HDF5Interface::delete_attribute(const hid_t hdf5_file_handle,
                                     const std::string dataset_name,
                                     const std::string attribute_name)
{
  herr_t status;

  // Open dataset or group by name
  const hid_t dset_id = H5Oopen(hdf5_file_handle, dataset_name.c_str(),
                                H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Delete attribute by name
  status = H5Adelete(dset_id, attribute_name.c_str());
  dolfin_assert(status != HDF5_FAIL);

  // Close dataset or group
  status = H5Oclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
herr_t HDF5Interface::attribute_iteration_function(hid_t loc_id,
                                                   const char* name,
                                                   const H5A_info_t* info,
                                                   void *str)
{
  std::vector<std::string>* s = (std::vector<std::string>*)str;
  std::string attr_name(name);
  s->push_back(name);
  return 0;
}
//-----------------------------------------------------------------------------
const std::vector<std::string>
HDF5Interface::list_attributes(const hid_t hdf5_file_handle,
                               const std::string dataset_name)
{
  // Open dataset or group by name
  const hid_t dset_id = H5Oopen(hdf5_file_handle, dataset_name.c_str(),
                                H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  hsize_t n = 0;
  std::vector<std::string> out_string;
  herr_t status = H5Aiterate2(dset_id, H5_INDEX_NAME, H5_ITER_INC, &n,
                              attribute_iteration_function,
                              (void *)&out_string);
  dolfin_assert(status != HDF5_FAIL);

  // Close dataset or group
  status = H5Oclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  return out_string;
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_attribute(const hid_t hdf5_file_handle,
                                  const std::string dataset_name,
                                  const std::string attribute_name)
{
  herr_t status;
  htri_t has_attr;

  // Open dataset or group by name
  const hid_t dset_id = H5Oopen(hdf5_file_handle, dataset_name.c_str(),
                                H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Check for attribute by name
  has_attr = H5Aexists(dset_id, attribute_name.c_str());
  dolfin_assert(has_attr != HDF5_FAIL);

  // Close dataset or group
  status = H5Oclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  return (has_attr > 0);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  herr_t status;
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  htri_t link_status = H5Lexists(hdf5_file_handle, group_name.c_str(), lapl_id);
  dolfin_assert(link_status >= 0);
  if(link_status==0)
  {
    // Close link access properties
    status = H5Pclose(lapl_id);
    dolfin_assert(status != HDF5_FAIL);
    return false;
  }

  H5O_info_t object_info;
  H5Oget_info_by_name(hdf5_file_handle, group_name.c_str(), &object_info,
                      lapl_id);

  // Close link access properties
  status = H5Pclose(lapl_id);
  dolfin_assert(status != HDF5_FAIL);

  return (object_info.type == H5O_TYPE_GROUP);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_dataset(const hid_t hdf5_file_handle,
                                const std::string dataset_name)
{
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  htri_t link_status = H5Lexists(hdf5_file_handle, dataset_name.c_str(), lapl_id);
  dolfin_assert(link_status >= 0);

  // Close link access properties
  herr_t status = H5Pclose(lapl_id);
  dolfin_assert(status != HDF5_FAIL);

  return link_status;
}
//-----------------------------------------------------------------------------
void HDF5Interface::add_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  std::string _group_name(group_name);

  // Cannot create the root level group
  if (_group_name.size() == 0 || _group_name == "/")
    return;

  // Prepend a slash if missing
  if (_group_name[0] != '/')
    _group_name = "/" + _group_name;

  // Starting from the root level, check and create groups if needed
  std::size_t pos = 0;
  while (pos != std::string::npos)
  {
    pos++;
    pos = _group_name.find('/', pos);
    const std::string parent_name(_group_name, 0, pos);

    if (!has_group(hdf5_file_handle, parent_name))
    {
      hid_t group_id_vis = H5Gcreate2(hdf5_file_handle, parent_name.c_str(),
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      dolfin_assert(group_id_vis != HDF5_FAIL);

      herr_t status = H5Gclose(group_id_vis);
      dolfin_assert(status != HDF5_FAIL);
    }
  }
}
//-----------------------------------------------------------------------------
int HDF5Interface::dataset_rank(const hid_t hdf5_file_handle,
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
std::vector<std::int64_t>
HDF5Interface::get_dataset_shape(const hid_t hdf5_file_handle,
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

  return std::vector<std::int64_t>(size.begin(), size.end());
}
//-----------------------------------------------------------------------------
int HDF5Interface::num_datasets_in_group(const hid_t hdf5_file_handle,
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
std::vector<std::string>
HDF5Interface::dataset_list(const hid_t hdf5_file_handle,
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
  for (hsize_t i = 0; i < num_datasets; i++)
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
