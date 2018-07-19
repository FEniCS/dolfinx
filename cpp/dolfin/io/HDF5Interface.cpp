// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Interface.h"
#include "HDF5File.h"
#include <boost/filesystem.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>

#define HDF5_MAXSTRLEN 80

using namespace dolfin;
using namespace dolfin::io;

//-----------------------------------------------------------------------------
hid_t HDF5Interface::open_file(MPI_Comm mpi_comm, const std::string filename,
                               const std::string mode, const bool use_mpi_io)
{
  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

#ifdef H5_HAVE_PARALLEL
  if (use_mpi_io)
  {
    MPI_Info info;
    MPI_Info_create(&info);
    herr_t status = H5Pset_fapl_mpio(plist_id, mpi_comm, info);
    assert(status >= 0);
    MPI_Info_free(&info);
  }
#endif

  hid_t file_id = -1;
  if (mode == "w") // Create file for write, overwriting any existing file
  {
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    if (file_id < 0)
      throw std::runtime_error("Failed to create HDF5 file.");
  }
  else if (mode == "a") // Open file to append, creating if does not exist
  {
    if (boost::filesystem::exists(filename))
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
    else
    {
      file_id
          = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    }
    if (file_id < 0)
      throw std::runtime_error(
          "Failed to create/open HDF5 file (append mode).");
  }
  else if (mode == "r") // Open file to read
  {
    if (boost::filesystem::exists(filename))
    {
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
      if (file_id < 0)
        throw std::runtime_error("Failed to open HDF5 file.");
    }
    else
    {
      throw std::runtime_error("Unable to open HDF5 file. File " + filename
                               + " does not exist.");
    }
  }

  if (H5Pclose(plist_id) < 0)
    throw std::runtime_error("Failed to close HDF5 file property list.");

  return file_id;
}
//-----------------------------------------------------------------------------
void HDF5Interface::close_file(const hid_t hdf5_file_handle)
{
  if (H5Fclose(hdf5_file_handle) < 0)
    throw std::runtime_error("Failed to close HDF5 file.");
}
//-----------------------------------------------------------------------------
void HDF5Interface::flush_file(const hid_t hdf5_file_handle)
{
  if (H5Fflush(hdf5_file_handle, H5F_SCOPE_GLOBAL) < 0)
    throw std::runtime_error("Failed to flush HDF5 file.");
}
//-----------------------------------------------------------------------------
std::string HDF5Interface::get_filename(hid_t hdf5_file_handle)
{
  // Get length of filename
  const ssize_t length = H5Fget_name(hdf5_file_handle, NULL, 0);
  if (length < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  // Allocate memory
  std::vector<char> name(length + 1);

  // Retrieve filename
  if (H5Fget_name(hdf5_file_handle, name.data(), length + 1) < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  return std::string(name.begin(), name.end());
}
//-----------------------------------------------------------------------------
const std::string
HDF5Interface::get_attribute_type(const hid_t hdf5_file_handle,
                                  const std::string dataset_path,
                                  const std::string attribute_name)
{
  // Open dataset or group by name
  const hid_t dset_id
      = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset.");

  // Open attribute by name and get its type
  const hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
  if (attr_id < 0)
    throw std::runtime_error("Failed to open HDF5 attribute.");
  const hid_t attr_type = H5Aget_type(attr_id);
  if (attr_type < 0)
    throw std::runtime_error("Failed to get HDF5 attribute type.");

  // Determine type of attribute
  const hid_t h5class = H5Tget_class(attr_type);
  if (h5class < 0)
    throw std::runtime_error("Failed to get HDF5 attribute type.");

  // Get size of space, will determine if it is a vector or not
  const hid_t dataspace = H5Aget_space(attr_id);
  if (dataspace < 0)
    throw std::runtime_error("Failed to get HDF5 dataspace.");
  hsize_t cur_size[10];
  hsize_t max_size[10];
  const int ndims = H5Sget_simple_extent_dims(dataspace, cur_size, max_size);
  if (ndims < 0)
    throw std::runtime_error("Call to H5Sget_simple_extent_dims unsuccessful");

  // FIXME: Use std::map (put in anonymous namespace)
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
  if (H5Sclose(dataspace) < 0)
    throw std::runtime_error("Call to H5Sclose unsuccessful");

  // Close attribute type
  if (H5Tclose(attr_type) < 0)
    throw std::runtime_error("Call to H5Tclose unsuccessful");

  // Close attribute
  if (H5Aclose(attr_id) < 0)
    throw std::runtime_error("Call to H5Aclose unsuccessful");

  // Close dataset or group
  if (H5Oclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Oclose unsuccessful");

  return attribute_type_description;
}
//-----------------------------------------------------------------------------
void HDF5Interface::delete_attribute(const hid_t hdf5_file_handle,
                                     const std::string dataset_path,
                                     const std::string attribute_name)
{
  // Open dataset or group by name
  hid_t dset_id
      = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset");

  // Delete attribute by name
  herr_t status = H5Adelete(dset_id, attribute_name.c_str());
  if (status < 0)
    throw std::runtime_error("Failed to delete HDF5 attribute");

  // Close dataset or group
  if (H5Oclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Oclose unsuccessful");
}
//-----------------------------------------------------------------------------
herr_t HDF5Interface::attribute_iteration_function(hid_t loc_id,
                                                   const char* name,
                                                   const H5A_info_t* info,
                                                   void* str)
{
  std::vector<std::string>* s = (std::vector<std::string>*)str;
  std::string attr_name(name);
  s->push_back(name);
  return 0;
}
//-----------------------------------------------------------------------------
const std::vector<std::string>
HDF5Interface::list_attributes(const hid_t hdf5_file_handle,
                               const std::string dataset_path)
{
  // Open dataset or group by name
  hid_t dset_id
      = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset");

  hsize_t n = 0;
  std::vector<std::string> out_string;
  herr_t status = H5Aiterate2(dset_id, H5_INDEX_NAME, H5_ITER_INC, &n,
                              attribute_iteration_function, (void*)&out_string);
  if (status < 0)
    throw std::runtime_error("Failed to iterate over attributes");

  // Close dataset or group
  if (H5Oclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Oclose unsuccessful");

  return out_string;
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_attribute(const hid_t hdf5_file_handle,
                                  const std::string dataset_path,
                                  const std::string attribute_name)
{
  // Open dataset or group by name
  hid_t dset_id
      = H5Oopen(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset");

  // Check for attribute by name
  htri_t has_attr = H5Aexists(dset_id, attribute_name.c_str());
  if (has_attr < 0)
    throw std::runtime_error("Failed to check HDF5 attribute existence");

  // Close dataset or group
  if (H5Oclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Oclose unsuccessful");

  return (has_attr > 0);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_group(const hid_t hdf5_file_handle,
                              const std::string group_name)
{
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  if (lapl_id < 0)
    throw std::runtime_error("Failed to create HDF5 property list");

  htri_t link_status = H5Lexists(hdf5_file_handle, group_name.c_str(), lapl_id);
  if (link_status < 0)
    throw std::runtime_error("Failed to check existence of HDF5 link in group");

  if (link_status == 0)
  {
    if (H5Pclose(lapl_id) < 0)
      throw std::runtime_error("Call to H5Pclose unsuccessful");
    return false;
  }

  H5O_info_t object_info;
  H5Oget_info_by_name(hdf5_file_handle, group_name.c_str(), &object_info,
                      lapl_id);

  if (H5Pclose(lapl_id) < 0)
    throw std::runtime_error("Call to H5Pclose unsuccessful");

  return (object_info.type == H5O_TYPE_GROUP);
}
//-----------------------------------------------------------------------------
bool HDF5Interface::has_dataset(const hid_t hdf5_file_handle,
                                const std::string dataset_path)
{
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  if (lapl_id < 0)
    throw std::runtime_error("Failed to create HDF5 property list");

  htri_t link_status
      = H5Lexists(hdf5_file_handle, dataset_path.c_str(), lapl_id);
  if (link_status < 0)
    throw std::runtime_error("Failed to check existence of HDF5 link in group");

  if (H5Pclose(lapl_id) < 0)
    throw std::runtime_error("Call to H5Pclose unsuccessful");

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
      if (group_id_vis < 0)
        throw std::runtime_error("Failed to add HDF5 group");

      if (H5Gclose(group_id_vis) < 0)
        throw std::runtime_error("Failed to close HDF5 group");
    }
  }
}
//-----------------------------------------------------------------------------
int HDF5Interface::dataset_rank(const hid_t hdf5_file_handle,
                                const std::string dataset_path)
{
  // Open dataset
  hid_t dset_id
      = H5Dopen2(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("");

  // Get the dataspace of the dataset
  hid_t space = H5Dget_space(dset_id);
  if (space < 0)
    throw std::runtime_error("Failed to get HDF5 dataspace");

  // Get dataset rank
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0)
    throw std::runtime_error("Failed to get dimensionality of dataspace");

  // Close dataspace and dataset
  if (H5Sclose(space) < 0)
    throw std::runtime_error("Call to H5Sclose unsuccessful");
  if (H5Dclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Dclose unsuccessful");

  return rank;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
HDF5Interface::get_dataset_shape(const hid_t hdf5_file_handle,
                                 const std::string dataset_path)
{
  // Open named dataset
  const hid_t dset_id
      = H5Dopen2(hdf5_file_handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset by name");

  const hid_t space = H5Dget_space(dset_id);
  if (space < 0)
    throw std::runtime_error("Failed to get dataspace of dataset");

  // Get rank
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0)
    throw std::runtime_error("Failed to get dimensionality of dataspace");

  // Allocate data
  std::vector<hsize_t> size(rank);

  // Get size in each dimension
  const int ndims = H5Sget_simple_extent_dims(space, size.data(), NULL);
  if (ndims < 0)
    throw std::runtime_error("Failed to get dimensionality of dataspace");
  assert(ndims == rank);

  // Close dataspace and dataset
  if (H5Sclose(space) < 0)
    throw std::runtime_error("Call to H5Sclose unsuccessful");
  if (H5Dclose(dset_id) < 0)
    throw std::runtime_error("Call to H5Dclose unsuccessful");

  return std::vector<std::int64_t>(size.begin(), size.end());
}
//-----------------------------------------------------------------------------
int HDF5Interface::num_datasets_in_group(const hid_t hdf5_file_handle,
                                         const std::string group_name)
{
  // Get group info by name
  H5G_info_t group_info;
  hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  if (lapl_id < 0)
    throw std::runtime_error("Failed to create HDF5 property list");

  herr_t status = H5Gget_info_by_name(hdf5_file_handle, group_name.c_str(),
                                      &group_info, lapl_id);
  if (status < 0)
    throw std::runtime_error("Call to H5Gget_info_by_name unsuccessful");

  return group_info.nlinks;
}
//-----------------------------------------------------------------------------
std::vector<std::string>
HDF5Interface::dataset_list(const hid_t hdf5_file_handle,
                            const std::string group_name)
{
  // List all member datasets of a group by name
  char namebuf[HDF5_MAXSTRLEN];

  // Open group by name group_name
  hid_t group_id = H5Gopen2(hdf5_file_handle, group_name.c_str(), H5P_DEFAULT);
  if (group_id < 0)
    throw std::runtime_error("Failed to open HDF5 group by name");

  // Count how many datasets in the group
  hsize_t num_datasets;
  herr_t status = H5Gget_num_objs(group_id, &num_datasets);
  if (status < 0)
    throw std::runtime_error("Failed to count datasets in group");

  // Iterate through group collecting all dataset names
  std::vector<std::string> list_of_datasets;
  for (hsize_t i = 0; i < num_datasets; i++)
  {
    ssize_t status
        = H5Gget_objname_by_idx(group_id, i, namebuf, HDF5_MAXSTRLEN);
    if (status < 0)
      throw std::runtime_error("Call to H5Gget_objname_by_idx unsuccessful");
    list_of_datasets.push_back(std::string(namebuf));
  }

  // Close group
  if (H5Gclose(group_id) < 0)
    throw std::runtime_error("Call to H5Gclose unsuccessful");

  return list_of_datasets;
}
//-----------------------------------------------------------------------------
void HDF5Interface::set_mpi_atomicity(const hid_t hdf5_file_handle,
                                      const bool atomic)
{
#ifdef H5_HAVE_PARALLEL
  herr_t status = H5Fset_mpi_atomicity(hdf5_file_handle, atomic);
  if (status < 0)
    throw std::runtime_error("Setting the MPI atomicity flag failed");
#endif
}
//-----------------------------------------------------------------------------
bool HDF5Interface::get_mpi_atomicity(const hid_t hdf5_file_handle)
{
  hbool_t atomic = false;
#ifdef H5_HAVE_PARALLEL
  herr_t status = H5Fget_mpi_atomicity(hdf5_file_handle, &atomic);
  if (status < 0)
    throw std::runtime_error("Getting the MPI atomicity flag failed");
#endif
  return (bool)atomic;
}
//-----------------------------------------------------------------------------
