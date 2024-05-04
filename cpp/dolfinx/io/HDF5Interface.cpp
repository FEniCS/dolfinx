// Copyright (C) 2012 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Interface.h"
#include <filesystem>

using namespace dolfinx;

namespace
{
/// Check for existence of group in HDF5 file
/// @param[in] handle HDF5 file handle
/// @param[in] group_name Name of the group to check
/// @return True if @p group_name is in the file
bool has_group(hid_t handle, const std::string& group_name)
{
  const hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  if (lapl_id < 0)
    throw std::runtime_error("Failed to create HDF5 property list");

  htri_t link_status = H5Lexists(handle, group_name.c_str(), lapl_id);
  if (link_status < 0)
    throw std::runtime_error("Failed to check existence of HDF5 link in group");

  if (link_status == 0)
  {
    if (H5Pclose(lapl_id) < 0)
      throw std::runtime_error("Call to H5Pclose unsuccessful");
    return false;
  }

  H5O_info_t object_info;
#if H5_VERSION_GE(1, 12, 0)
  herr_t err = H5Oget_info_by_name3(handle, group_name.c_str(), &object_info,
                                    H5O_INFO_ALL, lapl_id);
#else
  herr_t err
      = H5Oget_info_by_name1(handle, group_name.c_str(), &object_info, lapl_id);
#endif
  if (err < 0)
    throw std::runtime_error("Call to H5Oget_info_by_name unsuccessful");

  if (H5Pclose(lapl_id) < 0)
    throw std::runtime_error("Call to H5Pclose unsuccessful");

  return object_info.type == H5O_TYPE_GROUP;
}

} // namespace

//-----------------------------------------------------------------------------
hid_t io::hdf5::open_file(MPI_Comm comm, const std::filesystem::path& filename,
                          const std::string& mode, bool use_mpi_io)
{
  // Set parallel access with communicator
  const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

  if (use_mpi_io)
  {
    MPI_Info info;
    MPI_Info_create(&info);
    if (H5Pset_fapl_mpio(plist_id, comm, info) < 0)
      throw std::runtime_error("Call to H5Pset_fapl_mpio unsuccessful");
    MPI_Info_free(&info);
  }

  hid_t file_id = -1;
  if (mode == "w") // Create file for write, overwriting any existing file
  {
    if (auto d = filename.parent_path(); !d.empty())
      std::filesystem::create_directories(d);
    file_id = H5Fcreate(filename.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    if (file_id < 0)
      throw std::runtime_error("Failed to create HDF5 file.");
  }
  else if (mode == "a") // Open file to append, creating if does not exist
  {
    if (std::filesystem::exists(filename))
      file_id = H5Fopen(filename.string().c_str(), H5F_ACC_RDWR, plist_id);
    else
    {
      if (auto d = filename.parent_path(); !d.empty())
        std::filesystem::create_directories(d);
      file_id
          = H5Fcreate(filename.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    }

    if (file_id < 0)
    {
      throw std::runtime_error(
          "Failed to create/open HDF5 file (append mode).");
    }
  }
  else if (mode == "r") // Open file to read
  {
    if (std::filesystem::exists(filename))
    {
      file_id = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, plist_id);
      if (file_id < 0)
        throw std::runtime_error("Failed to open HDF5 file.");
    }
    else
    {
      throw std::runtime_error("Unable to open HDF5 file. File "
                               + filename.string() + " does not exist.");
    }
  }

  if (H5Pclose(plist_id) < 0)
    throw std::runtime_error("Failed to close HDF5 file property list.");

  return file_id;
}
//-----------------------------------------------------------------------------
void io::hdf5::close_file(hid_t handle)
{
  if (H5Fclose(handle) < 0)
    throw std::runtime_error("Failed to close HDF5 file.");
}
//-----------------------------------------------------------------------------
void io::hdf5::flush_file(hid_t handle)
{
  if (H5Fflush(handle, H5F_SCOPE_GLOBAL) < 0)
    throw std::runtime_error("Failed to flush HDF5 file.");
}
//-----------------------------------------------------------------------------
std::filesystem::path io::hdf5::get_filename(hid_t handle)
{
  // Get length of filename
  const ssize_t length = H5Fget_name(handle, nullptr, 0);
  if (length < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  // Allocate memory
  std::vector<char> name(length + 1);

  // Retrieve filename
  if (H5Fget_name(handle, name.data(), length + 1) < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  return std::filesystem::path(name.begin(), name.end());
}
//-----------------------------------------------------------------------------
bool io::hdf5::has_dataset(hid_t handle, const std::string& dataset_path)
{
  const hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
  if (lapl_id < 0)
    throw std::runtime_error("Failed to create HDF5 property list");

  const htri_t link_status = H5Lexists(handle, dataset_path.c_str(), lapl_id);
  if (link_status < 0)
    throw std::runtime_error("Failed to check existence of HDF5 link in group");

  if (H5Pclose(lapl_id) < 0)
    throw std::runtime_error("Call to H5Pclose unsuccessful");

  return link_status;
}
//-----------------------------------------------------------------------------
hid_t io::hdf5::open_dataset(hid_t handle, const std::string& path)
{
  return H5Dopen2(handle, path.c_str(), H5P_DEFAULT);
}
//-----------------------------------------------------------------------------
void io::hdf5::add_group(hid_t handle, const std::string& group_name)
{
  std::string _group_name(group_name);

  // Cannot create the root level group
  if (_group_name.size() == 0 or _group_name == "/")
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
    if (!has_group(handle, parent_name))
    {
      const hid_t group_id_vis = H5Gcreate2(
          handle, parent_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (group_id_vis < 0)
        throw std::runtime_error("Failed to add HDF5 group");

      if (H5Gclose(group_id_vis) < 0)
        throw std::runtime_error("Failed to close HDF5 group");
    }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
io::hdf5::get_dataset_shape(hid_t handle, const std::string& dataset_path)
{
  // Open named dataset
  const hid_t dset_id = H5Dopen2(handle, dataset_path.c_str(), H5P_DEFAULT);
  if (dset_id < 0)
    throw std::runtime_error("Failed to open HDF5 dataset by name");

  const hid_t space = H5Dget_space(dset_id);
  if (space < 0)
    throw std::runtime_error("Failed to get dataspace of dataset");

  // Get rank
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0)
    throw std::runtime_error("Failed to get dimensionality of dataspace");

  // Get size in each dimension
  std::vector<hsize_t> size(rank);
  const int ndims = H5Sget_simple_extent_dims(space, size.data(), nullptr);
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
void io::hdf5::set_mpi_atomicity(hid_t handle, bool atomic)
{
  if (H5Fset_mpi_atomicity(handle, atomic) < 0)
    throw std::runtime_error("Setting the MPI atomicity flag failed");
}
//-----------------------------------------------------------------------------
bool io::hdf5::get_mpi_atomicity(hid_t handle)
{
  hbool_t atomic = false;
  if (H5Fget_mpi_atomicity(handle, &atomic) < 0)
    throw std::runtime_error("Getting the MPI atomicity flag failed");
  else
    return static_cast<bool>(atomic);
}
//-----------------------------------------------------------------------------
