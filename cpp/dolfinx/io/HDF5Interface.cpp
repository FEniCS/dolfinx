// Copyright (C) 2012-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HDF5Interface.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace dolfinx;

namespace
{
/// @brief Check for an object of a given type in an HDF5 file.
/// @param[in] handle HDF5 file handle.
/// @param[in] name Path of the object to check.
/// @param[in] type Object type to check for.
/// @return True if @p name is in the file and has type @p type.
bool has_object(hid_t handle, std::string_view name, H5O_type_t type)
{
  const std::string name_str(name);
  htri_t link_status = H5Lexists(handle, name_str.c_str(), H5P_DEFAULT);
  if (link_status < 0)
    throw std::runtime_error("Failed to check existence of HDF5 link in group");
  if (link_status == 0)
    return false;

  H5O_info_t object_info;
#if H5_VERSION_GE(1, 12, 0)
  herr_t err = H5Oget_info_by_name3(handle, name_str.c_str(), &object_info,
                                    H5O_INFO_BASIC, H5P_DEFAULT);
#else
  herr_t err = H5Oget_info_by_name1(handle, name_str.c_str(), &object_info,
                                    H5P_DEFAULT);
#endif
  if (err < 0)
    throw std::runtime_error("Call to H5Oget_info_by_name unsuccessful");

  return object_info.type == type;
}

/// @brief Remove an attribute if it is already present.
///
/// The datatype and dataspace of an existing attribute are fixed, so a
/// value of a different size or length can only be stored by replacing
/// the attribute.
/// @param[in] handle Dataset or group handle.
/// @param[in] name Name of attribute.
void delete_attribute(hid_t handle, const std::string& name)
{
  htri_t attr_exists = H5Aexists(handle, name.c_str());
  if (attr_exists < 0)
    throw std::runtime_error("Failed to check for HDF5 attribute.");
  if (attr_exists > 0 and H5Adelete(handle, name.c_str()) < 0)
    throw std::runtime_error("Failed to delete HDF5 attribute.");
}

/// @brief Create an attribute and write a value to it.
/// @param[in] handle Dataset or group handle.
/// @param[in] name Name of attribute.
/// @param[in] type Datatype of the attribute.
/// @param[in] space Dataspace of the attribute.
/// @param[in] value Buffer holding the value to write.
void write_attribute(hid_t handle, const std::string& name, hid_t type,
                     hid_t space, const void* value)
{
  const io::hdf5::Handle attr_id(
      H5Acreate(handle, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT),
      H5Aclose);
  if (attr_id < 0)
    throw std::runtime_error("Failed to create HDF5 attribute.");
  if (H5Awrite(attr_id, type, value) < 0)
    throw std::runtime_error("Failed to write HDF5 attribute.");
}

/// @brief Replace an attribute with a 1D array of 32-bit integers.
/// @param[in] handle Dataset or group handle.
/// @param[in] name Name of attribute.
/// @param[in] value Values to write.
void set_attribute_int32(hid_t handle, const std::string& name,
                         std::span<const std::int32_t> value)
{
  delete_attribute(handle, name);

  hsize_t dims = value.size();
  const io::hdf5::Handle space_id(H5Screate_simple(1, &dims, nullptr),
                                  H5Sclose);
  if (space_id < 0)
    throw std::runtime_error("Failed to create HDF5 attribute dataspace.");

  write_attribute(handle, name, H5T_NATIVE_INT32, space_id, value.data());
}
} // namespace

//-----------------------------------------------------------------------------
hid_t io::hdf5::open_file(MPI_Comm comm, const std::filesystem::path& filename,
                          std::string_view mode, bool use_mpi_io)
{
  if (mode != "w" and mode != "a" and mode != "r")
  {
    throw std::invalid_argument(std::format(
        "Unknown HDF5 file mode \"{}\". Use \"w\", \"a\" or \"r\".", mode));
  }

  // Set parallel access with communicator
  const Handle plist_id(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  if (plist_id < 0)
    throw std::runtime_error("Failed to create HDF5 file access list.");

  if (use_mpi_io)
  {
    MPI_Info info;
    MPI_Info_create(&info);
    herr_t err = H5Pset_fapl_mpio(plist_id, comm, info);
    MPI_Info_free(&info);
    if (err < 0)
      throw std::runtime_error("Call to H5Pset_fapl_mpio unsuccessful");
  }

  hid_t file_id = H5I_INVALID_HID;
  if (mode == "w") // Create file for write, overwriting any existing file
  {
    if (auto d = filename.parent_path(); !d.empty())
      std::filesystem::create_directories(d);
    file_id = H5Fcreate(filename.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                        plist_id);
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
      file_id = H5Fcreate(filename.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT,
                          plist_id);
    }

    if (file_id < 0)
    {
      throw std::runtime_error(
          "Failed to create/open HDF5 file (append mode).");
    }
  }
  else // Open file to read
  {
    if (!std::filesystem::exists(filename))
    {
      throw std::runtime_error(
          std::format("Unable to open HDF5 file. File {} does not exist.",
                      filename.string()));
    }

    file_id = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, plist_id);
    if (file_id < 0)
      throw std::runtime_error("Failed to open HDF5 file.");
  }

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
  // Length excludes the null terminator
  const ssize_t length = H5Fget_name(handle, nullptr, 0);
  if (length < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  // Build the path from the null-terminated buffer, so that the
  // terminator is not part of the path
  std::vector<char> name(length + 1);
  if (H5Fget_name(handle, name.data(), name.size()) < 0)
    throw std::runtime_error("Failed to get HDF5 filename from handle.");

  return std::filesystem::path(name.data());
}
//-----------------------------------------------------------------------------
bool io::hdf5::has_dataset(hid_t handle, std::string_view dataset_path)
{
  return has_object(handle, dataset_path, H5O_TYPE_DATASET);
}
//-----------------------------------------------------------------------------
void io::hdf5::set_attribute(hid_t handle, std::string_view attr_name,
                             std::string_view value)
{
  const std::string name(attr_name);
  const std::string val(value);
  delete_attribute(handle, name);

  const Handle atype(H5Tcopy(H5T_C_S1), H5Tclose);
  if (atype < 0)
    throw std::runtime_error("Failed to copy HDF5 string datatype.");

  // A fixed-length string datatype must have non-zero size, so an empty
  // value is stored as a single null character
  if (H5Tset_size(atype, std::max<std::size_t>(val.size(), 1)) < 0)
    throw std::runtime_error("Failed to set HDF5 string datatype size.");
  if (H5Tset_strpad(atype, H5T_STR_NULLTERM) < 0)
    throw std::runtime_error("Failed to set HDF5 string datatype padding.");

  const Handle space_id(H5Screate(H5S_SCALAR), H5Sclose);
  if (space_id < 0)
    throw std::runtime_error("Failed to create HDF5 attribute dataspace.");

  write_attribute(handle, name, atype, space_id, val.c_str());
}
//-----------------------------------------------------------------------------
void io::hdf5::set_attribute(hid_t handle, std::string_view attr_name,
                             std::int32_t value)
{
  set_attribute_int32(handle, std::string(attr_name), {&value, 1});
}
//-----------------------------------------------------------------------------
void io::hdf5::set_attribute(hid_t handle, std::string_view attr_name,
                             const std::vector<std::int32_t>& value)
{
  set_attribute_int32(handle, std::string(attr_name), value);
}
//-----------------------------------------------------------------------------
hid_t io::hdf5::open_dataset(hid_t handle, std::string_view path)
{
  hid_t dset_id = H5Dopen2(handle, std::string(path).c_str(), H5P_DEFAULT);
  if (dset_id < 0)
  {
    throw std::runtime_error(
        std::format("Failed to open HDF5 dataset \"{}\".", path));
  }
  return dset_id;
}
//-----------------------------------------------------------------------------
void io::hdf5::add_group(hid_t handle, std::string_view dataset_path)
{
  std::string _group_name(dataset_path);

  // Cannot create the root level group
  if (_group_name.size() == 0 or _group_name == "/")
    return;

  // Prepend a slash if missing
  if (_group_name[0] != '/')
    _group_name = std::format("/{}", _group_name);

  // Starting from the root level, check and create groups if needed
  std::size_t pos = 0;
  while (pos != std::string::npos)
  {
    pos++;
    pos = _group_name.find('/', pos);
    const std::string parent_name(_group_name, 0, pos);
    if (!has_object(handle, parent_name, H5O_TYPE_GROUP))
    {
      const Handle group_id(H5Gcreate2(handle, parent_name.c_str(), H5P_DEFAULT,
                                       H5P_DEFAULT, H5P_DEFAULT),
                            H5Gclose);
      if (group_id < 0)
        throw std::runtime_error("Failed to add HDF5 group");
    }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
io::hdf5::get_dataset_shape(hid_t handle, std::string_view dataset_path)
{
  // Open named dataset
  const Handle dset_id(open_dataset(handle, dataset_path), H5Dclose);
  const Handle space(H5Dget_space(dset_id), H5Sclose);
  if (space < 0)
    throw std::runtime_error("Failed to get dataspace of dataset");

  // Get rank
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0)
    throw std::runtime_error("Failed to get dimensionality of dataspace");

  // Get size in each dimension
  std::vector<hsize_t> size(rank);
  if (H5Sget_simple_extent_dims(space, size.data(), nullptr) != rank)
    throw std::runtime_error("Failed to get dimensionality of dataspace");

  return std::vector<std::int64_t>(size.begin(), size.end());
}
//-----------------------------------------------------------------------------
