// Copyright (C) 2012 Chris N. Richardson
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
// Last Changed: 2012-09-26
// 
// 

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>

#include "HDF5Interface.h"

#define HDF5_FAIL -1
#define HDF5_MAXSTRLEN 80

using namespace dolfin;

// HDF5 calls to open a file descriptor on multiple processes
// Common file opening sequence
hid_t HDF5Interface::open_parallel_file(const std::string &filename)
{
  MPICommunicator comm;
  MPIInfo info;
  herr_t status;
  
  // Set parallel access with communicator
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  dolfin_assert(plist_id != HDF5_FAIL);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // Release file-access template 
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  return file_id;
}

//-----------------------------------------------------------------------------

void HDF5Interface::create(const std::string &filename)
{
  // make empty HDF5 file
  // overwriting any existing file
  // create some default 'folders' for storing different datasets

  hid_t  file_id;     // file and dataset identifiers
  hid_t  plist_id;    // property list identifier
  hid_t  group_id;
  herr_t status;

  MPICommunicator comm;
  MPIInfo info;

  // Set parallel access with communicator
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Create file, overwriting existing file, if present
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // Create subgroups suitable for storing different types of data.
  // DataVector - values for visualisation
  group_id = H5Gcreate(file_id, "/DataVector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  // Vector - for checkpointing etc
  group_id = H5Gcreate(file_id, "/Vector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  // Mesh
  group_id = H5Gcreate(file_id, "/Mesh", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  // Release file-access template 
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}

//-----------------------------------------------------------------------------
// Write data to existing HDF file as defined by range blocks on each process
// range: the local range on this processor
// width: is the width of the dataitem (e.g. 3 for x, y, z data)
// Versions for double, uint and int

void HDF5Interface::write(const std::string &filename,
                          const std::string dataset_name, 
                          const std::vector<double>& data,
                          const std::pair<uint, uint> range,
                          const uint width)
{
  write(filename, dataset_name, data, range, H5T_NATIVE_DOUBLE, width);
}
//-----------------------------------------------------------------------------
void HDF5Interface::write(const std::string &filename,
                          const std::string dataset_name, 
                          const std::vector<uint>& data,
                          const std::pair<uint, uint> range,
                          const uint width)
{
  write(filename, dataset_name, data, range, H5T_NATIVE_UINT, width);
}
//-----------------------------------------------------------------------------
void HDF5Interface::write(const std::string &filename,
                          const std::string dataset_name, 
                          const std::vector<int>& data,
                          const std::pair<uint, uint> range,
                          const uint width)
{
  write(filename, dataset_name, data, range, H5T_NATIVE_INT, width);
}
//-----------------------------------------------------------------------------

template <typename T>
void HDF5Interface::write(const std::string &filename,
                          const std::string dataset_name,
                          const std::vector<T>& data,
                          const std::pair<uint, uint> range,
                          const int h5type, const uint width)
{
  // Hyperslab selection parameters
  hsize_t count[2]  = {range.second - range.first, width};
  hsize_t offset[2] = {range.first, 0};

  // Dataset dimensions
  hsize_t dimsf[2] = {MPI::sum(count[0]), width};

  // Generic status report
  herr_t status;

  hid_t file_id = open_parallel_file(filename);
  
  // Create a global 2D data space
  hid_t filespace = H5Screate_simple(2, dimsf, NULL);
  dolfin_assert(filespace != HDF5_FAIL);

  // Create global dataset (using dataset_name)
  hid_t dset_id = H5Dcreate(file_id, dataset_name.c_str(), h5type, filespace,
                      H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  // Close global data space
  status = H5Sclose(filespace);
  dolfin_assert(status != HDF5_FAIL);

  // Create a local 2D data space
  hid_t memspace = H5Screate_simple(2, count, NULL);
  dolfin_assert(memspace != HDF5_FAIL);

  // Create a file dataspace within the global space - a hyperslab
  filespace = H5Dget_space(dset_id);
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
  dolfin_assert(status != HDF5_FAIL);

  // Set parallel access with communicator
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  dolfin_assert(status != HDF5_FAIL);

  // Write local dataset into selected hyperslab
  status = H5Dwrite(dset_id, h5type, memspace, filespace, plist_id, data.data());
  dolfin_assert(status != HDF5_FAIL);

  // close dataset collectively
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  // close hyperslab
  status = H5Sclose(filespace);
  dolfin_assert(status != HDF5_FAIL);

  // close local dataset
  status = H5Sclose(memspace);
  dolfin_assert(status != HDF5_FAIL);

  // Release file-access template 
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}

//-----------------------------------------------------------------------------
// Read data from a HDF5 dataset as defined by range blocks on each process
// range: the local range on this processor
// width: is the width of the dataitem (e.g. 3 for x, y, z data)
// Versions for double and uint
void HDF5Interface::read(const std::string &filename,
                         const std::string dataset_name,
                         std::vector<double>& data, const std::pair<uint, uint> range,
                         const uint width)
{
  read(filename, dataset_name, data, range, H5T_NATIVE_DOUBLE, width);
}
//-----------------------------------------------------------------------------
void HDF5Interface::read(const std::string &filename,
                         const std::string dataset_name,
                         std::vector<uint>& data, const std::pair<uint, uint> range,
                         const uint width)
{
  read(filename, dataset_name, data, range, H5T_NATIVE_UINT, width);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5Interface::read(const std::string &filename,
                         const std::string dataset_name,
                         std::vector<T>& data,  const std::pair<uint, uint> range,
                         const int h5type, const uint width)
{
  // read a generic block of 2D data from a HDF5 dataset in parallel

  dolfin_assert(data);
  if(data.size() != (range.second - range.first) * width)
    data.resize((range.second - range.first) * width);
  
  hid_t file_id;      // HDF5 file ID
  hid_t plist_id;     // File access template
  hid_t filespace;    // File dataspace ID
  hid_t memspace;     // memory dataspace ID
  hid_t dset_id;      // Dataset ID
  herr_t status;      // Generic return value

  // Hyperslab selection
  hsize_t offset[2] = {range.first, 0};
  hsize_t count[2] = {range.second - range.first, width};
  
  // Open file descriptor in parallel
  file_id = open_parallel_file(filename);

  // Open the dataset collectively
  dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Create a file dataspace independently
  filespace = H5Dget_space (dset_id);
  dolfin_assert(filespace != HDF5_FAIL);

  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                               count, NULL);
  dolfin_assert(status != HDF5_FAIL);

  // Create a memory dataspace independently
  memspace = H5Screate_simple (2, count, NULL);
  dolfin_assert (memspace != HDF5_FAIL);

  // read data independently
  status = H5Dread(dset_id, h5type, memspace, filespace,
                   H5P_DEFAULT, data.data());
  dolfin_assert(status != HDF5_FAIL);

  // close dataset collectively
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  // release all IDs created
  status = H5Sclose(filespace);
  dolfin_assert(status != HDF5_FAIL);

  // close the file collectively
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}

//-----------------------------------------------------------------------------
bool HDF5Interface::dataset_exists(const std::string &filename,const std::string &dataset_name)
{
  // Check if a named dataset exists in this file

  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_parallel_file(filename);

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
std::vector<std::string> HDF5Interface::dataset_list(const std::string &filename,
                                                     const std::string &group_name) 
{
  // List all member datasets of a group by name
  char namebuf[HDF5_MAXSTRLEN];

  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_parallel_file(filename);

  // Open group by name group_name
  hid_t group_id = H5Gopen(file_id,group_name.c_str());
  dolfin_assert(group_id != HDF5_FAIL);

  // Count how many datasets in the group
  hsize_t num_datasets;
  status = H5Gget_num_objs(group_id, &num_datasets);
  dolfin_assert(status != HDF5_FAIL);

  // Iterate through group collecting all dataset names
  std::vector<std::string> list_of_datasets;
  for(hsize_t i=0; i<num_datasets; i++)
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

std::pair<uint, uint> HDF5Interface::dataset_dimensions(
                      const std::string &filename,
                      const std::string &dataset_name) 
{
  // Get dimensions of a 2D dataset

  hsize_t cur_size[2];   // current dataset dimensions
  hsize_t max_size[2];   // maximum dataset dimensions

  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_parallel_file(filename);

  // Open named dataset
  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Get the dataspace of the dataset 
  hid_t space = H5Dget_space(dset_id);
  // Enquire dimensions of the dataspace
  int ndims = H5Sget_simple_extent_dims(space, cur_size, max_size);
  dolfin_assert(ndims == 2); // ensure it is a 2D dataset

  // Close dataset collectively
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return std::pair<uint,uint>(cur_size[0],cur_size[1]);
}

// Add uint or string type attributes to a dataset
// Template below, with type specific code below that
//-----------------------------------------------------------------------------
template void HDF5Interface::add_attribute(const std::string &filename,
                                           const std::string &dataset_name,
                                           const std::string &attribute_name,
                                           const std::string &attribute_value);

//-----------------------------------------------------------------------------
template void HDF5Interface::add_attribute(const std::string &filename,
                                           const std::string &dataset_name,
                                           const std::string &attribute_name,
                                           const uint &attribute_value);

//-----------------------------------------------------------------------------
template <typename T> 
void HDF5Interface::add_attribute(const std::string &filename,
                                  const std::string &dataset_name,
                                  const std::string &attribute_name,
                                  const T &attribute_value)
{
  herr_t status;

  // Open file
  hid_t file_id = open_parallel_file(filename);

  // Open named dataset
  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Create a scalar dataspace
  hid_t dataspace_id = H5Screate (H5S_SCALAR);

  // Add attribute of appropriate type
  _add_attribute_value(dset_id, dataspace_id, attribute_name, attribute_value);
  
  // Close dataset 
  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  // Close file
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}

//-----------------------------------------------------------------------------
void HDF5Interface::_add_attribute_value(const hid_t &dset_id,
                                   const hid_t &dataspace_id,
                                   const std::string &attribute_name, 
                                   const uint &attribute_value)
{
  // Add uint attribute to dataset

  // Copy uint type from H5 types and create attribute
  hid_t datatype_id = H5Tcopy(H5T_NATIVE_UINT);
  hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(), datatype_id,
                                  dataspace_id, H5P_DEFAULT);
  // Write attribute to dataset
  herr_t status = H5Awrite(attribute_id, datatype_id, &attribute_value);
  dolfin_assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attribute_id);
  dolfin_assert(status != HDF5_FAIL);

}
//-----------------------------------------------------------------------------
void HDF5Interface::_add_attribute_value(const hid_t &dset_id,
                                   const hid_t &dataspace_id,
                                   const std::string &attribute_name, 
                                   const std::string &attribute_value)
{
  // Add string attribute to dataset

  // Copy basic string type from HDF5 types
  hid_t datatype_id = H5Tcopy(H5T_C_S1);
  // Set length of string
  herr_t status = H5Tset_size(datatype_id, attribute_value.size());

  // Create attribute in the dataspace with the given string
  hid_t attribute_id = H5Acreate(dset_id, attribute_name.c_str(), datatype_id,
                                  dataspace_id, H5P_DEFAULT);
  // Write attribute to dataset
  status = H5Awrite(attribute_id, datatype_id, attribute_value.c_str());
  dolfin_assert(status != HDF5_FAIL);

  // Close attribute
  status = H5Aclose(attribute_id);
  dolfin_assert(status != HDF5_FAIL);

}

// Get uint or string type attributes of a dataset
// Template below, with type specific code below that
//-----------------------------------------------------------------------------
template void HDF5Interface::get_attribute(const std::string &filename,
                                           const std::string &dataset_name,
                                           const std::string &attribute_name,
                                           uint  &attribute_value);

//-----------------------------------------------------------------------------
template void HDF5Interface::get_attribute(const std::string &filename,
                                           const std::string &dataset_name,
                                           const std::string &attribute_name,
                                           std::string &attribute_value);

//-----------------------------------------------------------------------------
template <typename T> 
void HDF5Interface::get_attribute(const std::string &filename,
                             const std::string &dataset_name,
                             const std::string &attribute_name,
                             T &attribute_value)
{
  herr_t status;

  // Try to open existing HDF5 file
  hid_t file_id = open_parallel_file(filename);

  // Open dataset by name
  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // Open attribute by name and get its type
  hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
  hid_t attr_type = H5Aget_type(attr_id);

  // Specific code for each type of data template
  _get_attribute_value(attr_type, attr_id, attribute_value);
  
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
// Internal code for getting a uint from a dataset attribute
void HDF5Interface::_get_attribute_value(const hid_t &attr_type,
                                         const hid_t &attr_id,
                                         uint &attribute_value)
{
  // FIXME: more complete check of type
  dolfin_assert(H5Tget_class(attr_type)==H5T_INTEGER);
  // Read value
  herr_t status = H5Aread(attr_id, H5T_NATIVE_UINT, &attribute_value);
  dolfin_assert(status != HDF5_FAIL);
  
}

//-----------------------------------------------------------------------------
// Internal code for getting a string from a dataset attribute
void HDF5Interface::_get_attribute_value(const hid_t &attr_type,
                                         const hid_t &attr_id, 
                                         std::string &attribute_value)
{
  // Check this attribute is a string
  dolfin_assert(H5Tget_class(attr_type)==H5T_STRING);

  // Copy string type from HDF5 types and set length accordingly
  hid_t memtype = H5Tcopy(H5T_C_S1);
  int string_length = H5Tget_size(attr_type)+1;
  herr_t status = H5Tset_size(memtype,string_length);
  dolfin_assert(status != HDF5_FAIL);
  // FIXME: messy
  // Copy string value into temporary vector
  // std::vector::data can be copied into
  // (std::string::data cannot)
  std::vector<char> attribute_data(string_length);
  status = H5Aread(attr_id, memtype, attribute_data.data());
  dolfin_assert(status != HDF5_FAIL);
  
  attribute_value.assign(attribute_data.data());
}
