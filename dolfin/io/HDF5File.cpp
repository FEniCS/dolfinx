// Copyright (C) 2012 Chris N Richardson
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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-06-01
// Last changed: 2012-09-14

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>

// Use version 1.6 API for stability
// Fairly easy to switch to later version
// But requires adding extra fields to several calls
//
#define H5_USE_16_API
#include <hdf5.h>

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5File.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename) : GenericFile(filename, "H5")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void HDF5File::operator>>(Mesh& input_mesh)
{
  // FIXME: Figure out how to handle multiple meshes in file

  // List all datasets in the /Mesh folder - should be two, one starting
  // with "Topology" and one starting with "Coordinates"
  std::vector<std::string> listing;
  listing = list("/Mesh");

  // TODO - should do a more comprehensive check
  if(listing.size() != 2)
  {
    dolfin_error("HDF5File.cpp",
                 "read mesh from file",
                 "Invalid number of Mesh datasets in HDF5 file");
  }

  LocalMeshData mesh_data;

  // Clear mesh data
  mesh_data.clear();

  // Coordinates

  std::string coords_name("/Mesh/");
  coords_name.append(listing[0]); // hopefully 'Coordinates' - need to make more robust
  std::pair<uint,uint> coords_dim = dataset_dimensions(coords_name);
  std::cout << "Coordinate dataset:" << coords_name << " : ";
  std::cout << coords_dim.first << " " << coords_dim.second << std::endl;

  const uint num_global_vertices = coords_dim.first;
  mesh_data.num_global_vertices = num_global_vertices;

  std::pair<uint,uint> vertex_range = MPI::local_range(num_global_vertices);
  uint nv = vertex_range.second-vertex_range.first;
  mesh_data.vertex_indices.reserve(nv);
  std::cout << "Reserved space for " << nv << " vertices" << std::endl;
  std::vector<double> data;
  data.reserve(nv*3); // Mesh always saved in 3D regardless, so may need to decimate
  read(data[0], vertex_range, coords_name, H5T_NATIVE_DOUBLE, 3);

  for(uint i = 0; i < nv*3; i++)
    std::cout << data[i] << std::endl;

  //  for(){
  //    mesh_data.vertex_coordinates.push_back(v);
  //    mesh_data.vertex_indices.push_back(i);
  //  }

  // Topology

  mesh_data.gdim = 2; //need to fix this
  mesh_data.tdim = 2;

  std::string topo_name("/Mesh/");
  topo_name.append(listing[1]);
  std::pair<uint,uint> topo_dim = dataset_dimensions(topo_name);
  std::cout << "Topological dataset:" << topo_name << " : ";
  std::cout << topo_dim.first << " " << topo_dim.second << std::endl;

  const uint num_global_cells = topo_dim.first;
  mesh_data.num_global_cells = num_global_cells;
  std::pair<uint,uint> cell_range = MPI::local_range(num_global_cells);
  uint num_cells = cell_range.second-cell_range.first;
  mesh_data.global_cell_indices.reserve(num_cells);
  mesh_data.cell_vertices.reserve(num_cells);
  uint num_vertices_per_cell = topo_dim.second;
  mesh_data.num_vertices_per_cell = num_vertices_per_cell;

  std::vector<uint> topo_data(num_cells*num_vertices_per_cell);
  topo_data.reserve(num_cells*num_vertices_per_cell);
  read(topo_data[0], cell_range, topo_name, H5T_NATIVE_INT, num_vertices_per_cell);

  // copy data to mesh...
  //  for(){
  //    mesh_data.cell_vertices.push_back(cell);
  //    mesh_data.global_cell_indices.push_back(i);
  //  }

  MeshPartitioning::build_distributed_mesh(input_mesh, mesh_data);
}
//-----------------------------------------------------------------------------
void HDF5File::operator<<(const Mesh& mesh)
{
  // if no existing file, create...
  // FIXME: better way to check? MPI safe?
  if(boost::filesystem::file_size(filename.c_str()) == 0)
    create();

  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();

  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);

  // Get offset and size of local cell topology usage in global terms
  const uint cell_offset = MPI::global_offset(num_local_cells, true);
  std::pair<uint, uint>topo_range(cell_offset, cell_offset + num_local_cells);

  // get offset and size of local vertex usage in global terms
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);
  std::pair<uint, uint>vertex_range(vertex_offset, vertex_offset + num_local_vertices);

  std::vector<uint> topological_data;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    for (VertexIterator v(*cell); !v.end(); ++v)
      topological_data.push_back(v->index() + vertex_range.first);

  std::vector<double> vertex_coords;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    vertex_coords.push_back(p.x());
    vertex_coords.push_back(p.y());
    vertex_coords.push_back(p.z());
  }

  std::stringstream s;
  s << mesh_coords_dataset_name(mesh);
  if (!exists(s.str()))
    write(vtx_coords[0], vertex_range, s.str(), 3); //xyz coords

  s.str("");
  s << mesh_topo_dataset_name(mesh);
  if (!exists(s.str()))
  {
    write(topological_data[0], topo_range, s.str(), cell_dim + 1); //connectivity
    add_attribute(s.str(),"celltype", cell_type);
  }

}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& output)
{
  // Create HDF File and add a dataset
  // Allow multiple vectors within one file

  uint dim = 0;
  std::pair<uint, uint> range;
  std::vector<double> data;

  range = output.local_range(dim);
  output.get_local(data);

  if (counter == 0)
    create(); //overwrite any existing file

  std::stringstream s("");
  s << "/Vector/" << counter;
  write(data[0], range,s.str().c_str(), 1);

  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{
  const std::pair<uint, uint> range = input.local_range(0);
  std::vector<double> data(range.second-range.first);
  read(data[0], range, "/Vector/0", H5T_NATIVE_DOUBLE, 1);
  input.set_local(data);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const double& data, const std::pair<uint,uint>& range,
                     const std::string& dataset_name, const uint width)
{
  // Write data to existing HDF file as defined by range blocks on each process
  // range: the local range on this processor
  // width: is the width of the dataitem (e.g. 3 for x,y,z data)
  write(data,range,dataset_name,H5T_NATIVE_DOUBLE,width);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const uint& data, const std::pair<uint,uint>& range,
                     const std::string& dataset_name, const uint width)
{
  // Write data to existing HDF file as defined by range blocks on each process
  // range: the local range on this processor
  // width: is the width of the dataitem (e.g. 3 for x,y,z data)
  write(data,range,dataset_name,H5T_NATIVE_INT,width);
}
//-----------------------------------------------------------------------------
void HDF5File::create()
{
  // make empty HDF5 file
  // overwriting any existing file
  // create some default 'folders' for storing different datasets

  hid_t       file_id;         /* file and dataset identifiers */
  hid_t	      plist_id;           /* property list identifier */
  hid_t       group_id;
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);
  dolfin_assert(status != HDF5_FAIL);
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // create subgroups suitable for storing different types of data.
  // DataVector - values for visualisation
  group_id = H5Gcreate(file_id, "/DataVector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  // Vector - for checkpointing etc
  group_id = H5Gcreate(file_id, "/Vector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  assert(status != HDF5_FAIL);

  // Mesh
  group_id = H5Gcreate(file_id, "/Mesh", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}

//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::read(T& data,  const std::pair<uint, uint>& range,
                     const std::string& dataset_name,
                     const int h5type, const uint width) const
{
  // read a generic block of 2D data from a HDF5 dataset

  // Read input in parallel. Assumes the input vector
  // is correctly allocated to receive the data.

  hid_t file_id;		  // HDF5 file ID
  hid_t plist_id;		  // File access template
  hid_t filespace;	  // File dataspace ID
  hid_t memspace;	    // memory dataspace ID
  hid_t dset_id;      // Dataset ID
  hsize_t count[2];   // hyperslab selection parameters
  hsize_t offset[2];
  herr_t status;      // Generic return value
  MPICommunicator comm;
  MPIInfo info;


  offset[0] = range.first;
  offset[1] = 0;
  count[0] = range.second - range.first;
  count[1] = width;

  std::cout << dataset_name << std::endl;
  std::cout << offset[0] << " " << offset[1] << std::endl;
  std::cout << count[0] << " " << count[1] << std::endl;

  /* setup file access template */
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  dolfin_assert(plist_id != HDF5_FAIL);

  /* set Parallel access with communicator */
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Open the file collectively
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR,plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  /* Release file-access template */
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  // open the dataset collectively
  dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // create a file dataspace independently
  filespace = H5Dget_space (dset_id);
  dolfin_assert(filespace != HDF5_FAIL);

  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                               count, NULL);
  dolfin_assert(status != HDF5_FAIL);

  // create a memory dataspace independently
  memspace = H5Screate_simple (2, count, NULL);
  dolfin_assert (memspace != HDF5_FAIL);

  // read data independently

  status = H5Dread(dset_id, h5type, memspace, filespace,
                   H5P_DEFAULT, &data);
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

  input.set_local(data);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const double& data, const std::pair<uint, uint>& range,
                     const std::string& dataset_name, const uint width)
{
  // Write data to existing HDF file as defined by range blocks on each process
  // range: the local range on this processor
  // width: is the width of the dataitem (e.g. 3 for x, y, z data)
  write(data, range, dataset_name, H5T_NATIVE_DOUBLE, width);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const uint& data, const std::pair<uint, uint>& range,
                     const std::string& dataset_name, const uint width)
{
  // Write data to existing HDF file as defined by range blocks on each process
  // range: the local range on this processor
  // width: is the width of the dataitem (e.g. 3 for x, y, z data)
  write(data, range, dataset_name, H5T_NATIVE_INT, width);
}
//-----------------------------------------------------------------------------
void HDF5File::create()
{
  // make empty HDF5 file
  // overwriting any existing file
  // create some default 'folders' for storing different datasets

  hid_t       file_id;         /* file and dataset identifiers */
  hid_t	      plist_id;           /* property list identifier */
  hid_t       group_id;
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);
  dolfin_assert(status != HDF5_FAIL);
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  // create subgroups suitable for storing different types of data.
  // DataVector - values for visualisation
  group_id = H5Gcreate(file_id, "/DataVector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  // Vector - for checkpointing etc
  group_id = H5Gcreate(file_id, "/Vector", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  assert(status != HDF5_FAIL);

  // Mesh
  group_id = H5Gcreate(file_id, "/Mesh", H5P_DEFAULT);
  dolfin_assert(group_id != HDF5_FAIL);
  status = H5Gclose (group_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5File::write(const T& data, const std::pair<uint, uint>& range,
                     const std::string& dataset_name,
                     const int h5type, const uint width) const
{
  // write a generic block of 2D data into a HDF5 dataset
  // in parallel. Pre-existing file.

  hid_t       file_id, dset_id;         /* file and dataset identifiers */
  hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
  hsize_t     dimsf[2];                 /* dataset dimensions */
  hsize_t     count[2];	          /* hyperslab selection parameters */
  hsize_t     offset[2];
  hid_t	      plist_id;           /* property list identifier */
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;

  offset[0] = range.first;
  offset[1] = 0;
  count[0] = range.second-range.first;
  count[1] = width;
  dimsf[0] = MPI::sum((int)count[0]);
  dimsf[1] = width;

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  // define a 2D dataset
  filespace = H5Screate_simple(2, dimsf, NULL);
  assert(filespace != HDF5_FAIL);

  dset_id = H5Dcreate(file_id, dataset_name.c_str(), h5type, filespace,
                      H5P_DEFAULT);
  dolfin_assert(dset_id != HDF5_FAIL);

  status = H5Sclose(filespace);
  dolfin_assert(status != HDF5_FAIL);

  memspace = H5Screate_simple(2, count, NULL);

  filespace = H5Dget_space(dset_id);
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
  dolfin_assert(status != HDF5_FAIL);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  assert(status != HDF5_FAIL);

  status = H5Dwrite(dset_id, h5type, memspace, filespace, plist_id, &data);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Sclose(filespace);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Sclose(memspace);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);
  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);
}
//-----------------------------------------------------------------------------
// check existence of a dataset in this file
bool HDF5File::exists(const std::string& dataset_name){

  MPICommunicator comm;
  MPIInfo info;
  herr_t status;

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  // disable error reporting
  herr_t (*old_func)(void*);
  void *old_client_data;
  H5Eget_auto(&old_func, &old_client_data);

  status = H5Eset_auto(NULL, NULL);
  dolfin_assert(status != HDF5_FAIL);

  //try to open dataset - returns HDF5_FAIL if non-existent
  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());

  if(dset_id != HDF5_FAIL)
    H5Dclose(dset_id);

  //re-enable error reporting
  status = H5Eset_auto(old_func, old_client_data);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return (dset_id!=HDF5_FAIL);
}
//-----------------------------------------------------------------------------

// make a listing of all dataset names in a HDF5 group
std::vector<std::string> HDF5File::list(const std::string& group_name)
{
  char namebuf[HDF5_MAXSTRLEN];

  MPICommunicator comm;
  MPIInfo info;
  herr_t status;

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  hid_t group_id = H5Gopen(file_id,group_name.c_str());
  dolfin_assert(group_id != HDF5_FAIL);

  // count how many datasets in the group
  hsize_t num_obj;
  status = H5Gget_num_objs(group_id, &num_obj);
  dolfin_assert(status != HDF5_FAIL);

  std::vector<std::string> lvec;
  std::string str;
  // go through all objects
  for(hsize_t i=0; i<num_obj; i++)
  {
    H5Gget_objname_by_idx(group_id, i, namebuf, HDF5_MAXSTRLEN);
    str=namebuf;
    lvec.push_back(str);
  }

  status = H5Gclose(group_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return lvec;
}
//-----------------------------------------------------------------------------
// get dimensions of a 2D dataset
std::pair<uint,uint> HDF5File::dataset_dimensions(const std::string& dataset_name)
{
  hsize_t     cur_size[2];   /* current dataset dimensions */
  hsize_t     max_size[2];   /* maximum dataset dimensions */
  hid_t       space;         /* data space                 */
  int         ndims;          /* dimensionality             */

  MPICommunicator comm;
  MPIInfo info;
  herr_t status;

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  space = H5Dget_space(dset_id);
  ndims = H5Sget_simple_extent_dims(space, cur_size, max_size);
  dolfin_assert(ndims==2);

  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return std::pair<uint,uint>(cur_size[0],cur_size[1]);
}
//-----------------------------------------------------------------------------
void HDF5File::add_attribute(const std::string& dataset_name,
			     const std::string& attribute_name,
			     const std::string& attribute_value)
{

  MPICommunicator comm;
  MPIInfo info;
  herr_t status;

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // add string attribute
  hid_t datatype_id = H5Tcopy (H5T_C_S1);
  status = H5Tset_size (datatype_id, attribute_value.size());
  hid_t dataspaces_id = H5Screate (H5S_SCALAR);
  hid_t attribute_id = H5Acreate (dset_id, attribute_name.c_str(), datatype_id,
				  dataspaces_id, H5P_DEFAULT);
  status = H5Awrite(attribute_id, datatype_id, attribute_value.c_str());
  dolfin_assert(status != HDF5_FAIL);

  status = H5Aclose(attribute_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

}
//-----------------------------------------------------------------------------
std::string HDF5File::get_attribute(const std::string& dataset_name,
				    const std::string& attribute_name)
{
  MPICommunicator comm;
  MPIInfo info;
  herr_t status;

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id,*comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  // Try to open existing HDF5 file
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  status = H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  hid_t dset_id = H5Dopen(file_id, dataset_name.c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  hid_t attr_id = H5Aopen(dset_id, attribute_name.c_str(), H5P_DEFAULT);
  hid_t filetype = H5Aget_type(attr_id);
  int slen = H5Tget_size(filetype);
  slen++;

  //  hid_t space_id = H5Aget_space(attr_id);
  hid_t memtype = H5Tcopy (H5T_C_S1);

  status = H5Tset_size(memtype,slen);
  dolfin_assert(status != HDF5_FAIL);

  std::vector<char> str(slen);
  status = H5Aread(attr_id, memtype, &str[0]);

  status = H5Aclose(attr_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Dclose(dset_id);
  dolfin_assert(status != HDF5_FAIL);

  status = H5Fclose(file_id);
  dolfin_assert(status != HDF5_FAIL);

  return std::string(&str[0]);
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_coords_dataset_name(const Mesh& mesh)
{
  std::stringstream mc_name;
  mc_name << "/Mesh/Coordinates_" << std::setfill('0')
	  << std::hex << std::setw(8) << mesh.coordinates_hash();
  return mc_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_topo_dataset_name(const Mesh& mesh)
{
  std::stringstream mc_name;
  mc_name << "/Mesh/Topology_" << std::setfill('0')
	  << std::hex << std::setw(8) << mesh.topology_hash();
  return mc_name.str();
}
//-----------------------------------------------------------------------------

#endif
