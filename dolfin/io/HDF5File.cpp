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
// Last changed: 2012-07-14

#include <cstdio>
#include <iostream>
#include <fstream>

#define H5_USE_16_API 1
#include <hdf5.h>

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5File.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename)
  : GenericFile(filename, "H5")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void HDF5File::operator<<(const Mesh& mesh)
{
  // Write a Mesh to an existing HDF5 file.
  // TODO: create file if not existing.

  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();

  // get offset and size of local cell topology usage in global terms
  uint off=MPI::global_offset(num_local_cells,true);
  std::pair<uint,uint>topo_range(off,off+num_local_cells);

  // get offset and size of local vertex usage in global terms
  off=MPI::global_offset(num_local_vertices,true);
  std::pair<uint,uint>vertex_range(off,off+num_local_vertices);

  std::vector<uint> topo_data;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topo_data.push_back(v->index()+vertex_range.first);

  std::vector<double>vtx_coords;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    vtx_coords.push_back(p.x());
    vtx_coords.push_back(p.y());
    vtx_coords.push_back(p.z());
  }

  write(vtx_coords[0], vertex_range, "/Mesh/Coordinates", 3); //xyz coords
  write(topo_data[0], topo_range, "/Mesh/Topology", cell_dim + 1); //connectivity
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& output)
{
// Create HDF File and add a dataset
// Allow multiple vectors within one file

  uint dim = 0;
  std::pair<uint, uint> range;
  std::vector<double> data;

  range=output.local_range(dim);
  output.get_local(data);

  if(counter==0)
    create(); //overwrite any existing file

  std::stringstream s("");
  s << "/Vector/" << counter;
  write(data[0], range,s.str().c_str(), 1);

  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{
  // Read input in parallel. Assumes the input vector
  // is correctly allocated to receive the data.

  hid_t file_id;		  // HDF5 file ID
  hid_t plist_id;		  // File access template
  hid_t group_id;
  hid_t filespace;	  // File dataspace ID
  hid_t memspace;	    // memory dataspace ID
  hid_t dset_id;      // Dataset ID
  // hsize_t dimsf;     // dataset dimensions
  hsize_t count[2];   // hyperslab selection parameters
  hsize_t offset[2];
  hsize_t num_obj;    // number of objects in group
  herr_t status;      // Generic return value
  MPICommunicator comm;
  MPIInfo info;

  //    dimsf=input.size(dim);
  const std::pair<uint, uint> range = input.local_range(0);
  offset[0] = range.first;
  offset[1] = 0;
  count[0] = range.second - range.first;
  count[1] = 1;

  /* setup file access template */
  plist_id = H5Pcreate (H5P_FILE_ACCESS);
  dolfin_assert(plist_id != HDF5_FAIL);

  /* set Parallel access with communicator */
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);
  dolfin_assert(status != HDF5_FAIL);

  /* open the file collectively */
  file_id=H5Fopen(filename.c_str(),H5F_ACC_RDWR,plist_id);
  dolfin_assert(file_id != HDF5_FAIL);

  /* Release file-access template */
  status=H5Pclose(plist_id);
  dolfin_assert(status != HDF5_FAIL);

  group_id = H5Gopen(file_id,"/Vector");
  dolfin_assert(group_id != HDF5_FAIL);

  // count how many datasets in the /Vector directory
  status=H5Gget_num_objs(group_id, &num_obj);
  dolfin_assert(status != HDF5_FAIL);

  status=H5Gclose(group_id);
  dolfin_assert(status != HDF5_FAIL);

  std::stringstream s("");  //load last vector
  s << "/Vector/" << (num_obj - 1);

  // open the dataset collectively
  dset_id = H5Dopen(file_id, s.str().c_str());
  dolfin_assert(dset_id != HDF5_FAIL);

  // create a file dataspace independently
  filespace = H5Dget_space (dset_id);
  dolfin_assert(filespace != HDF5_FAIL);

  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                               count, NULL);
  dolfin_assert(status != HDF5_FAIL);

  // create a memory dataspace independently
  memspace = H5Screate_simple (1, count, NULL);
  dolfin_assert (memspace != HDF5_FAIL);

  // read data independently
  std::vector<double> data(count[0]);
  status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
                   H5P_DEFAULT, &data[0]);
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
  // VertexVector - for visualisation
  group_id = H5Gcreate(file_id, "/VertexVector", H5P_DEFAULT);
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
void HDF5File::write(T& data, const std::pair<uint, uint>& range,
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
