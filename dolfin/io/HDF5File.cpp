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
//
// First added:  2012-06-01
// Last changed: 2012-06-29

#include <cstdio>
#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
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

#define H5_USE_16_API 1
#include "HDF5File.h"

#include <hdf5.h>

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

//void HDF5File::operator<<(const Function& u){
  //use xdmf format instead
  //could migrate some code from XDMFFile.cpp to here

  //create HDF5 file
  //add coordinates and topology (from mesh)
  //add vector
  //add dofmap?
  
//}

// write a generic block of 2D data into a HDF5 dataset
// in parallel. Pre-existing file.
template <typename T>
void HDF5File::write(T& data, 
		     const std::pair<uint,uint>& range,
		     const std::string& dataset_name,
		     const int h5type,
		     const uint width){


  hid_t       file_id, dset_id;         /* file and dataset identifiers */
  hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
  hsize_t     dimsf[2];                 /* dataset dimensions */
  hsize_t     count[2];	          /* hyperslab selection parameters */
  hsize_t     offset[2];
  hid_t	      plist_id;           /* property list identifier */
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;

  offset[0]=range.first;
  offset[1]=0;
  count[0]=range.second-range.first;
  count[1]=width;
  dimsf[0]=MPI::sum((int)count[0]);
  dimsf[1]=width;

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status=H5Pset_fapl_mpio(plist_id,*comm, *info); 
  assert(status != HDF5_FAIL);

  // try to open existing HDF5 file
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
  assert(file_id != HDF5_FAIL);

  status=H5Pclose(plist_id);
  assert(status != HDF5_FAIL);

  // define a 2D dataset
  filespace = H5Screate_simple(2, dimsf, NULL); 
  assert(filespace != HDF5_FAIL);

  dset_id = H5Dcreate(file_id, dataset_name.c_str(), h5type, filespace,
		      H5P_DEFAULT);
  assert(dset_id != HDF5_FAIL);
  status=H5Sclose(filespace);
  assert(status != HDF5_FAIL);

  memspace = H5Screate_simple(2, count, NULL);

  filespace = H5Dget_space(dset_id);
  status=H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
  assert(status != HDF5_FAIL);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  status=H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  assert(status != HDF5_FAIL);

  status = H5Dwrite(dset_id, h5type, memspace, filespace,
		    plist_id, &data);
  assert(status != HDF5_FAIL);

  status=H5Dclose(dset_id);
  assert(status != HDF5_FAIL);
  status=H5Sclose(filespace);
  assert(status != HDF5_FAIL);
  status=H5Sclose(memspace);
  assert(status != HDF5_FAIL);
  status=H5Pclose(plist_id);
  assert(status != HDF5_FAIL);
  status=H5Fclose(file_id);
  assert(status != HDF5_FAIL);

}

// Write data to existing HDF file as defined by range blocks on each process
// range: the local range on this processor
// width: is the width of the dataitem (e.g. 3 for x,y,z data)

void HDF5File::write(const double& data,
		     const std::pair<uint,uint>& range,
		     const std::string& dataset_name,
		     const uint width){
  write(data,range,dataset_name,H5T_NATIVE_DOUBLE,width);
}

void HDF5File::write(const uint& data,
		     const std::pair<uint,uint>& range,
		     const std::string& dataset_name,
		     const uint width){
  write(data,range,dataset_name,H5T_NATIVE_INT,width);
}

void HDF5File::create(){  //maybe this should be in the constructor
  // make empty HDF5 file
  // overwriting any existing file

  hid_t       file_id;         /* file and dataset identifiers */
  hid_t	      plist_id;           /* property list identifier */
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  status = H5Pset_fapl_mpio(plist_id, *comm, *info);     
  assert(status != HDF5_FAIL);
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  assert(file_id != HDF5_FAIL);
  status=H5Pclose(plist_id);
  assert(status != HDF5_FAIL);
  status=H5Fclose(file_id);
  assert(status != HDF5_FAIL);
}

// Create HDF File and add a dataset of dolfin_vector
void HDF5File::operator<< (const GenericVector& output)
{
  uint dim=0;

  std::pair<uint,uint> range;
  std::vector<double>data;

  range=output.local_range(dim);
  output.get_local(data);

  create(); //overwrite any existing file
  write(data[0],range,"dolfin_vector",1);

}


//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{

    hid_t file_id;		/* HDF5 file ID */
    hid_t plist_id;		/* File access template */
    hid_t filespace;	/* File dataspace ID */
    hid_t memspace;	/* memory dataspace ID */
    hid_t dset_id;       	/* Dataset ID */
    // hsize_t     dimsf;                 /* dataset dimensions */
    hsize_t     count;	          /* hyperslab selection parameters */
    hsize_t     offset;
    herr_t status;         	/* Generic return value */

    uint dim=0;

    MPICommunicator comm;
    MPIInfo info;

    std::pair<uint,uint> range;

    //    dimsf=input.size(dim);
    range=input.local_range(dim);
    offset=range.first;
    count=range.second-range.first;

    std::vector<double>data(count);

    /* setup file access template */
    plist_id = H5Pcreate (H5P_FILE_ACCESS);
    assert(plist_id != HDF5_FAIL);
    /* set Parallel access with communicator */
    status = H5Pset_fapl_mpio(plist_id, *comm, *info);     
    assert(status != HDF5_FAIL);

    /* open the file collectively */
    file_id=H5Fopen(filename.c_str(),H5F_ACC_RDWR,plist_id);
    assert(file_id != HDF5_FAIL);

    /* Release file-access template */
    status=H5Pclose(plist_id);
    assert(status != HDF5_FAIL);

    /* open the dataset collectively */
    dset_id = H5Dopen(file_id, "dolfin_vector");
    assert(dset_id != HDF5_FAIL);

    /* create a file dataspace independently */
    filespace = H5Dget_space (dset_id);
    assert(filespace != HDF5_FAIL);

    status=H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    assert(status != HDF5_FAIL);

    /* create a memory dataspace independently */
    memspace = H5Screate_simple (1, &count, NULL);
    assert (memspace != HDF5_FAIL);

    /* read data independently */
    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
	    H5P_DEFAULT, &data[0]);
    assert(status != HDF5_FAIL);

    /* close dataset collectively */
    status=H5Dclose(dset_id);
    assert(status != HDF5_FAIL);

    /* release all IDs created */
    status=H5Sclose(filespace);
    assert(status != HDF5_FAIL);

    /* close the file collectively */
    status=H5Fclose(file_id);
    assert(status != HDF5_FAIL);

    input.set_local(data);

}
