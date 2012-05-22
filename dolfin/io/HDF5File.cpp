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
// First added:  2009-03-03
// Last changed: 2011-09-27

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
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>

#include <hdf5.h>

#include "HDF5File.h"

#define HDF5_FAIL -1

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename)
  : GenericFile(filename, "H5")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HDF5File::HDF5File(std::ostream& s)
  : GenericFile("", "H5"), outstream(&s, NoDeleter())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& output)
{

  hid_t       file_id, dset_id;         /* file and dataset identifiers */
  hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
  hsize_t     dimsf;                 /* dataset dimensions */
  hsize_t     count;	          /* hyperslab selection parameters */
  hsize_t     offset;
  hid_t	      plist_id;           /* property list identifier */
  int         i;
  herr_t      status;

  MPICommunicator comm;
  MPIInfo info;
  uint dim=0;

  std::pair<uint,uint> range;
  std::vector<double>data;

  dimsf=output.size(dim);
  range=output.local_range(dim);
  offset=range.first;
  count=range.second-range.first;

  output.get_local(data);

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id,*comm, *info); 

  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  filespace = H5Screate_simple(1, &dimsf, NULL); 

  dset_id = H5Dcreate(file_id, "dolfin_vector", H5T_NATIVE_DOUBLE, filespace,
		      H5P_DEFAULT);
  H5Sclose(filespace);
  
  memspace = H5Screate_simple(1, &count, NULL);

  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
  status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
		    plist_id, &data[0]);

  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Fclose(file_id);

}


//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{

    hid_t file_id;		/* HDF5 file ID */
    hid_t plist_id;		/* File access template */
    hid_t filespace;	/* File dataspace ID */
    hid_t memspace;	/* memory dataspace ID */
    hid_t dset_id;       	/* Dataset ID */
    hsize_t     dimsf;                 /* dataset dimensions */
    hsize_t     count;	          /* hyperslab selection parameters */
    hsize_t     offset;
    herr_t ret;         	/* Generic return value */

    uint dim=0;

    MPICommunicator comm;
    MPIInfo info;

    std::pair<uint,uint> range;

    dimsf=input.size(dim);
    range=input.local_range(dim);
    offset=range.first;
    count=range.second-range.first;

    std::vector<double>data(count);


    /* setup file access template */
    plist_id = H5Pcreate (H5P_FILE_ACCESS);
    assert(plist_id != HDF5_FAIL);
    /* set Parallel access with communicator */
    ret = H5Pset_fapl_mpio(plist_id, *comm, *info);     
    assert(ret != HDF5_FAIL);

    /* open the file collectively */
    file_id=H5Fopen(filename.c_str(),H5F_ACC_RDWR,plist_id);
    assert(file_id != HDF5_FAIL);

    /* Release file-access template */
    ret=H5Pclose(plist_id);
    assert(ret != HDF5_FAIL);

    /* open the dataset collectively */
    dset_id = H5Dopen(file_id, "dolfin_vector");
    assert(dset_id != HDF5_FAIL);

    /* create a file dataspace independently */
    filespace = H5Dget_space (dset_id);
    assert(filespace != HDF5_FAIL);

    ret=H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    assert(ret != HDF5_FAIL);

    /* create a memory dataspace independently */
    memspace = H5Screate_simple (1, &count, NULL);
    assert (memspace != HDF5_FAIL);

    /* read data independently */
    ret = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
	    H5P_DEFAULT, &data[0]);
    assert(ret != HDF5_FAIL);

    /* close dataset collectively */
    ret=H5Dclose(dset_id);
    assert(ret != HDF5_FAIL);

    /* release all IDs created */
    H5Sclose(filespace);

    /* close the file collectively */
    H5Fclose(file_id);

    input.set_local(data);

}
