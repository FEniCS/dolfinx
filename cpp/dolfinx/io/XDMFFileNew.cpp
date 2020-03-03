// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// #include "xdmf_read.h"
// #include "xdmf_utils.h"
// #include "xdmf_write.h"

#include "HDF5File.h"
// #include "HDF5Utility.h"
#include "XDMFFileNew.h"
// #include "cells.h"
#include "pugixml.hpp"
// #include <algorithm>
// #include <boost/algorithm/string.hpp>
// #include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
// #include <boost/format.hpp>
// #include <boost/lexical_cast.hpp>
// #include <dolfinx/common/MPI.h>
// #include <dolfinx/common/defines.h>
// #include <dolfinx/common/log.h>
// #include <dolfinx/common/utils.h>
// #include <dolfinx/fem/DofMap.h>
// #include <dolfinx/function/Function.h>
// #include <dolfinx/function/FunctionSpace.h>
// #include <dolfinx/graph/AdjacencyList.h>
// #include <dolfinx/la/PETScVector.h>
// #include <dolfinx/la/utils.h>
// #include <dolfinx/mesh/Mesh.h>
// #include <dolfinx/mesh/MeshEntity.h>
// #include <dolfinx/mesh/MeshIterator.h>
// #include <dolfinx/mesh/MeshValueCollection.h>
// #include <dolfinx/mesh/Partitioning.h>
// #include <iomanip>
// #include <memory>
// #include <petscvec.h>
// #include <set>
// #include <string>
// #include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------

/// Construct HDF5 filename from XDMF filename
std::string get_hdf5_filename(std::string filename)
{
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  if (p.string() == filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p.string();
}
//-----------------------------------------------------------------------------

/// TODO
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix)
{
  LOG(INFO) << "Adding mesh to node \"" << xml_node.path('/') << "\"";

  // Add grid node and attributes
  pugi::xml_node grid_node = xml_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = "mesh";
  grid_node.append_attribute("GridType") = "Uniform";

  // // Add topology node and attributes (including writing data)
  // const int tdim = mesh.topology().dim();
  // add_topology_data(comm, grid_node, h5_id, path_prefix, mesh, tdim);

  // // Add geometry node and attributes (including writing data)
  // add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh.geometry());
}
//----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
XDMFFileNew::XDMFFileNew(MPI_Comm comm, const std::string filename,
                         Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _xml_doc(new pugi::xml_document),
      _encoding(encoding)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XDMFFileNew::~XDMFFileNew() { close(); }
//-----------------------------------------------------------------------------
void XDMFFileNew::close()
{
  // Close the HDF5 file
  _hdf5_file.reset();
}
//-----------------------------------------------------------------------------
void XDMFFileNew::write(const mesh::Mesh& mesh)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(mesh.mpi_comm(),
                                         get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Reset pugi doc
  _xml_doc->reset();

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  assert(domain_node);

  // Add the mesh Grid to the domain
  // xdmf_write::add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
