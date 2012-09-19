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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-05-22
// Last changed: 2012-09-19

#ifndef __DOLFIN_XDMFFILE_H
#define __DOLFIN_XDMFFILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <boost/scoped_ptr.hpp>
#include "GenericFile.h"

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  // Forward declarations
  class Function;
  class HDF5File;
  class Mesh;

  /// This class supports the output of meshes and functions in XDMF
  /// (http://www.xdmf.org) format. It creates an XML file that describes
  /// the data and points to a HDF5 file that stores the actual problem
  /// data. Output of data in parallel is supported.
  ///
  /// XDMF is not suitable for checkpointing as it may decimate
  /// some data.

  class XDMFFile: public GenericFile
  {
  public:

    /// Constructor
    XDMFFile(const std::string filename);

    /// Destructor
    ~XDMFFile();

    /// Save a mesh for visualisation, with e.g. ParaView. Creates a HDF5
    /// file to store the mesh, and a related XDMF file with metadata.
    void operator<<(const Mesh& mesh);

    /// Save a Function to XDMF/HDF files for visualisation.
    /// Downgrading may occur due to collecting the values
    /// to vertices using compute_vertex_values()
    ///
    /// Creates an HDF5 file for storing Mesh and Vertex/Cell Values,
    /// and an associated XDMF file for metadata.
    /// Subsequent calls will store additional Vertex/Cell Values
    /// in the same HDF5 file, and update the XDMF metadata
    /// to represent a time series.
    void operator<<(const Function& u);

    /// Save Function + time stamp to file
    void operator<<(const std::pair<const Function*, double> ut);

    /// Save MeshFunction to file
    /// Data is stored in an HDF5 file along with the Mesh
    /// An XML description is stored in an associated XDMF file
    void operator<<(const MeshFunction<uint>& meshfunction);
    void operator<<(const MeshFunction<double>& meshfunction);

  private:

    // HDF5 data file
    boost::scoped_ptr<HDF5File> hdf5_file;
    
    template<typename T>
      void write_mesh_function(const MeshFunction<T>& meshfunction);
    
  };

}
#endif
#endif
