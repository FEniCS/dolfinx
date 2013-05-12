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
// Last changed: 2013-03-04

#ifndef __DOLFIN_XDMFFILE_H
#define __DOLFIN_XDMFFILE_H

#ifdef HAS_HDF5

#include <string>
#include <utility>
#include <boost/scoped_ptr.hpp>

#include "dolfin/common/Variable.h"

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
  template<typename T> class MeshFunction;

  /// This class supports the output of meshes and functions in XDMF
  /// (http://www.xdmf.org) format. It creates an XML file that describes
  /// the data and points to a HDF5 file that stores the actual problem
  /// data. Output of data in parallel is supported.
  ///
  /// XDMF is not suitable for checkpointing as it may decimate
  /// some data.

  class XDMFFile : public GenericFile, public Variable
  {
  public:

    /// Constructor
    explicit XDMFFile(const std::string filename);

    /// Destructor
    ~XDMFFile();

    /// Save a mesh for visualisation, with e.g. ParaView. Creates a HDF5
    /// file to store the mesh, and a related XDMF file with metadata.
    void operator<< (const Mesh& mesh);

    /// Read in a mesh from the associated HDF5 file
    void operator>> (Mesh& mesh);

    /// Save a Function to XDMF/HDF5 files for visualisation.
    void operator<< (const Function& u);

    /// Save Function + time stamp to file
    void operator<< (const std::pair<const Function*, double> ut);

    /// Save MeshFunction to file
    void operator<< (const MeshFunction<bool>& meshfunction);
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<std::size_t>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);

    /// Read first MeshFunction from file
    void operator>> (MeshFunction<bool>& meshfunction);
    void operator>> (MeshFunction<int>& meshfunction);
    void operator>> (MeshFunction<std::size_t>& meshfunction);
    void operator>> (MeshFunction<double>& meshfunction);

  private:

    // HDF5 data file
    boost::scoped_ptr<HDF5File> hdf5_file;

    // HDF5 filename
    std::string hdf5_filename;

    // HDF5 file mode (r/w)
    std::string hdf5_filemode;

    // Generic MeshFunction writer
    template<typename T>
      void write_mesh_function(const MeshFunction<T>& meshfunction);

    // Generic MeshFunction reader
    template<typename T>
      void read_mesh_function(MeshFunction<T>& meshfunction);

    // Write XML description for Function and MeshFunction output
    // updating time-series if need be
    void output_xml(const double time_step, const bool vertex_data,
                    const std::size_t cell_dim,
                    const std::size_t num_global_cells,
                    const std::size_t gdim,
                    const std::size_t num_total_vertices,
                    const std::size_t value_rank,
                    const std::size_t padded_value_size,
                    const std::string name,
                    const std::string dataset_name) const;

    // Helper function to add topology reference to XDMF XML file
    void xml_mesh_topology(pugi::xml_node& xdmf_topology,
                           const std::size_t cell_dim,
                           const std::size_t num_global_cells,
                           const std::string topology_dataset_name) const;

    // Helper function to add geometry section to XDMF XML file
    void xml_mesh_geometry(pugi::xml_node& xdmf_geometry,
                           const std::size_t num_all_local_cells,
                           const std::size_t gdim,
                           const std::string geometry_dataset_name) const;

    // Most recent mesh name
    std::string current_mesh_name;
  };
}
#endif
#endif
