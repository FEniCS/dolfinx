// Copyright (C) 2012-2015 Chris N. Richardson and Garth N. Wells
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

#ifndef __DOLFIN_XDMFFILE_H
#define __DOLFIN_XDMFFILE_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class Function;
  class HDF5File;
  class Mesh;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;
  class Point;

  /// This class supports the output of meshes and functions in XDMF
  /// (http://www.xdmf.org) format. It creates an XML file that
  /// describes the data and points to a HDF5 file that stores the
  /// actual problem data. Output of data in parallel is supported.
  ///
  /// XDMF is not suitable for checkpointing as it may decimate some
  /// data.

  class XDMFFile : public Variable
  {
  public:

    /// File encoding type
    enum class Encoding {HDF5, ASCII};

    /// Constructor
    XDMFFile(MPI_Comm comm, const std::string filename, const Encoding=Encoding::HDF5);

    /// Destructor
    ~XDMFFile();

    /// Save a mesh for visualisation, with e.g. ParaView. Creates a
    /// HDF5 file to store the mesh, and a related XDMF file with
    /// metadata.
    void operator<< (const Mesh& mesh);

    /// Save a mesh for visualisation, with e.g. ParaView. Creates a
    /// HDF5 file to store the mesh, and a related XDMF file with
    /// metadata.
    void write (const Mesh& mesh);

    /// Read in a mesh from the associated HDF5 file, optionally using
    /// stored partitioning, if possible when the same number of
    /// processes are being used.
    void read(Mesh& mesh, bool use_partition_from_file);

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


    /// Save a cloud of points to file
    void write(const std::vector<Point>& points);

    /// Save a cloud of points, with scalar values
    void write(const std::vector<Point>& points,
               const std::vector<double>& values);

    /// Read first MeshFunction from file
    void operator>> (MeshFunction<bool>& meshfunction);
    void operator>> (MeshFunction<int>& meshfunction);
    void operator>> (MeshFunction<std::size_t>& meshfunction);
    void operator>> (MeshFunction<double>& meshfunction);

    // Write out mesh value collection (subset)
    void write(const MeshValueCollection<std::size_t>& mvc);

  private:

    // MPI communicator
    MPI_Comm _mpi_comm;

    // HDF5 data file
    std::unique_ptr<HDF5File> hdf5_file;

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

    // Write XML description of point clouds, with value_size = 0, 1
    // or 3 (for either no point data, scalar, or vector)
    void write_point_xml(const std::string dataset_name,
                         const std::size_t num_global_points,
                         const unsigned int value_size);

    // Get point data values for linear or quadratic mesh into
    // flattened 2D array in data_values with given width
    void get_point_data_values(std::vector<double>& data_values,
                               std::size_t width, const Function& u);

    std::string xdmf_format_str()
    { return (_encoding == XDMFFile::Encoding::HDF5) ? "HDF" : "XML"; }

    // Most recent mesh name
    std::string current_mesh_name;

    // Cached filename
    const std::string _filename;

    // Counter for time series
    std::size_t counter;

    // Encoding for storing the actual data, either inline as ASCII in the XML
    // or as an external HDF5 file
    const Encoding _encoding;

  };
}

#endif
