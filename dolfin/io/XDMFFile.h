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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef HAS_HDF5
#include <hdf5.h>
#else
typedef int hid_t;
#endif

#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>

namespace boost
{
  namespace filesystem
  {
    class path;
  }
}

namespace pugi
{
  class xml_node;
  class xml_document;
}

namespace dolfin
{

  // Forward declarations
  class Function;
#ifdef HAS_HDF5
  class HDF5File;
#endif
  class LocalMeshData;
  class Mesh;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;
  class Point;
  class XDMFxml;

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
    XDMFFile(const std::string filename)
      : XDMFFile(MPI_COMM_WORLD, filename) {}

    /// Constructor
    XDMFFile(MPI_Comm comm, const std::string filename);

    /// Destructor
    ~XDMFFile();

    /// Save a mesh to XDMF format, either using an associated HDF5
    /// file, or storing the data inline as XML Create function on
    /// given function space
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         A mesh to save.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const Mesh& mesh, Encoding encoding=Encoding::HDF5);

    /// Save a Function to XDMF file for visualisation, using an
    /// associated HDF5 file, or storing the data inline as XML.
    ///
    /// *Arguments*
    ///     u (_Function_)
    ///         A function to save.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const Function& u, Encoding encoding=Encoding::HDF5);

    /// Save a Function with timestamp to XDMF file for visualisation,
    /// using an associated HDF5 file, or storing the data inline as
    /// XML.
    ///
    /// *Arguments*
    ///     u (_Function_)
    ///         A function to save.
    ///     t (_double_)
    ///         Timestep
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const Function& u, double t, Encoding encoding=Encoding::HDF5);

    /// Save MeshFunction to file using an associated HDF5 file, or
    /// storing the data inline as XML.
    ///
    /// *Arguments*
    ///     meshfunction (_MeshFunction_)
    ///         A meshfunction to save.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const MeshFunction<bool>& meshfunction,
               Encoding encoding=Encoding::HDF5);
    void write(const MeshFunction<int>& meshfunction,
               Encoding encoding=Encoding::HDF5);
    void write(const MeshFunction<std::size_t>& meshfunction,
               Encoding encoding=Encoding::HDF5);
    void write(const MeshFunction<double>& meshfunction,
               Encoding encoding=Encoding::HDF5);

    /// Write out mesh value collection (subset) using an associated
    /// HDF5 file, or storing the data inline as XML.
    ///
    /// *Arguments*
    ///     mvc (_MeshValueCollection<std::size_t>_)
    ///         A list of points to save.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const MeshValueCollection<std::size_t>& mvc,
               Encoding encoding=Encoding::HDF5);

    /// Save a cloud of points to file using an associated HDF5 file,
    /// or storing the data inline as XML.
    ///
    /// *Arguments*
    ///     points (_std::vector<Point>_)
    ///         A list of points to save.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const std::vector<Point>& points,
               Encoding encoding=Encoding::HDF5);

    /// Save a cloud of points, with scalar values using an associated
    /// HDF5 file, or storing the data inline as XML.
    ///
    /// *Arguments*
    ///     points (_std::vector<Point>_)
    ///         A list of points to save.
    ///     values (_std::vector<double>_)
    ///         A list of values at each point.
    ///     encoding (_Encoding_)
    ///         Encoding to use: HDF5 or ASCII
    ///
    void write(const std::vector<Point>& points,
               const std::vector<double>& values,
               Encoding encoding=Encoding::HDF5);

    /// Read in a mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///
    void read(Mesh& mesh) const;

    /// Read first MeshFunction from file
    /// @param meshfunction
    void read(MeshFunction<bool>& meshfunction);

    /// Read first MeshFunction from file
    /// @param meshfunction
    void read(MeshFunction<int>& meshfunction);

    /// Read first MeshFunction from file
    /// @param meshfunction
    void read(MeshFunction<std::size_t>& meshfunction);

    /// Read first MeshFunction from file
    /// @param meshfunction
    void read(MeshFunction<double>& meshfunction);

  private:

    // Build mesh (serial)
    static void build_mesh(Mesh& mesh, std::string cell_type_str,
                           std::int64_t num_points, std::int64_t num_cells,
                           int num_points_per_cell,
                           int tdim, int gdim,
                           const pugi::xml_node& topology_dataset_node,
                           const pugi::xml_node& geometry_dataset_node,
                           const boost::filesystem::path& parent_path);

    // Build local mesh data structure
    static void
      build_local_mesh_data (LocalMeshData& local_mesh_data,
                             const CellType& cell_type,
                             const std::int64_t num_points,
                             const std::int64_t num_cells,
                             const int num_points_per_cell,
                             const int tdim, const int gdim,
                             const pugi::xml_node& topology_dataset_node,
                             const pugi::xml_node& geometry_dataset_node,
                             const boost::filesystem::path& parent_path);

    // Add mesh to XDMF xml_node (usually a Domain or Time Grid) and write data
    static void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node,
                         hid_t h5_id, const Mesh& mesh,
                         const std::string path_prefix);

    // Add set of points to XDMF xml_node and write data
    static void add_points(MPI_Comm comm, pugi::xml_node& xml_node,
                           hid_t h5_id, const std::vector<Point>& points);

    // Add topology node to xml_node (includes writing data to XML or  HDF5
    // file)
    template<typename T>
    static void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  hid_t h5_id, const std::string path_prefix,
                                  const Mesh& mesh, int tdim);

    // Add geometry node and data to xml_node
    static void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  hid_t h5_id, const std::string path_prefix,
                                  const Mesh& mesh);

    // Add DataItem node to an XML node. If HDF5 is open (h5_id > 0) the data is
    // written to the HDFF5 file with the path 'h5_path'. Otherwise, data is
    // witten to the XML node and 'h5_path' is ignored
    template<typename T>
    static void add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                              hid_t h5_id, const std::string h5_path, const T& x,
                              const std::vector<std::int64_t> dimensions,
                              const std::string number_type="");

    // Calculate set of entities of dimension cell_dim which should are duplicated
    // on other processes and should not be output on this process
    static std::set<unsigned int> compute_nonlocal_entities(const Mesh& mesh,
                                                            int cell_dim);

    // Return topology data on this process as a flat vector
    template<typename T>
    static std::vector<T> compute_topology_data(const Mesh& mesh, int cell_dim);

    // Return data which is local
    template<typename T>
      std::vector<T> compute_value_data(const MeshFunction<T>& meshfunction);

    // Get DOLFIN cell type string from XML topology node
    static std::string get_cell_type(const pugi::xml_node& topology_node);

    // Get dimensions from an XML DataSet node
    static std::vector<std::int64_t>
      get_dataset_shape(const pugi::xml_node& dataset_node);

    // Get number of cells from an XML Topology node
    static std::int64_t get_num_cells(const pugi::xml_node& topology_node);

    // Return data associated with a data set node
    template <typename T>
    static std::vector<T> get_dataset(MPI_Comm comm,
                                      const pugi::xml_node& dataset_node,
                                      const boost::filesystem::path& parent_path);

    // Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem node
    static std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& dataitem_node);

    static std::string get_hdf5_filename(std::string xdmf_filename);

    // Generic MeshFunction reader
    template<typename T>
    void read_mesh_function(MeshFunction<T>& meshfunction);

    // Generic MeshFunction writer
    template<typename T>
    void write_mesh_function(const MeshFunction<T>& meshfunction, Encoding encoding);

    // Get data width - normally the same as u.value_size(), but expand for 2D
    // vector/tensor because XDMF presents everything as 3D
    static std::int64_t get_padded_width(const Function& u);

    // Returns true for DG0 Functions
    static bool has_cell_centred_data(const Function& u);

    // Get point data values for linear or quadratic mesh into
    // flattened 2D array in data_values with given width
    static std::vector<double> get_point_data_values(const Function& u);

    // Get cell data values as a flattened 2D array in data_values
    static std::vector<double> get_cell_data_values(const Function& u);

    // Check whether the requested encoding is supported
    void check_encoding(Encoding encoding) const;

    // Generate the XDMF format string based on the Encoding
    // enumeration
    static std::string xdmf_format_str(Encoding encoding)
    { return (encoding == XDMFFile::Encoding::HDF5) ? "HDF" : "XML"; }

    // Write MVC to ascii string to store in XDMF XML file
    template <typename T>
    void write_ascii_mesh_value_collection(const MeshValueCollection<T>& mesh_values,
                                             std::string data_name);

    static std::string vtk_cell_type_str(CellType::Type cell_type, int order);

    // Return a string of the form "x y"
    template <typename X, typename Y>
    static std::string to_string(X x, Y y);

    // Return a vector of numerical values from a vector of stringstream
    template <typename T>
    static std::vector<T> string_to_vector(const std::vector<std::string>& x_str);

    // Convert a value_rank to the XDMF string description (Scalar, Vector, Tensor)
    static std::string rank_to_string(std::size_t value_rank);

    // MPI communicator
    MPI_Comm _mpi_comm;

    // HDF5 data file
#ifdef HAS_HDF5
    std::unique_ptr<HDF5File> _hdf5_file;
#endif

    // Cached filename
    const std::string _filename;

    // Counter for time series
    std::size_t _counter;

    // The XML document currently representing the XDMF
    // which needs to be kept open for time series etc.
    std::unique_ptr<pugi::xml_document> _xml_doc;

  };

  // Specialisation for std::vector<bool>, as HDF5 does not support it natively
  template<> inline
  void XDMFFile::add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                               hid_t h5_id, const std::string h5_path,
                               const std::vector<bool>& x,
                               const std::vector<std::int64_t> shape,
                               const std::string number_type)
  {
    // HDF5 cannot accept 'bool' so copy to 'int'
    std::vector<int> x_int(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      x_int[i] = (int)x[i];
    add_data_item(comm, xml_node, h5_id, h5_path, x_int, shape, number_type);
  }

}

#endif
