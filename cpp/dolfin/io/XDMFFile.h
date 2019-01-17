// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <hdf5.h>
#include <memory>
#include <petscsys.h>
#include <string>
#include <utility>
#include <vector>

namespace boost
{
namespace filesystem
{
class path;
}
} // namespace boost

namespace pugi
{
class xml_node;
class xml_document;
} // namespace pugi

namespace dolfin
{
namespace function
{
class Function;
class FunctionSpace;
} // namespace function

namespace geometry
{
class Point;
}

namespace mesh
{
enum class GhostMode : int;
class Mesh;
template <typename T>
class MeshFunction;
template <typename T>
class MeshValueCollection;
class MeshPartitioning;
} // namespace mesh

namespace io
{
class HDF5File;

/// Read and write mesh::Mesh, function::Function, mesh::MeshFunction and other
/// objects in XDMF

/// This class supports the output of meshes and functions in XDMF
/// (http://www.xdmf.org) format. It creates an XML file that
/// describes the data and points to a HDF5 file that stores the
/// actual problem data. Output of data in parallel is supported.
///
/// XDMF is not suitable for checkpointing as it may decimate some
/// data.

// FIXME: Set read mode when creating file obejct?

// FIXME: Set encoding when opening file

// FIXME: Remove the duplicate read_mf_foo functions. Challenge is the
// templated reader code would then expose a lot code publically.
// Refactor large, templated functions into parts that (i) depend on the
// template argument and (ii) parts that do not. Same applies to
// MeshValueCollection.

class XDMFFile : public common::Variable
{
public:
  /// File encoding type
  enum class Encoding
  {
    HDF5,
    ASCII
  };

  /// Default encoding type
  static const Encoding default_encoding = Encoding::HDF5;

  /// Constructor
  XDMFFile(MPI_Comm comm, const std::string filename,
           Encoding encoding = default_encoding);

  /// Destructor
  ~XDMFFile();

  /// Close the file
  ///
  /// This closes any open HDF5 files. In ASCII mode the XML file is
  /// closed each time it is written to or read from, so close() has
  /// no effect.
  ///
  /// From Python you can also use XDMFFile as a context manager:
  ///
  ///     with XDMFFile(mpi_comm_world(), 'name.xdmf') as xdmf:
  ///         xdmf.write(mesh)
  ///
  /// The file is automatically closed at the end of the with block
  void close();

  /// Save a mesh to XDMF format, either using an associated HDF5
  /// file, or storing the data inline as XML Create function on
  /// given function space
  ///
  /// @param    mesh (_Mesh_)
  ///         A mesh to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::Mesh& mesh);

  /// Save a function::Function to XDMF file for checkpointing, using an
  /// associated HDF5 file, or storing the data inline as XML.
  ///
  /// If the file where we would like to write exists, then
  /// the function is appended to the file.
  ///
  /// @param    u (_Function_)
  ///         A function to save.
  /// @param    function_name (string)
  ///         Name of the function used in XDMF XML and HDF file
  ///         (if encoding = HDF). The string is used to fill value for XML
  ///         attribute Name of Grid node. It is also used in HDF file in
  ///         path to datasets. Must be the same on all processes in parallel.
  /// @param    time_step (_double_)
  ///         Time step. It is saved only in XDMF file. The function could
  ///         be saved with the same time step several times. There is an
  ///         internal "counter" value stored in XDMF which differentiates
  ///         the same time steps.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write_checkpoint(const function::Function& u, std::string function_name,
                        double time_step = 0.0);

  /// Save a function::Function to XDMF file for visualisation, using an
  /// associated HDF5 file, or storing the data inline as XML.
  ///
  /// @param    u (_Function_)
  ///         A function to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const function::Function& u);

  /// Save a function::Function with timestamp to XDMF file for visualisation,
  /// using an associated HDF5 file, or storing the data inline as
  /// XML.
  ///
  /// You can control the output with the following boolean
  /// parameters on the XDMFFile class:
  ///
  /// * rewrite_function_mesh (default true):
  ///   Controls whether the mesh will be rewritten every timestep.
  ///   If the mesh does not change this can be turned off to create
  ///   smaller files.
  ///
  /// * functions_share_mesh (default false):
  ///   Controls whether all functions on a single time step share
  ///   the same mesh. If true the files created will be smaller and
  ///   also behave better in Paraview, at least in version 5.3.0
  ///
  /// @param    u (_Function_)
  ///         A function to save.
  /// @param    t (_double_)
  ///         Timestep
  /// @param   encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const function::Function& u, double t);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param    meshfunction (_MeshFunction_)
  ///         A meshfunction to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshFunction<bool>& meshfunction);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param    meshfunction (_MeshFunction_)
  ///         A meshfunction to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshFunction<int>& meshfunction);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param    meshfunction (_MeshFunction_)
  ///         A meshfunction to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshFunction<std::size_t>& meshfunction);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param    meshfunction (_MeshFunction_)
  ///         A meshfunction to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshFunction<double>& meshfunction);

  /// Write out mesh value collection (subset) using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param mvc (_MeshValueCollection<bool>_)
  ///         mesh::MeshValueCollection to save
  /// @param encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshValueCollection<bool>& mvc);

  /// Write out mesh value collection (subset) using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param mvc (_MeshValueCollection<int>_)
  ///         mesh::MeshValueCollection to save
  /// @param encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshValueCollection<int>& mvc);

  /// Write out mesh value collection (subset) using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param  mvc (_MeshValueCollection<int>_)
  ///         mesh::MeshValueCollection to save
  /// @param  encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshValueCollection<std::size_t>& mvc);

  /// Write out mesh value collection (subset) using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param mvc (_MeshValueCollection<double>_)
  ///         mesh::MeshValueCollection to save
  /// @param encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const mesh::MeshValueCollection<double>& mvc);

  /// Save a cloud of points to file using an associated HDF5 file,
  /// or storing the data inline as XML.
  ///
  /// @param    points (_std::vector<geometry::Point>_)
  ///         A list of points to save.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const std::vector<geometry::Point>& points);

  /// Save a cloud of points, with scalar values using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param   points (_std::vector<geometry::Point>_)
  ///         A list of points to save.
  /// @param    values (_std::vector<double>_)
  ///         A list of values at each point.
  /// @param    encoding (_Encoding_)
  ///         Encoding to use: HDF5 or ASCII
  ///
  void write(const std::vector<geometry::Point>& points,
             const std::vector<double>& values);

  /// Read in the first mesh::Mesh in XDMF file
  ///
  /// @param comm (MPI_Comm)
  ///        MPI Communicator
  /// @param ghost_mode (GhostMode)
  ///        Ghost mode for mesh partition
  /// @returns mesh::Mesh
  ///        Mesh
  mesh::Mesh read_mesh(MPI_Comm comm, const mesh::GhostMode ghost_mode) const;

  /// Read a function from the XDMF file. Supplied function must
  /// come with already initialized and compatible function space.
  ///
  /// function::Functions saved as time-series must be addressed through
  /// (integer) internal counter and function name. function::Function name
  /// is a name of function that was saved. Counter stands for
  /// the number of function from time-series, e.g. counter=0
  /// refers to first saved function regardless of its time-step value.
  ///
  /// @param    V (std::shared_ptr<function::FunctionSpace>)
  ///         FunctionSpace
  /// @param    func_name (_string_)
  ///         A name of a function to read. Must be the same on all processes
  ///         in parallel.
  /// @param    counter (_int64_t_)
  ///         Internal integer counter - used in time-series. Default value
  ///         is -1 which points to last saved function. Counter works same as
  ///         python array position key, i.e. counter = -2 points to the
  ///         function before the last one.
  /// @returns function::Function
  ///         Function
  function::Function
  read_checkpoint(std::shared_ptr<const function::FunctionSpace> V,
                  std::string func_name, std::int64_t counter = -1) const;

  /// Read first mesh::MeshFunction from file
  /// @param meshfunction (_MeshFunction<bool>_)
  ///        mesh::MeshFunction to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshFunction<bool> read_mf_bool(std::shared_ptr<const mesh::Mesh> mesh,
                                        std::string name = "") const;

  /// Read first mesh::MeshFunction from file
  /// @param meshfunction (_MeshFunction<int>_)
  ///        mesh::MeshFunction to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshFunction<int> read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                                      std::string name = "") const;

  /// Read mesh::MeshFunction from file, optionally specifying dataset name
  /// @param meshfunction (_MeshFunction<std::size_t>_)
  ///        mesh::MeshFunction to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshFunction<std::size_t>
  read_mf_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                 std::string name = "") const;

  /// Read mesh::MeshFunction from file, optionally specifying dataset name
  /// @param meshfunction (_MeshFunction<double>_)
  ///        mesh::MeshFunction to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshFunction<double>
  read_mf_double(std::shared_ptr<const mesh::Mesh> mesh,
                 std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying dataset
  /// name
  /// @param mvc (_MeshValueCollection<bool>_)
  ///        mesh::MeshValueCollection to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshValueCollection<bool>
  read_mvc_bool(std::shared_ptr<const mesh::Mesh> mesh,
                std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying dataset
  /// name
  /// @param mvc (_MeshValueCollection<int>_)
  ///        mesh::MeshValueCollection to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshValueCollection<int>
  read_mvc_int(std::shared_ptr<const mesh::Mesh> mesh,
               std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying dataset
  /// name
  /// @param mvc (_MeshValueCollection<std::size_t>_)
  ///        mesh::MeshValueCollection to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshValueCollection<std::size_t>
  read_mvc_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                  std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying dataset
  /// name
  /// @param mvc (_MeshValueCollection<double>_)
  ///        mesh::MeshValueCollection to restore
  /// @param name (std::string)
  ///        Name of data attribute in XDMF file
  mesh::MeshValueCollection<double>
  read_mvc_double(std::shared_ptr<const mesh::Mesh> mesh,
                  std::string name = "") const;

  // Rewrite the mesh at every time step in a time series. Should be
  // turned off if the mesh remains constant.
  bool rewrite_function_mesh = true;

  // function::Functions share the same mesh for the same time step. The
  // files produced are smaller and work better in Paraview
  bool functions_share_mesh = false;

  // FIXME: This is only relevant to HDF5
  // Flush datasets to disk at each timestep. Allows inspection of the
  // HDF5 file whilst running, at some performance cost.
  bool flush_output = false;

private:
  // Generic MVC writer
  template <typename T>
  void write_mesh_value_collection(const mesh::MeshValueCollection<T>& mvc);

  // Generic MVC reader
  template <typename T>
  mesh::MeshValueCollection<T>
  read_mesh_value_collection(std::shared_ptr<const mesh::Mesh> mesh,
                             std::string name) const;

  // Remap meshfunction data, scattering data to appropriate processes
  template <typename T>
  static void
  remap_meshfunction_data(mesh::MeshFunction<T>& meshfunction,
                          const std::vector<std::int64_t>& topology_data,
                          const std::vector<T>& value_data);

  // Add mesh to XDMF xml_node (usually a Domain or Time Grid) and
  // write data
  static void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const mesh::Mesh& mesh, const std::string path_prefix);

  // Add function to a XML node
  static void add_function(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                           std::string h5_path, const function::Function& u,
                           std::string function_name, const mesh::Mesh& mesh,
                           const std::string component = "");

  // Add set of points to XDMF xml_node and write data
  static void add_points(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                         const std::vector<geometry::Point>& points);

  // Add topology node to xml_node (includes writing data to XML or  HDF5
  // file)
  template <typename T>
  static void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                hid_t h5_id, const std::string path_prefix,
                                const mesh::Mesh& mesh, int tdim);

  // Add geometry node and data to xml_node
  static void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                hid_t h5_id, const std::string path_prefix,
                                const mesh::Mesh& mesh);

  // Add DataItem node to an XML node. If HDF5 is open (h5_id > 0) the
  // data is written to the HDFF5 file with the path 'h5_path'.
  // Otherwise, data is witten to the XML node and 'h5_path' is ignored
  template <typename T>
  static void add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
                            hid_t h5_id, const std::string h5_path, const T& x,
                            const std::vector<std::int64_t> dimensions,
                            const std::string number_type = "");

  // Calculate set of entities of dimension cell_dim which are
  // duplicated on other processes and should not be output on this
  // process
  static std::set<std::uint32_t>
  compute_nonlocal_entities(const mesh::Mesh& mesh, int cell_dim);

  // Return topology data on this process as a flat vector
  template <typename T>
  static std::vector<T> compute_topology_data(const mesh::Mesh& mesh,
                                              int cell_dim);

  // Return data which is local
  template <typename T>
  std::vector<T> compute_value_data(const mesh::MeshFunction<T>& meshfunction);

  // Get DOLFIN cell type string from XML topology node
  static std::pair<std::string, int>
  get_cell_type(const pugi::xml_node& topology_node);

  // Get dimensions from an XML DataSet node
  static std::vector<std::int64_t>
  get_dataset_shape(const pugi::xml_node& dataset_node);

  // Get number of cells from an XML Topology node
  static std::int64_t get_num_cells(const pugi::xml_node& topology_node);

  // Return data associated with a data set node
  template <typename T>
  static std::vector<T>
  get_dataset(MPI_Comm comm, const pugi::xml_node& dataset_node,
              const boost::filesystem::path& parent_path,
              std::array<std::int64_t, 2> range = {{0, 0}});

  // Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem
  // node
  static std::array<std::string, 2>
  get_hdf5_paths(const pugi::xml_node& dataitem_node);

  static std::string get_hdf5_filename(std::string xdmf_filename);

  // Generic mesh::MeshFunction reader
  template <typename T>
  mesh::MeshFunction<T>
  read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
                     std::string name = "") const;

  // Generic mesh::MeshFunction writer
  template <typename T>
  void write_mesh_function(const mesh::MeshFunction<T>& meshfunction);

  // Get data width - normally the same as u.value_size(), but expand
  // for 2D vector/tensor because XDMF presents everything as 3D
  static std::int64_t get_padded_width(const function::Function& u);

  // Returns true for DG0 function::Functions
  static bool has_cell_centred_data(const function::Function& u);

  // Get point data values for linear or quadratic mesh into flattened
  // 2D array
  static std::vector<PetscScalar>
  get_point_data_values(const function::Function& u);

  // Get point data values collocated at P2 geometry points (vertices
  // and edges) flattened as a 2D array
  static std::vector<PetscScalar>
  get_p2_data_values(const function::Function& u);

  // Get cell data values as a flattened 2D array
  static std::vector<PetscScalar>
  get_cell_data_values(const function::Function& u);

  // Check that string is the same on all processes. Returns true of
  // same on all processes.
  bool name_same_on_all_procs(std::string name) const;

  // Generate the XDMF format string based on the Encoding
  // enumeration
  static std::string xdmf_format_str(Encoding encoding)
  {
    return (encoding == XDMFFile::Encoding::HDF5) ? "HDF" : "XML";
  }

  static std::string vtk_cell_type_str(mesh::CellType::Type cell_type,
                                       int order);

  // Return a string of the form "x y"
  template <typename X, typename Y>
  static std::string to_string(X x, Y y);

  // Return a vector of numerical values from a vector of
  // stringstream
  template <typename T>
  static std::vector<T> string_to_vector(const std::vector<std::string>& x_str);

  // Convert a value_rank to the XDMF string description (Scalar,
  // Vector, Tensor)
  static std::string rank_to_string(std::size_t value_rank);

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;

  // HDF5 data file
  std::unique_ptr<HDF5File> _hdf5_file;

  // Cached filename
  const std::string _filename;

  // Counter for time series
  std::size_t _counter;

  // The XML document currently representing the XDMF which needs to be
  // kept open for time series etc.
  std::unique_ptr<pugi::xml_document> _xml_doc;

  const Encoding _encoding;
};

#ifndef DOXYGEN_IGNORE
// Specialisation for std::vector<bool>, as HDF5 does not support it
// natively
template <>
inline void XDMFFile::add_data_item(MPI_Comm comm, pugi::xml_node& xml_node,
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
#endif
} // namespace io
} // namespace dolfin
