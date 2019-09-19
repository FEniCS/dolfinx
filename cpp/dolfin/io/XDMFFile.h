// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfin/common/MPI.h>
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

namespace mesh
{
enum class GhostMode : int;
class Mesh;
template <typename T>
class MeshFunction;
template <typename T>
class MeshValueCollection;
class Partitioning;
} // namespace mesh

namespace io
{
class HDF5File;

/// Read and write mesh::Mesh, function::Function, mesh::MeshFunction
/// and other objects in XDMF.

/// This class supports the output of meshes and functions in XDMF
/// (http://www.xdmf.org) format. It creates an XML file that describes
/// the data and points to a HDF5 file that stores the actual problem
/// data. Output of data in parallel is supported.
///
/// XDMF is not suitable for checkpointing as it may decimate some data.

// FIXME: Set read mode when creating file object?

// FIXME: Set encoding when opening file

// FIXME: Remove the duplicate read_mf_foo functions. Challenge is the
// templated reader code would then expose a lot code publically.
// Refactor large, templated functions into parts that (i) depend on the
// template argument and (ii) parts that do not. Same applies to
// MeshValueCollection.

class XDMFFile
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

  /// Save a mesh to XDMF format, either using an associated HDF5 file,
  /// or storing the data inline as XML Create function on given
  /// function space
  /// @param[in] mesh The Mesh to save
  void write(const mesh::Mesh& mesh);

  /// Save a function::Function to XDMF file for checkpointing, using an
  /// associated HDF5 file, or storing the data inline as XML.
  ///
  /// If the file where we would like to write exists, then the function
  /// is appended to the file.
  ///
  /// @param[in] u The Function to save
  /// @param[in] function_name Name of the function used in XDMF XML and
  ///                          HDF file (if encoding = HDF). The string
  ///                          is used to fill value for XML attribute
  ///                          Name of Grid node. It is also used in HDF
  ///                          file in path to datasets. Must be the
  ///                          same on all processes in parallel.
  /// @param[in] time_step The current time (optional). It is saved only
  ///                      in XDMF file. The function could be saved
  ///                      with the same time step several times. There
  ///                      is an internal "counter" value stored in XDMF
  ///                      which differentiates the same time steps.
  void write_checkpoint(const function::Function& u, std::string function_name,
                        double time_step = 0.0);

  /// Save a function::Function to XDMF file for visualisation, using an
  /// associated HDF5 file, or storing the data inline as XML.
  ///
  /// @param[in] u The Function to save
  void write(const function::Function& u);

  /// Save a function::Function with timestamp to XDMF file for
  /// visualisation, using an associated HDF5 file, or storing the data
  /// inline as XML.
  ///
  /// You can control the output with the following boolean parameters
  /// on the XDMFFile class:
  ///
  /// * rewrite_function_mesh (default true): Controls whether the mesh
  ///   will be rewritten every timestep. If the mesh does not change
  ///   this can be turned off to create smaller files.
  ///
  /// * functions_share_mesh (default false): Controls whether all
  ///   functions on a single time step share the same mesh. If true the
  ///   files created will be smaller and also behave better in
  ///   Paraview, at least in version 5.3.0
  ///
  /// @param[in] u The Function to save
  /// @param[in] t The time
  void write(const function::Function& u, double t);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param[in] meshfunction The meshfunction to save
  void write(const mesh::MeshFunction<int>& meshfunction);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param[in] meshfunction The meshfunction to save
  void write(const mesh::MeshFunction<std::size_t>& meshfunction);

  /// Save mesh::MeshFunction to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param[in] meshfunction The meshfunction to save
  void write(const mesh::MeshFunction<double>& meshfunction);

  /// Write out mesh value collection (subset) using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param[in] mvc The mesh::MeshValueCollection to save
  void write(const mesh::MeshValueCollection<int>& mvc);

  /// Write out mesh value collection (subset) using an associated HDF5
  /// file, or storing the data inline as XML.
  ///
  /// @param[in] mvc The mesh::MeshValueCollection to save
  void write(const mesh::MeshValueCollection<std::size_t>& mvc);

  /// Write out mesh value collection (subset) using an associated HDF5
  /// file, or storing the data inline as XML.
  ///
  /// @param[in] mvc The mesh::MeshValueCollection to save
  void write(const mesh::MeshValueCollection<double>& mvc);

  /// Save a cloud of points to file using an associated HDF5 file, or
  /// storing the data inline as XML.
  ///
  /// @param[in] points A list of points to save.
  void write(const
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
          points);

  /// Save a cloud of points, with scalar values using an associated
  /// HDF5 file, or storing the data inline as XML.
  ///
  /// @param[in] points A list of points to save.
  /// @param[in] values A list of values at each point
  void write(const
      Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
          points,
      const std::vector<double>& values);

  /// Read in the first mesh::Mesh in XDMF file
  ///
  /// @param[in] ghost_mode Ghost mode for mesh partition
  /// @return A distributed Mesh
  mesh::Mesh read_mesh(const mesh::GhostMode ghost_mode) const;

  /// Read in the data from the first mesh in XDMF file
  ///
  /// @param[in] comm The MPI Communicator
  /// @return Geometric points on each process (EigenRowArrayXXd),
  ///         Topological cells with global vertex indexing
  ///         (EigenRowArrayXXi64), and the Cell type (mesh::CellType)
  std::tuple<mesh::CellType, EigenRowArrayXXd, EigenRowArrayXXi64,
             std::vector<std::int64_t>>
  read_mesh_data(MPI_Comm comm) const;

  /// Read a function from the XDMF file. Supplied function must come
  /// with already initialized and compatible function space.
  ///
  /// function::Functions saved as time-series must be addressed through
  /// (integer) internal counter and function name. function::Function
  /// name is a name of function that was saved. Counter stands for the
  /// number of function from time-series, e.g. counter=0 refers to
  /// first saved function regardless of its time-step value.
  ///
  /// @param[in] V The FunctionSpace
  /// @param[in] func_name A name of a function to read. Must be the
  ///                      same on all processes in parallel.
  /// @param[in] counter Internal integer counter - used in time-series.
  ///                    Default value is -1 which points to last saved
  ///                    function. Counter works same as python array
  ///                    position key, i.e. counter = -2 points to the
  ///                    function before the last one.
  /// @return A Function
  function::Function
  read_checkpoint(std::shared_ptr<const function::FunctionSpace> V,
                  std::string func_name, std::int64_t counter = -1) const;

  /// Read first mesh::MeshFunction from file
  /// @param[in] mesh The associated Mesh
  /// @param[in] name Name of data attribute in XDMF file
  /// @return A MeshFunction
  mesh::MeshFunction<int> read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                                      std::string name = "") const;

  /// Read mesh::MeshFunction from file, optionally specifying dataset
  /// name
  /// @param[in] mesh (_MeshFunction<std::size_t>_)
  ///        mesh::MeshFunction to restore
  /// @param name (std::string) Name of data attribute in XDMF file
  /// @return A MeshFunction
  mesh::MeshFunction<std::size_t>
  read_mf_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                 std::string name = "") const;

  /// @param[in] mesh The associated Mesh
  /// @param[in] name Name of data attribute in XDMF file
  /// @return A MeshFunction
  mesh::MeshFunction<double>
  read_mf_double(std::shared_ptr<const mesh::Mesh> mesh,
                 std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying
  /// dataset name
  /// @param[in] mesh The associated Mesh
  /// @param[in] name Name of data attribute in XDMF file
  /// @return A MeshValueCollection
  mesh::MeshValueCollection<int>
  read_mvc_int(std::shared_ptr<const mesh::Mesh> mesh,
               std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying
  /// dataset name
  /// @param[in] mesh The associated Mesh
  /// @param[in] name Name of data attribute in XDMF file
  /// @return A MeshValueCollection
 mesh::MeshValueCollection<std::size_t>
  read_mvc_size_t(std::shared_ptr<const mesh::Mesh> mesh,
                  std::string name = "") const;

  /// Read mesh::MeshValueCollection from file, optionally specifying dataset
  /// name
  /// @param[in] mesh The associated Mesh
  /// @param[in] name Name of data attribute in XDMF file
  /// @return A MeshValueCollection
  mesh::MeshValueCollection<double>
  read_mvc_double(std::shared_ptr<const mesh::Mesh> mesh,
                  std::string name = "") const;

  /// Rewrite the mesh at every time step in a time series. Should be
  /// turned off if the mesh remains constant.
  bool rewrite_function_mesh = true;

  /// function::Functions share the same mesh for the same time step. The
  /// files produced are smaller and work better in Paraview
  bool functions_share_mesh = false;

  /// FIXME: This is only relevant to HDF5
  /// Flush datasets to disk at each timestep. Allows inspection of the
  /// HDF5 file whilst running, at some performance cost.
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

  // Generic mesh::MeshFunction reader
  template <typename T>
  mesh::MeshFunction<T>
  read_mesh_function(std::shared_ptr<const mesh::Mesh> mesh,
                     std::string name = "") const;

  // Generic mesh::MeshFunction writer
  template <typename T>
  void write_mesh_function(const mesh::MeshFunction<T>& meshfunction);

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

} // namespace io
} // namespace dolfin
