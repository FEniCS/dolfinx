// Copyright (C) 2002-2012 Johan Hoffman, Anders Logg and Garth N. Wells
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
// Modified by Magnus Vikstrom 2007
// Modified by Nuno Lopes 2008
// Modified by Ola Skavhaug 2009

#ifndef __FILE_H
#define __FILE_H

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <dolfin/common/MPI.h>
#include "GenericFile.h"

namespace dolfin
{

  /// A File represents a data file for reading and writing objects.
  /// Unless specified explicitly, the format is determined by the
  /// file name suffix.

  /// A list of objects that can be read/written to file can be found in
  /// GenericFile.h. Compatible file formats include:
  ///     * Binary (.bin)
  ///     * RAW    (.raw)
  ///     * SVG    (.svg)
  ///     * XD3    (.xd3)
  ///     * XML    (.xml)
  ///     * XYZ    (.xyz)
  ///     * VTK    (.pvd)

  class File
  {
  public:

    /// File formats
    enum class Type {x3d, xml, vtk, raw, xyz, binary, svg};

    /// Create a file with given name
    ///
    /// *Arguments*
    ///     filename (std::string)
    ///         Name of file.
    ///     encoding (std::string)
    ///         Optional argument specifying encoding, ASCII is default.
    ///
    /// *Example*
    ///    .. code-block:: c++
    ///
    ///         // Save solution to file
    ///         File file("solution.pvd");
    ///         file << u;
    ///
    ///         // Read mesh data from file
    ///         File mesh_file("mesh.xml");
    ///         mesh_file >> mesh;
    ///
    ///         // Using compressed binary format
    ///         File comp_file("solution.pvd", "compressed");
    ///
    File(const std::string filename, std::string encoding="ascii");

    /// Create a file with given name with MPI communicator
    ///
    /// *Arguments*
    ///     communicator (MPI_Comm)
    ///         The MPI communicator.
    ///     filename (std::string)
    ///         Name of file.
    ///     encoding (std::string)
    ///         Optional argument specifying encoding, ascii is default.
    ///
    /// *Example*
    ///    .. code-block:: c++
    ///
    ///         // Save solution to file
    ///         File file(u.mesh()->mpi_comm(), "solution.pvd");
    ///         file << u;
    ///
    ///         // Read mesh data from file
    ///         File mesh_file(MPI_COMM_WORLD, "mesh.xml");
    ///         mesh_file >> mesh;
    ///
    ///         // Using compressed binary format
    ///         File comp_file(u.mesh()->mpi_comm(), "solution.pvd",
    ///                        "compressed");
    ///
    File(MPI_Comm comm, const std::string filename,
         std::string encoding="ascii");

    /// Create a file with given name and type (format)
    ///
    /// *Arguments*
    ///     filename (std::string)
    ///         Name of file.
    ///     type (Type)
    ///         File format.
    ///     encoding (std::string)
    ///         Optional argument specifying encoding, ascii is default.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("solution", vtk);
    ///
    File(const std::string filename, Type type, std::string encoding="ascii");

    /// Create a file with given name and type (format) with MPI communicator
    ///
    /// *Arguments*
    ///     communicator (MPI_Comm)
    ///         The MPI communicator.
    ///     filename (std::string)
    ///         Name of file.
    ///     type (Type)
    ///         File format.
    ///     encoding (std::string)
    ///         Optional argument specifying encoding, ascii is default.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file(MPI_COMM_WORLD, "solution", vtk);
    ///
    File(MPI_Comm comm, const std::string filename, Type type,
         std::string encoding="ascii");

    /// Create an outfile object writing to stream
    ///
    /// *Arguments*
    ///     outstream (std::ostream)
    ///         The stream.
    File(std::ostream& outstream);

    /// Destructor
    ~File();

    /// Read from file
    template<typename T> void operator>>(T& t)
    {
      file->_read();
      *file >> t;
    }

    /// Write Function to file
    //void operator<<(const Function& u);

    /// Write Mesh to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("mesh.pvd", "compressed");
    ///         file << std::make_pair<const Mesh*, double>(&mesh, t);
    ///
    void operator<<(const std::pair<const Mesh*, double> mesh);

    /// Write MeshFunction to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("markers.pvd", "compressed");
    ///         file << std::make_pair<const MeshFunction<int>*, double>(&f, t);
    ///
    void operator<<(const std::pair<const MeshFunction<int>*, double> f);

    /// Write MeshFunction to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("markers.pvd", "compressed");
    ///         file << std::make_pair<const MeshFunction<std::size_t>*, double>(&f, t);
    ///
    void operator<<
      (const std::pair<const MeshFunction<std::size_t>*, double> f);

    /// Write MeshFunction to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("markers.pvd", "compressed");
    ///         file << std::make_pair<const MeshFunction<double>*, double>(&f, t);
    ///
    void operator<< (const std::pair<const MeshFunction<double>*, double> f);

    /// Write MeshFunction to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("markers.pvd", "compressed");
    ///         file << std::make_pair<const MeshFunction<bool>*, double>(&f, t);
    ///
    void operator<<(const std::pair<const MeshFunction<bool>*, double> f);

    /// Write Function to file with time
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         File file("solution.pvd", "compressed");
    ///         file << std::make_pair<const Function*, double>(&u, t);
    ///
    void operator<<(const std::pair<const Function*, double> u);

    /// Write object to file
    template<typename T> void operator<<(const T& t)
    {
      file->_write(MPI::rank(_mpi_comm));
      *file << t;
    }

    /// Check if file exists
    ///
    /// *Arguments*
    ///     filename (std::string)
    ///         Name of file.
    ///
    /// *Returns*
    ///     bool
    ///         True if the file exists.
    static bool exists(std::string filename);

    // Create parent path for file if file has a parent path
    ///
    /// *Arguments*
    ///     filename (std::string)
    ///         Name of file / path.
    static void create_parent_path(std::string filename);

  private:

    // Initialise GenericFile (using file extension to determine type)
    void init(MPI_Comm comm, const std::string filename, std::string encoding);

    // Initialise GenericFile  (with specified type)
    void init(MPI_Comm comm, const std::string filename, Type type,
              std::string encoding);

    // FIXME: Remove when GenericFile::write is cleaned up
    // MPI communicator
    const MPI_Comm _mpi_comm;

    // Pointer to implementation (envelope-letter design)
    std::unique_ptr<GenericFile> file;

  };

}

#endif
