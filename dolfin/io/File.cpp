// Copyright (C) 2002-2011 Johan Hoffman, Anders Logg and Garth N. Wells
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
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2015-03-27

#include <clocale>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/log/log.h>
#include "RAWFile.h"
#include "SVGFile.h"
#include "VTKFile.h"
#include "X3DFile.h"
#include "XMLFile.h"
#include "XYZFile.h"

#include "File.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string filename, std::string encoding)
  : _mpi_comm(MPI_COMM_WORLD)
{
  init(MPI_COMM_WORLD, filename, encoding);
}
//-----------------------------------------------------------------------------
File::File(MPI_Comm comm, const std::string filename, std::string encoding)
  : _mpi_comm(comm)
{
  init(comm, filename, encoding);
}
//-----------------------------------------------------------------------------
File::File(const std::string filename, Type type, std::string encoding)
  : _mpi_comm(MPI_COMM_WORLD)
{
  init(MPI_COMM_WORLD, filename, type, encoding);
}
//-----------------------------------------------------------------------------
File::File(MPI_Comm comm, const std::string filename, Type type,
           std::string encoding) : _mpi_comm(comm)
{
  init(comm, filename, type, encoding);
}
//-----------------------------------------------------------------------------
File::File(std::ostream& outstream) : _mpi_comm(MPI_COMM_SELF)
{
  file.reset(new XMLFile(outstream));
}
//-----------------------------------------------------------------------------
File::~File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const Mesh*, double> mesh)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << mesh;
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const MeshFunction<int>*, double> f)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << f;
}
//-----------------------------------------------------------------------------
void
File::operator<<(const std::pair<const MeshFunction<std::size_t>*, double> f)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << f;
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const MeshFunction<double>*, double> f)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << f;
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const MeshFunction<bool>*, double> f)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << f;
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const Function*, double> u)
{
  file->_write(MPI::rank(_mpi_comm));
  *file << u;
}
//-----------------------------------------------------------------------------
bool File::exists(std::string filename)
{
  std::ifstream file(filename.c_str());
  if (!file.is_open())
    return false;
  else
  {
    file.close();
    return true;
  }
}
//-----------------------------------------------------------------------------
void File::create_parent_path(std::string filename)
{
  const boost::filesystem::path path(filename);
  if (path.has_parent_path()
      && !boost::filesystem::is_directory(path.parent_path()))
  {
    boost::filesystem::create_directories(path.parent_path());
    if (!boost::filesystem::is_directory(path.parent_path()))
    {
      dolfin_error("File.cpp",
                   "open file",
                   "Could not create directory \"%s\"",
                   path.parent_path().string().c_str());
    }
  }
}
//-----------------------------------------------------------------------------
void File::init(MPI_Comm comm, const std::string filename,
                std::string encoding)
{
  // Make sure locale is set to "C". This prevents spurious bugs in converting
  // from string to double. See DOLFIN Issue #498.
  std::setlocale(LC_ALL, "C");

  // Create parent path for file if file has a parent path
  create_parent_path(filename);

  // Get file path and extension
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  // Choose format based on extension
  if (extension == ".gz")
  {
    // Get suffix after discarding .gz
    const std::string ext =
      boost::filesystem::extension(boost::filesystem::basename(path));
    if (ext == ".xml")
      file.reset(new XMLFile(comm, filename));
    else
    {
      dolfin_error("File.cpp",
                   "open file",
                   "Unknown file type (\"%s\") for file \"%s\"",
                   ext.c_str(), filename.c_str());
    }
  }
  else if (extension == ".x3d")
    file.reset(new X3DFile(filename));
  else if (extension == ".xml")
    file.reset(new XMLFile(comm, filename));
  else if (extension == ".pvd")
    file.reset(new VTKFile(filename, encoding));
  else if (extension == ".raw")
    file.reset(new RAWFile(filename));
  else if (extension == ".xyz")
    file.reset(new XYZFile(filename));
  else if (extension == ".svg")
    file.reset(new SVGFile(filename));
  else
  {
    dolfin_error("File.cpp",
                 "open file",
                 "Unknown file type (\"%s\") for file \"%s\"",
                 extension.c_str(), filename.c_str());
  }
}
//-----------------------------------------------------------------------------
void File::init(MPI_Comm comm, const std::string filename, Type type,
                std::string encoding)
{
  switch (type)
  {
  case Type::x3d:
    file.reset(new X3DFile(filename));
    break;
  case Type::xml:
    file.reset(new XMLFile(comm, filename));
    break;
  case Type::vtk:
    file.reset(new VTKFile(filename, encoding));
    break;
  case Type::raw:
    file.reset(new RAWFile(filename));
    break;
  case Type::xyz:
    file.reset(new XYZFile(filename));
    break;
  default:
    dolfin_error("File.cpp",
                 "open file",
                 "Unknown file type (\"%d\") for file \"%s\"",
                 type, filename.c_str());
  }
}
//-----------------------------------------------------------------------------
