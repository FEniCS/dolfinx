// Copyright (C) 2002-2009 Johan Hoffman and Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2005-2010.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2010-09-23

#include <fstream>
#include <boost/filesystem.hpp>
#include <dolfin/function/Function.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "File.h"
#include "XMLFile.h"
#include "PythonFile.h"
#include "VTKFile.h"
#include "RAWFile.h"
#include "XYZFile.h"
#include "BinaryFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string filename, std::string encoding)
{
  // Get file path and extension
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  // Create directory if we have a parent path (does nothing if directory exists)
  if (path.has_parent_path())
  {
    boost::filesystem::create_directories(path.parent_path());
    if (!boost::filesystem::is_directory(path.parent_path()))
      error("Could not create directory %s.", path.parent_path().string().c_str());
  }

  // Choose format based on extension
  if (extension == ".gz")
  {
    // Get suffix after discarding .gz
    const std::string ext =
      boost::filesystem::extension(boost::filesystem::basename(path));
    if (ext == ".xml")
      file.reset(new XMLFile(filename, true));
    else
      error("Unknown file type for \"%s\".", filename.c_str());
  }
  else if (extension == ".xml")
    file.reset(new XMLFile(filename, false));
  else if (extension == ".py")
    file.reset(new PythonFile(filename));
  else if (extension == ".pvd")
    file.reset(new VTKFile(filename, encoding));
  else if (extension == ".raw")
    file.reset(new RAWFile(filename));
  else if (extension == ".xyz")
    file.reset(new XYZFile(filename));
  else if (extension == ".bin")
    file.reset(new BinaryFile(filename));
  else
    error("Unknown file type for \"%s\".", filename.c_str());
}
//-----------------------------------------------------------------------------
File::File(const std::string filename, Type type, std::string encoding)
{
  switch (type)
  {
  case xml:
    file.reset(new XMLFile(filename, false));
    break;
  case python:
    file.reset(new PythonFile(filename));
    break;
  case vtk:
    file.reset(new VTKFile(filename, encoding));
    break;
  case raw:
    file.reset(new RAWFile(filename));
    break;
  case xyz:
    file.reset(new XYZFile(filename));
    break;
  case binary:
    file.reset(new BinaryFile(filename));
    break;
  default:
    error("Unknown file type for \"%s\".", filename.c_str());
  }
}
//-----------------------------------------------------------------------------
File::File(std::ostream& outstream)
{
  file.reset(new XMLFile(outstream));
}
//-----------------------------------------------------------------------------
File::~File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void File::operator<<(const Function& u)
{
  u.gather();
  file->write();
  *file << u;
}
//-----------------------------------------------------------------------------
void File::operator<<(const std::pair<const Function*, double> u)
{
  u.first->gather();
  file->write();
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
