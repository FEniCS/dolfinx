// Copyright (C) 2002-2009 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2010.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2010-08-17

#include <boost/filesystem.hpp>
#include <fstream>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/function/Function.h>
#include "File.h"
#include "XMLFile.h"
#include "MatlabFile.h"
#include "OctaveFile.h"
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

  // Create directory if we have a parent path
  if ( path.has_parent_path() )
  {
    const boost::filesystem::path directory = path.parent_path();
    cout << "Creating directory \"" << directory.string() << "\"." << endl;
    boost::filesystem::create_directories(directory);
  }

  // Choose format based on extension
  if (extension == ".gz")
  {
    // Get suffix after discarding .gz
    const std::string ext = boost::filesystem::extension(boost::filesystem::basename(path));
    if (ext == ".xml")
      file.reset(new XMLFile(filename, true));
    else
      error("Unknown file type for \"%s\".", filename.c_str());
  }
  else if (extension == ".xml")
    file.reset(new XMLFile(filename, false));
  else if (extension == ".m")
    file.reset(new OctaveFile(filename));
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
  case matlab:
    file.reset(new MatlabFile(filename));
    break;
  case octave:
    file.reset(new OctaveFile(filename));
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
