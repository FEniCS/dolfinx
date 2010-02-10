// Copyright (C) 2002-2009 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2010-02-10

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
  // Choose file type base on suffix.
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);
  if (extension == ".gz")
  {
    // Get suffix after discarding .gz
    const std::string ext = boost::filesystem::extension(boost::filesystem::basename(path));
    if (ext == ".xml")
      file = new XMLFile(filename, true);
    else
      error("Unknown file type for \"%s\".", filename.c_str());
  }
  else if (extension == ".xml")
    file = new XMLFile(filename, false);
  else if (extension == ".m")
    file = new OctaveFile(filename);
  else if (extension == ".py")
    file = new PythonFile(filename);
  else if (extension == ".pvd")
    file = new VTKFile(filename, encoding);
  else if (extension == ".raw")
    file = new RAWFile(filename);
  else if (extension == ".xyz")
    file = new XYZFile(filename);
  else if (extension == ".bin")
    file = new BinaryFile(filename);
  else
    error("Unknown file type for \"%s\".", filename.c_str());
}
//-----------------------------------------------------------------------------
File::File(const std::string filename, Type type, std::string encoding)
{
  switch (type)
  {
  case xml:
    file = new XMLFile(filename, false);
    break;
  case matlab:
    file = new MatlabFile(filename);
    break;
  case octave:
    file = new OctaveFile(filename);
    break;
  case python:
    file = new PythonFile(filename);
    break;
  case vtk:
    file = new VTKFile(filename, encoding);
    break;
  case raw:
    file = new RAWFile(filename);
    break;
  case xyz:
    file = new XYZFile(filename);
    break;
  case binary:
    file = new BinaryFile(filename);
    break;
  default:
    file = 0;
    error("Unknown file type for \"%s\".", filename.c_str());
  }
}
//-----------------------------------------------------------------------------
File::File(std::ostream& outstream)
{
  file = new XMLFile(outstream);
}
//-----------------------------------------------------------------------------
File::~File()
{
  delete file;
  file = 0;
}
//-----------------------------------------------------------------------------
void File::operator<<(const Function& u)
{
  u.gather();
  file->write();
  *file << u;
}
//-----------------------------------------------------------------------------
bool File::exists(std::string filename)
{
  std::ifstream file(filename.c_str());
  if (!file.is_open())
    return false;
  file.close();
  return true;
}
//-----------------------------------------------------------------------------
