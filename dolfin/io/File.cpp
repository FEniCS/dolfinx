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
// Last changed: 2009-06-15

#include <string>
#include <boost/filesystem.hpp>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "File.h"
#include "GenericFile.h"
#include "XMLFile.h"
#include "MatlabFile.h"
#include "OctaveFile.h"
#include "PythonFile.h"
#include "PVTKFile.h"
#include "VTKFile.h"
#include "RAWFile.h"
#include "XYZFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string filename)
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
  {
    if (MPI::num_processes() > 1)
      file = new PVTKFile(filename);
    else
      file = new VTKFile(filename);
  }
  else if (extension == ".raw")
    file = new RAWFile(filename);
  else if (extension == ".xyz")
    file = new XYZFile(filename);
  else
    error("Unknown file type for \"%s\".", filename.c_str());
}
//-----------------------------------------------------------------------------
File::File(const std::string filename, Type type)
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
  case vtk:
    file = new VTKFile(filename);
    break;
  case python:
    file = new PythonFile(filename);
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
void File::operator>> (GenericVector& x)
{
  file->read();

  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>> (GenericMatrix& A)
{
  file->read();

  *file >> A;
}
//-----------------------------------------------------------------------------
void File::operator>> (Mesh& mesh)
{
  file->read();

  *file >> mesh;
}
//-----------------------------------------------------------------------------
void File::operator>> (LocalMeshData& data)
{
  file->read();

  *file >> data;
}
//-----------------------------------------------------------------------------
void File::operator>> (MeshFunction<int>& meshfunction)
{
  file->read();

  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>> (MeshFunction<uint>& meshfunction)
{
  file->read();

  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>> (MeshFunction<double>& meshfunction)
{
  file->read();

  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>> (MeshFunction<bool>& meshfunction)
{
  file->read();

  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>> (Sample& sample)
{
  file->read();

  *file >> sample;
}
//-----------------------------------------------------------------------------
void File::operator>> (ParameterList& parameters)
{
  file->read();

  *file >> parameters;
}
//-----------------------------------------------------------------------------
void File::operator>> (Graph& graph)
{
  file->read();

  *file >> graph;
}
//-----------------------------------------------------------------------------
void File::operator>> (FunctionPlotData& data)
{
  file->read();

  *file >> data;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::vector<int>& x)
{
  file->read();

  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::vector<uint>& x)
{
  file->read();

  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::vector<double>& x)
{
  file->read();

  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, int>& map)
{
  file->read();

  *file >> map;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, uint>& map)
{
  file->read();

  *file >> map;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, double>& map)
{
  file->read();

  *file >> map;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, std::vector<int> >& array_map)
{
  file->read();

  *file >> array_map;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, std::vector<uint> >& array_map)
{
  file->read();

  *file >> array_map;
}
//-----------------------------------------------------------------------------
void File::operator>> (std::map<uint, std::vector<double> >& array_map)
{
  file->read();

  *file >> array_map;
}
//-----------------------------------------------------------------------------
void File::operator<< (const GenericVector& x)
{
  file->write();

  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<< (const GenericMatrix& A)
{
  file->write();

  *file << A;
}
//-----------------------------------------------------------------------------
void File::operator<< (const Mesh& mesh)
{
  file->write();

  *file << mesh;
}
//-----------------------------------------------------------------------------
void File::operator<< (const LocalMeshData& data)
{
  file->write();

  *file << data;
}
//-----------------------------------------------------------------------------
void File::operator<< (const MeshFunction<int>& meshfunction)
{
  file->write();

  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<< (const MeshFunction<uint>& meshfunction)
{
  file->write();

  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<< (const MeshFunction<double>& meshfunction)
{
  file->write();

  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<< (const MeshFunction<bool>& meshfunction)
{
  file->write();

  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<< (const Function& v)
{
  file->write();

  *file << v;
}
//-----------------------------------------------------------------------------
void File::operator<< (const Sample& sample)
{
  file->write();

  *file << sample;
}
//-----------------------------------------------------------------------------
void File::operator<< (const ParameterList& parameters)
{
  file->write();

  *file << parameters;
}
//-----------------------------------------------------------------------------
void File::operator<< (const Graph& graph)
{
  file->write();

  *file << graph;
}
//-----------------------------------------------------------------------------
void File::operator<< (const FunctionPlotData& data)
{
  file->write();

  *file << data;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::vector<int>& x)
{
  file->write();

  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::vector<uint>& x)
{
  file->write();

  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::vector<double>& x)
{
  file->write();

  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::map<uint, int>& array_map)
{
  file->write();

  *file << array_map;
}
void File::operator<< (const std::map<uint, uint>& array_map)
{
  file->write();

  *file << array_map;
}
void File::operator<< (const std::map<uint, double>& array_map)
{
  file->write();

  *file << array_map;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::map<uint, std::vector<int> >& array_map)
{
  file->write();

  *file << array_map;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::map<uint, std::vector<uint> >& array_map)
{
  file->write();

  *file << array_map;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::map<uint, std::vector<double> >& array_map)
{
  file->write();

  *file << array_map;
}
//-----------------------------------------------------------------------------
