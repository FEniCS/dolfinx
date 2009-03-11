// Copyright (C) 2002-2009 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005, 2006.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2009-03-04

#include <string>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "File.h"
#include "GenericFile.h"
#include "NewXMLFile.h"
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
File::File(const std::string& filename, bool new_style)
{
  // Choose file type base on suffix.

  // FIXME: Use correct funtion to find the suffix; using rfind() makes
  // FIXME: it essential that the suffixes are checked in the correct order.

  if (filename.rfind(".xml.gz") != filename.npos)
    if ( new_style )
      file = new NewXMLFile(filename, true);
    else
      file = new XMLFile(filename, true);
  else if (filename.rfind(".xml") != filename.npos)
    if ( new_style )
      file = new NewXMLFile(filename, false);
    else
      file = new XMLFile(filename, false);
  else if (filename.rfind(".m") != filename.npos)
    file = new OctaveFile(filename);
  else if (filename.rfind(".py") != filename.npos)
    file = new PythonFile(filename);
  else if (filename.rfind(".pvd") != filename.npos)
  {
    if (MPI::num_processes() > 1)
      file = new PVTKFile(filename);
    else
      file = new VTKFile(filename);
  }
  else if (filename.rfind(".raw") != filename.npos)
    file = new RAWFile(filename);
  else if (filename.rfind(".xyz") != filename.npos)
    file = new XYZFile(filename);
  else
  {
    file = 0;
    error("Unknown file type for \"%s\".", filename.c_str());
  }
}
//-----------------------------------------------------------------------------
File::File(const std::string& filename, Type type)
{
  switch (type) {
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
  file = new NewXMLFile(outstream);
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
void File::operator>> (Function& v)
{
  file->read();
  
  *file >> v;
}
//-----------------------------------------------------------------------------
void File::operator>> (Sample& sample)
{
  file->read();
  
  *file >> sample;
}
//-----------------------------------------------------------------------------
void File::operator>> (FiniteElementSpec& spec)
{
  file->read();
  
  *file >> spec;
}
//-----------------------------------------------------------------------------
void File::operator>> (ParameterList& parameters)
{
  file->read();
  
  *file >> parameters;
}
//-----------------------------------------------------------------------------
void File::operator>> (BLASFormData& blas)
{
  file->read();
  
  *file >> blas;
}
//-----------------------------------------------------------------------------
void File::operator>> (Graph& graph)
{
  file->read();
  
  *file >> graph;
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
void File::operator<< (const FiniteElementSpec& spec)
{
  file->write();
  
  *file << spec;
}
//-----------------------------------------------------------------------------
void File::operator<< (const ParameterList& parameters)
{
  file->write();
  
  *file << parameters;
}
//-----------------------------------------------------------------------------
void File::operator<< (const BLASFormData& blas)
{
  file->write();
  
  *file << blas;
}
//-----------------------------------------------------------------------------
void File::operator<< (const Graph& graph)
{
  file->write();
  
  *file << graph;
}
//-----------------------------------------------------------------------------
void File::operator<< (const std::vector<int>& x)
{
  file->write();
  
  *file << x;
}
//-----------------------------------------------------------------------------void File::operator<< (const std::vector<int>& ix)
void File::operator<< (const std::vector<uint>& x)
{
  file->write();
  
  *file << x;
}
//-----------------------------------------------------------------------------void File::operator<< (const std::vector<int>& ix)
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
