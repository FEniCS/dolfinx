// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005, 2006.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008
// Modified by Niclas Jansson 2008.
//
// First added:  2002-11-12
// Last changed: 2008-09-16

#include <string>
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
File::File(const std::string& filename)
{
  // Choose file type base on suffix.

  // FIXME: Use correct funtion to find the suffix; using rfind() makes
  // FIXME: it essential that the suffixes are checked in the correct order.

  if ( filename.rfind(".xml") != filename.npos )
    file = new XMLFile(filename);
  else if ( filename.rfind(".xml.gz") != filename.npos )
    file = new XMLFile(filename);
  else if ( filename.rfind(".m") != filename.npos )
    file = new OctaveFile(filename);
  else if ( filename.rfind(".py") != filename.npos )
    file = new PythonFile(filename);
  else if ( filename.rfind(".pvd") != filename.npos )
    if(MPI::num_processes() > 1)
      file = new PVTKFile(filename);
    else
      file = new VTKFile(filename);
  else if ( filename.rfind(".raw") != filename.npos )
    file = new RAWFile(filename);
  else if ( filename.rfind(".xyz") != filename.npos )
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
  switch ( type ) {
  case xml:
    file = new XMLFile(filename);
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
File::~File()
{
  if ( file )
    delete file;
  file = 0;
}
//-----------------------------------------------------------------------------
void File::operator>>(GenericVector& x)
{
  file->read();
  
  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>>(GenericMatrix& A)
{
  file->read();
  
  *file >> A;
}
//-----------------------------------------------------------------------------
void File::operator>>(Mesh& mesh)
{
  file->read();
  
  *file >> mesh;
}
//-----------------------------------------------------------------------------
void File::operator>>(MeshFunction<int>& meshfunction)
{
  file->read();
  
  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>>(MeshFunction<unsigned int>& meshfunction)
{
  file->read();
  
  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>>(MeshFunction<double>& meshfunction)
{
  file->read();
  
  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>>(MeshFunction<bool>& meshfunction)
{
  file->read();
  
  *file >> meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator>>(Function& f)
{
  file->read();
  
  *file >> f;
}
//-----------------------------------------------------------------------------
void File::operator>>(Sample& sample)
{
  file->read();
  
  *file >> sample;
}
//-----------------------------------------------------------------------------
void File::operator>>(FiniteElementSpec& spec)
{
  file->read();
  
  *file >> spec;
}
//-----------------------------------------------------------------------------
void File::operator>>(ParameterList& parameters)
{
  file->read();
  
  *file >> parameters;
}
//-----------------------------------------------------------------------------
void File::operator>>(BLASFormData& blas)
{
  file->read();
  
  *file >> blas;
}
//-----------------------------------------------------------------------------
void File::operator>>(Graph& graph)
{
  file->read();
  
  *file >> graph;
}
//-----------------------------------------------------------------------------
void File::operator<<(GenericVector& x)
{
  file->write();
  
  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<<(GenericMatrix& A)
{
  file->write();
	 
  *file << A;
}
//-----------------------------------------------------------------------------
void File::operator<<(Mesh& mesh)
{
  file->write();
  
  *file << mesh;
}
//-----------------------------------------------------------------------------
void File::operator<<(MeshFunction<int>& meshfunction)
{
  file->write();
  
  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<<(MeshFunction<unsigned int>& meshfunction)
{
  file->write();
  
  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<<(MeshFunction<double>& meshfunction)
{
  file->write();
  
  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<<(MeshFunction<bool>& meshfunction)
{
  file->write();
  
  *file << meshfunction;
}
//-----------------------------------------------------------------------------
void File::operator<<(Function& u)
{
  file->write();
  
  *file << u;
}
//-----------------------------------------------------------------------------
void File::operator<<(Sample& sample)
{
  file->write();
  
  *file << sample;
}
//-----------------------------------------------------------------------------
void File::operator<<(FiniteElementSpec& spec)
{
  file->write();
  
  *file << spec;
}
//-----------------------------------------------------------------------------
void File::operator<<(ParameterList& parameters)
{
  file->write();
  
  *file << parameters;
}
//-----------------------------------------------------------------------------
void File::operator<<(BLASFormData& blas)
{
  file->write();
  
  *file << blas;
}
//-----------------------------------------------------------------------------
void File::operator<<(Graph& graph)
{
  file->write();
  
  *file << graph;
}
//-----------------------------------------------------------------------------
