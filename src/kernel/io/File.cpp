// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2006.
// Modified by Haiko Etzel 2005.
//
// First added:  2002-11-12
// Last changed: 2006-11-01

#include <string>
#include <dolfin/dolfin_log.h>
#include <dolfin/File.h>
#include <dolfin/GenericFile.h>
#include <dolfin/XMLFile.h>
#include <dolfin/MatlabFile.h>
#include <dolfin/MTXFile.h>
#include <dolfin/OctaveFile.h>
#include <dolfin/OpenDXFile.h>
#include <dolfin/PythonFile.h>
#include <dolfin/GiDFile.h>
#include <dolfin/VTKFile.h>

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
  else if ( filename.rfind(".mat") != filename.npos )
    file = new MTXFile(filename);
  else if ( filename.rfind(".mtx") != filename.npos )
    file = new MTXFile(filename);
  else if ( filename.rfind(".msh") != filename.npos )
    file = new GiDFile(filename);
  else if ( filename.rfind(".res") != filename.npos )
    file = new GiDFile(filename);
  else if ( filename.rfind(".m") != filename.npos )
    file = new OctaveFile(filename);
  else if ( filename.rfind(".dx") != filename.npos )
    file = new OpenDXFile(filename);
  else if ( filename.rfind(".py") != filename.npos )
    file = new PythonFile(filename);
  else if ( filename.rfind(".pvd") != filename.npos )
    file = new VTKFile(filename);
  else
  {
    file = 0;
    dolfin_error1("Unknown file type for \"%s\".", filename.c_str());
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
  case matrixmarket:
    file = new MTXFile(filename);
    break;
  case octave:
    file = new OctaveFile(filename);
    break;
  case opendx:
    file = new OpenDXFile(filename);
    break;
  case gid:
    file = new GiDFile(filename);
    break;
  case vtk:
    file = new VTKFile(filename);
    break;
  case python:
    file = new PythonFile(filename);
    break;
  default:
    file = 0;
    dolfin_error1("Unknown file type for \"%s\".", filename.c_str());
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
void File::operator>>(Vector& x)
{
  file->read();
  
  *file >> x;
}
//-----------------------------------------------------------------------------
void File::operator>>(Matrix& A)
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
void File::operator>>(NewMesh& mesh)
{
  file->read();
  
  *file >> mesh;
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
void File::operator<<(Vector& x)
{
  file->write();
  
  *file << x;
}
//-----------------------------------------------------------------------------
void File::operator<<(Matrix& A)
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
void File::operator<<(NewMesh& mesh)
{
  file->write();
  
  *file << mesh;
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
