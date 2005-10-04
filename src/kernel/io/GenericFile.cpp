// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-11-12
// Last changed: 2005-10-03

// FIXME: Use streams instead of stdio
#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename) :
  filename(filename), 
  type("Unknown file type"),
  opened_read(false),
  opened_write(false),
  check_header(false),
  no_meshes(0),
  no_frames(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(Vector& x)
{
  read_not_impl("Vector");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(Matrix& A)
{
  read_not_impl("Matrix");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(Mesh& mesh)
{
  read_not_impl("Mesh");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(Function& u)
{
  read_not_impl("Function");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(Sample& sample)
{
  read_not_impl("Sample");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(ParameterList& parameters)
{
  read_not_impl("ParameterList");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator>>(BLASFormData& blas)
{
  read_not_impl("BLASFormData");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(Vector& x)
{
  write_not_impl("Vector");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(Matrix& A)
{
  write_not_impl("Matrix");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(Mesh& mesh)
{
  write_not_impl("Mesh");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(Function& u)
{
  write_not_impl("Function");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(Sample& sample)
{
  write_not_impl("Sample");
}
//-­---------------------------------------------------------------------------
void GenericFile::operator<<(ParameterList& parameters)
{
  write_not_impl("ParameterList");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(BLASFormData& blas)
{
  write_not_impl("BLASFormData");
}
//-----------------------------------------------------------------------------
void GenericFile::read()
{
  opened_read = true;
}
//-----------------------------------------------------------------------------
void GenericFile::write()
{
  if ( !opened_write ) {
    // Clear file
    FILE* fp = fopen(filename.c_str(), "w");
    fclose(fp);
  }
  
  opened_write = true;
}
//-----------------------------------------------------------------------------
void GenericFile::read_not_impl(const std::string object)
{
  dolfin_error2("Unable to read objects of type %s from %s files.",
		object.c_str(), type.c_str());
}
//-----------------------------------------------------------------------------
void GenericFile::write_not_impl(const std::string object)
{
  dolfin_error2("Unable to write objects of type %s to %s files.",
		object.c_str(), type.c_str());
}
//-----------------------------------------------------------------------------
