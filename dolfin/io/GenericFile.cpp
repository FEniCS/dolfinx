// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2002-11-12
// Last changed: 2008-09-16

// FIXME: Use streams instead of stdio
#include <stdio.h>

#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "GenericFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename) :
  filename(filename), 
  type("Unknown file type"),
  opened_read(false),
  opened_write(false),
  check_header(false),
  counter(0),
  counter1(0),
  counter2(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(GenericVector& x)
{
  read_not_impl("Vector");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(GenericMatrix& A)
{
  read_not_impl("Matrix");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(Mesh& mesh)
{
  read_not_impl("Mesh");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(MeshFunction<int>& meshfunction)
{
  read_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(MeshFunction<unsigned int>& meshfunction)
{
  read_not_impl("MeshFunction<unsigned int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(MeshFunction<double>& meshfunction)
{
  read_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(MeshFunction<bool>& meshfunction)
{
  read_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(Function& f)
{
  read_not_impl("Function");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(Sample& sample)
{
  read_not_impl("Sample");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(FiniteElementSpec& spec)
{
  read_not_impl("FiniteElementSpec");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(ParameterList& parameters)
{
  read_not_impl("ParameterList");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(BLASFormData& blas)
{
  read_not_impl("BLASFormData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>>(Graph& graph)
{
  read_not_impl("Graph");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(GenericVector& x)
{
  write_not_impl("Vector");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(GenericMatrix& A)
{
  write_not_impl("Matrix");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(Mesh& mesh)
{
  write_not_impl("Mesh");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(MeshFunction<int>& meshfunction)
{
  write_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(MeshFunction<unsigned int>& meshfunction)
{
  write_not_impl("MeshFunction<unsigned int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(MeshFunction<double>& meshfunction)
{
  write_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(MeshFunction<bool>& meshfunction)
{
  write_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(Function& u)
{
  write_not_impl("Function");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(Sample& sample)
{
  write_not_impl("Sample");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<<(FiniteElementSpec& spec)
{
  write_not_impl("FiniteElementSpec");
}
//-----------------------------------------------------------------------------
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
void GenericFile::operator<<(Graph& graph)
{
  write_not_impl("Graph");
}
//-----------------------------------------------------------------------------
void GenericFile::read()
{
  opened_read = true;
}
//-----------------------------------------------------------------------------
void GenericFile::write()
{

  //FIXME .pvd files should only be cleared by one processor
  if ( type == "VTK" && MPI::processNumber() > 0)
    opened_write = true;

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
  error("Unable to read objects of type %s from %s files.",
		object.c_str(), type.c_str());
}
//-----------------------------------------------------------------------------
void GenericFile::write_not_impl(const std::string object)
{
  error("Unable to write objects of type %s to %s files.",
		object.c_str(), type.c_str());
}
//-----------------------------------------------------------------------------
