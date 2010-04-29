// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2010-04-29

#include <dolfin/log/log.h>
#include <istream>
#include <fstream>
#include <ios>
#include <boost/scoped_array.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include "BinaryFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BinaryFile::BinaryFile(const std::string filename) : GenericFile(filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BinaryFile::~BinaryFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (std::vector<double>& values)
{
  open_read();

  uint n = read_uint();
  values.resize(n);
  read_array(n, &values[0]);

  close_read();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (GenericVector& vector)
{
  open_read();

  uint n = read_uint();
  Array<double> values(n);
  read_array(n, values);

  vector.resize(n);
  vector.set_local(values);

  close_read();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (Mesh& mesh)
{
  warning("Reading mesh in binary format not implemented.");
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const std::vector<double>& values)
{
  open_write();

  write_uint(values.size());
  write_array(values.size(), &values[0]);

  close_write();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const GenericVector& vector)
{
  open_write();

  uint n = vector.size();
  Array<double> values(n);
  vector.get_local(values);
  write_uint(n);
  write_array(n, values);

  close_write();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const Mesh& mesh)
{
  warning("Writing mesh in binary format not implemented.");
}
//-----------------------------------------------------------------------------
void BinaryFile::open_read()
{
  ifile.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ifile.is_open())
    error("Unable to open file for reading: \"%s\".", filename.c_str());
}
//-----------------------------------------------------------------------------
void BinaryFile::open_write()
{
  ofile.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofile.is_open())
    error("Unable to open file for writing: \"%s\".", filename.c_str());
}
//-----------------------------------------------------------------------------
void BinaryFile::close_read()
{
  ifile.close();
}
//-----------------------------------------------------------------------------
void BinaryFile::close_write()
{
  ofile.close();
}
//-----------------------------------------------------------------------------
dolfin::uint BinaryFile::read_uint()
{
  uint value = 0;
  ifile.read((char*) &value, sizeof(uint));
  return value;
}
//-----------------------------------------------------------------------------
void BinaryFile::read_array(uint n, uint* values)
{
  for (uint i = 0; i < n; ++i)
    ifile.read((char*) (values + i), sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::read_array(uint n, double* values)
{
  for (uint i = 0; i < n; ++i)
    ifile.read((char*) (values + i), sizeof(double));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_uint(uint value)
{
  ofile.write((char*) &value, sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_array(uint n, const uint* values)
{
  for (uint i = 0; i < n; ++i)
    ofile.write((char*) &values[i], sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_array(uint n, const double* values)
{
  for (uint i = 0; i < n; ++i)
    ofile.write((char*) &values[i], sizeof(double));
}
//-----------------------------------------------------------------------------
