// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2010-02-10

#include <dolfin/log/log.h>
#include <istream>
#include <fstream>
#include <ios>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/la/GenericVector.h>
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
  // Read size
  std::ifstream file(filename.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open())
    error("Unable to open file \"%s\".", filename.c_str());
  uint n;
  file.read((char*) &n, sizeof(uint));

  dolfin_debug2("Reading %d vector values in binary from %s", n, filename.c_str());

  // Read vector values
  values.resize(n);
  for (uint i = 0; i < n; ++i)
    file.read((char*) &values[i], sizeof(double));
  file.close();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (GenericVector& vector)
{
  // Read size
  std::ifstream file(filename.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open())
    error("Unable to open file \"%s\".", filename.c_str());
  uint n;
  file.read((char*) &n, sizeof(uint));

  dolfin_debug2("Reading %d vector values in binary from %s", n, filename.c_str());

  // Read vector values
  boost::scoped_array<double> values(new double[n]);
  for (uint i = 0; i < n; ++i)
    file.read((char*) &values[i], sizeof(double));
  file.close();

  // Set vector values
  vector.resize(n);
  vector.set_local(values.get());
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (Mesh& mesh)
{
  warning("Reading mesh in binary format not implemented.");
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const std::vector<double>& values)
{
  // Get size
  const uint n = values.size();

  dolfin_debug2("Writing %d array value(s) in binary to %s", n, filename.c_str());

  // Write size and values
  std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open())
    error("Unable to open file \"%s\".", filename.c_str());
  file.write((char*) &n, sizeof(uint));
  for (uint i = 0; i < n; ++i)
    file.write((char*) &values[i], sizeof(double));
  file.close();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const GenericVector& vector)
{
  // Get size and vector values
  const uint n = vector.size();
  boost::scoped_array<double> values(new double[n]);
  vector.get_local(values.get());

  dolfin_debug2("Writing %d vector value(s) in binary to %s", n, filename.c_str());

  // Write size and values
  std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open())
    error("Unable to open file \"%s\".", filename.c_str());
  file.write((char*) &n, sizeof(uint));
  for (uint i = 0; i < n; ++i)
    file.write((char*) &values[i], sizeof(double));
  file.close();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const Mesh& mesh)
{
  warning("Writing mesh in binary format not implemented.");
}
//-----------------------------------------------------------------------------
