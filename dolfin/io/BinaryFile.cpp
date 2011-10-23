// Copyright (C) 2009-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-11-11
// Last changed: 2011-10-23

#include <fstream>
#include <istream>
#include <ios>
#include <boost/scoped_array.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/operations.hpp>
#include <iosfwd> 

#include <dolfin/common/Array.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include "BinaryFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BinaryFile::BinaryFile(const std::string filename)
  : GenericFile(filename, "Binary")
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

  const uint n = read_uint();
  values.resize(n);
  read_array(n, &values[0]);

  close_read();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (GenericVector& vector)
{
  open_read();

  const uint n = read_uint();
  Array<double> values(n);
  read_array(n, values.data().get());

  vector.resize(n);
  vector.set_local(values);

  close_read();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (Mesh& mesh)
{
  // Open file
  open_read();

  // Clear mesh
  mesh.clear();

  // Read mesh topology
  MeshTopology& t = mesh._topology;
  uint D = read_uint();
  t._dim = D;
  t.num_entities = new uint[D + 1];
  read_array(D + 1, t.num_entities);
  t.connectivity = new MeshConnectivity**[D + 1];
  for (uint i = 0; i <= D; i++)
  {
    t.connectivity[i] = new MeshConnectivity*[D + 1];
    for (uint j = 0; j <= D; j++)
    {
      t.connectivity[i][j] = new MeshConnectivity(i, j);
      MeshConnectivity& c = *t.connectivity[i][j];
      c._size = read_uint();
      if (c._size > 0)
      {
        c.num_entities = read_uint();
        c.connections = new uint[c._size];
        read_array(c._size, c.connections);
        c.offsets = new uint[c.num_entities + 1];
        read_array(c.num_entities + 1, c.offsets);
      }
    }
  }

  // Read mesh geometry (ignoring higher order stuff)
  MeshGeometry& g = mesh._geometry;
  g._dim = read_uint();
  g._size = read_uint();
  g.coordinates = new double[g._dim * g._size];
  read_array(g._dim * g._size, g.coordinates);

  // Read cell type
  mesh._cell_type = CellType::create(static_cast<CellType::Type>(read_uint()));

  // Read mesh data
  // FIXME: Not implemented

  // Read mesh domains
  // FIXME: Not implemented

  // Initialize mesh domains
  mesh._domains.init(D);

  // Close file
  close_read();
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

  const uint n = vector.size();
  Array<double> values(n);
  vector.get_local(values);
  write_uint(n);
  write_array(n, values.data().get());

  close_write();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const Mesh& mesh)
{
  // Open file
  open_write();

  // Write mesh topology
  const MeshTopology& t = mesh._topology;
  const uint D = t._dim;
  write_uint(D);
  write_array(D + 1, t.num_entities);
  for (uint i = 0; i <= D; i++)
  {
    for (uint j = 0; j <= D; j++)
    {
      const MeshConnectivity& c = *t.connectivity[i][j];
      write_uint(c._size);
      if (c._size > 0)
      {
        write_uint(c.num_entities);
        write_array(c._size, c.connections);
        write_array(c.num_entities + 1, c.offsets);
      }
    }
  }

  // Write mesh geometry (ignoring higher order stuff)
  const MeshGeometry& g = mesh._geometry;
  write_uint(g._dim);
  write_uint(g._size);
  write_array(g._dim * g._size, g.coordinates);

  // Write cell type
  write_uint(static_cast<uint>(mesh._cell_type->cell_type()));

  // Write mesh data
  // FIXME: Not implemented

  // Close file
  close_write();
}
//-----------------------------------------------------------------------------
void BinaryFile::open_read()
{
  // Get file path and extension
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  // FIXME: Check that file exists
  if (!boost::filesystem::is_regular_file(filename))
    error("File \"%s\" does not exist or is not a regular file. Cannot be read by XML parser.", filename.c_str());

  // Load xml file (unzip if necessary) into parser
  if (extension == ".gz")
    // Decompress file
    ifilter.push(boost::iostreams::gzip_decompressor());
  
  ifile.open(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifile.is_open())
    error("Unable to open file for reading: \"%s\".", filename.c_str());
  ifilter.push(ifile);
  
}
//-----------------------------------------------------------------------------
void BinaryFile::open_write()
{
  // Compress if filename has extension '.gz'
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  if (extension == ".gz")
    ofilter.push(boost::iostreams::gzip_compressor());
  
  ofile.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofile.is_open())
    error("Unable to open file for writing: \"%s\".", filename.c_str());
  ofilter.push(ofile);
}
//-----------------------------------------------------------------------------
void BinaryFile::close_read()
{
  ifilter.reset();
}
//-----------------------------------------------------------------------------
void BinaryFile::close_write()
{
  ofilter.reset();
}
//-----------------------------------------------------------------------------
dolfin::uint BinaryFile::read_uint()
{
  uint value = 0;
  boost::iostreams::read(ifilter, (char*) &value, (std::streamsize) sizeof(uint));
  return value;
}
//-----------------------------------------------------------------------------
void BinaryFile::read_array(uint n, uint* values)
{
  for (uint i = 0; i < n; ++i)
    boost::iostreams::read(ifilter, (char*) (values + i), (std::streamsize) sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::read_array(uint n, double* values)
{
  for (uint i = 0; i < n; ++i)
    boost::iostreams::read(ifilter, (char*) (values + i), (std::streamsize) sizeof(double));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_uint(uint value)
{
  boost::iostreams::write(ofilter, (char*) &value, (std::streamsize) sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_array(uint n, const uint* values)
{
  for (uint i = 0; i < n; ++i)
    boost::iostreams::write(ofilter, (char*) &values[i], (std::streamsize) sizeof(uint));
}
//-----------------------------------------------------------------------------
void BinaryFile::write_array(uint n, const double* values)
{
  for (uint i = 0; i < n; ++i)
    boost::iostreams::write(ofilter, (char*) &values[i], (std::streamsize) sizeof(double));
}
//-----------------------------------------------------------------------------
