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
// Last changed: 2011-11-23

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
BinaryFile::BinaryFile(const std::string filename, bool store_connectivity)
  : GenericFile(filename, "Binary"), _store_connectivity(store_connectivity)
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

  const std::size_t n = read_uint();
  values.resize(n);
  read_array(n, &values[0]);

  close_read();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator>> (GenericVector& vector)
{
  open_read();

  const std::size_t n = read_uint();
  std::vector<double> values(n);
  read_array(n, values.data());

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
  std::size_t D = read_uint();

  std::vector<std::size_t> t_num_entities(D + 1);
  read_array(D + 1, t_num_entities.data());
  //t.num_entities.resize(D + 1);
  //read_array(D + 1, t.num_entities.data());

  t.num_entities = std::vector<unsigned int>(t_num_entities.begin(), t_num_entities.end());

  t.connectivity.resize(D + 1);
  for (std::size_t i = 0; i <= D; i++)
  {
    for (std::size_t j = 0; j <= D; j++)
    {
      t.connectivity[i].push_back(MeshConnectivity(i, j));
      MeshConnectivity& c = t.connectivity[i][j];
      const std::size_t size = read_uint();
      if (size > 0)
      {
        const std::size_t num_entities = read_uint();
        c.connections = std::vector<unsigned int>(size);
        read_array(size, &(c.connections)[0]);
        c.index_to_position.resize(num_entities + 1);
        read_array(c.index_to_position.size(), c.index_to_position.data());
      }
    }
  }

  // Read mesh geometry (ignoring higher order stuff)
  MeshGeometry& g = mesh._geometry;
  g._dim = read_uint();
  const std::size_t size = read_uint();
  g.coordinates.resize(g._dim*size);
  read_array(g._dim*size, g.coordinates.data());

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

  const std::size_t n = vector.size();
  std::vector<double> values(n);
  vector.get_local(values);
  write_uint(n);
  write_array(n, values.data());

  close_write();
}
//-----------------------------------------------------------------------------
void BinaryFile::operator<< (const Mesh& mesh)
{
  // Open file
  open_write();

  // Write mesh topology
  const MeshTopology& t = mesh._topology;
  const std::size_t D = t.dim();
  write_uint(D);
  if (_store_connectivity)
  {
    std::vector<std::size_t> t_num_entities(t.num_entities.begin(), t.num_entities.end());
    write_array(D + 1, t_num_entities.data());
    //write_array(D + 1, t.num_entities.data());
  }
  else
  {
    for (std::size_t i = 0; i <= D; i++)
    {
      if (i==0 || i == D)
        write_uint(t.size(i));
      else
        write_uint(0);
    }
  }
  for (std::size_t i = 0; i <= D; i++)
  {
    for (std::size_t j = 0; j <= D; j++)
    {
      const MeshConnectivity& c = t.connectivity[i][j];

      // If store all connectivity or if storing cell connectivity
      if (_store_connectivity || (i == D && j == 0))
      {
        write_uint(c.size());
        if (!c.empty())
        {
          write_uint(c.index_to_position.size() - 1);
          write_array(c.size(), c.connections.data());
          write_array(c.index_to_position.size(), c.index_to_position.data());
        }
      }
      else
        write_uint(0);
    }
  }

  // Write mesh geometry (ignoring higher order stuff)
  const MeshGeometry& g = mesh._geometry;
  write_uint(g._dim);
  write_uint(g.size());
  write_array(g._dim*g.size(), g.coordinates.data());

  // Write cell type
  write_uint(static_cast<std::size_t>(mesh._cell_type->cell_type()));

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
  {
    dolfin_error("BinaryFile.cpp",
                 "open binary file",
                 "File \"%s\" does not exist or is not a regular file",
                 filename.c_str());
  }

  // Load xml file (unzip if necessary) into parser
  if (extension == ".gz")
    // Decompress file
    ifilter.push(boost::iostreams::gzip_decompressor());

  ifile.open(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifile.is_open())
  {
    dolfin_error("BinaryFile.cpp",
                 "open binary file",
                 "Cannot open file \"%s\" for reading", filename.c_str());
  }
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
  {
    dolfin_error("BinaryFile.cpp",
                 "open binary file",
                 "Cannot open file \"%s\" for writing", filename.c_str());
  }
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
std::size_t BinaryFile::read_uint()
{
  std::size_t value = 0;
  boost::iostreams::read(ifilter, (char*) &value, (std::streamsize) sizeof(std::size_t));
  return value;
}
//-----------------------------------------------------------------------------
template <typename T>
void BinaryFile::read_array(std::size_t n, T* values)
{
  for (std::size_t i = 0; i < n; ++i)
    boost::iostreams::read(ifilter, (char*) (values + i), (std::streamsize) sizeof(T));
}
//-----------------------------------------------------------------------------
/*
void BinaryFile::read_array(std::size_t n, double* values)
{
  for (std::size_t i = 0; i < n; ++i)
    boost::iostreams::read(ifilter, (char*) (values + i), (std::streamsize) sizeof(double));
}
*/
//-----------------------------------------------------------------------------
void BinaryFile::write_uint(std::size_t value)
{
  boost::iostreams::write(ofilter, (char*) &value, (std::streamsize) sizeof(std::size_t));
}
//-----------------------------------------------------------------------------
template <typename T>
void BinaryFile::write_array(std::size_t n, const T* values)
{
  for (std::size_t i = 0; i < n; ++i)
    boost::iostreams::write(ofilter, (char*) &values[i], (std::streamsize) sizeof(T));
}
//-----------------------------------------------------------------------------
/*
void BinaryFile::write_array(std::size_t n, const double* values)
{
  for (std::size_t i = 0; i < n; ++i)
    boost::iostreams::write(ofilter, (char*) &values[i], (std::streamsize) sizeof(double));
}
*/
//-----------------------------------------------------------------------------
