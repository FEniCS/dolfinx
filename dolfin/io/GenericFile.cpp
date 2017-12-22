// Copyright (C) 2002-2011 Johan Hoffman and Anders Logg
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
// Modified by Niclas Jansson 2008
// Modified by Ola Skavhaug 2009
// Modified by Garth N. Wells 2009
//
// First added:  2002-11-12
// Last changed: 2013-03-11

#include <fstream>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "GenericFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(std::string filename, std::string filetype)
 : _filename(filename), _filetype(filetype),
   opened_read(false),  opened_write(false), check_header(false),
   counter(0), counter1(0), counter2(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFile::~GenericFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericFile::write(const Mesh & mesh)
{
  write_not_impl("Mesh");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<int>& mesh_function)
{
  write_not_impl("MeshFunction<int>");
}
void GenericFile::write(const MeshFunction<std::size_t>& mesh_function)
{
  write_not_impl("MeshFunction<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<double>& mesh_function)
{
  write_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<bool>& mesh_function)
{
  write_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshValueCollection<int>& mesh_markers)
{
  write_not_impl("MeshValueCollection<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshValueCollection<std::size_t>& mesh_markers)
{
  write_not_impl("MeshValueCollection<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshValueCollection<double>& mesh_markers)
{
  write_not_impl("MeshValueCollection<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshValueCollection<bool>& mesh_markers)
{
  write_not_impl("MeshValueCollection<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const Function& u)
{
  write_not_impl("Function");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const Mesh& mesh, double time)
{
  write_not_impl("Mesh, time");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<int>& mf, double time)
{
  write_not_impl("MeshFunction<int>, time");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<std::size_t>& mf, double time)
{
  write_not_impl("MeshFunction<std::size_t>, time");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<double>& mf, double time)
{
  write_not_impl("MeshFunction<double>, time");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const MeshFunction<bool>& mf, double time)
{
  write_not_impl("MeshFunction<bool>, time");
}
//-----------------------------------------------------------------------------
void GenericFile::write(const Function& u, double time)
{
  write_not_impl("Function, time");
}
//-----------------------------------------------------------------------------
void GenericFile::_read()
{
  opened_read = true;
}
//-----------------------------------------------------------------------------
void GenericFile::_write(std::size_t process_number)
{
  // pvd files should only be cleared by one process
  if (_filetype == "VTK" && process_number > 0)
    opened_write = true;

  // Open file
  if (!opened_write)
  {
    // Clear file
    std::ofstream file(_filename.c_str(), std::ios::trunc);
    if (!file.good())
    {
      dolfin_error("GenericFile.cpp",
                   "write to file",
                   "Unable to open file \"%s\" for writing",
                   _filename.c_str());
    }
    file.close();
  }
  opened_write = true;
}
//-----------------------------------------------------------------------------
void GenericFile::read_not_impl(const std::string object) const
{
  dolfin_error("GenericFile.cpp",
               "read object from file",
               "Cannot read objects of type %s from %s files",
               object.c_str(), _filetype.c_str());
}
//-----------------------------------------------------------------------------
void GenericFile::write_not_impl(const std::string object) const
{
  dolfin_error("GenericFile.cpp",
               "write object to file",
               "Cannot write objects of type %s to %s files",
               object.c_str(), _filetype.c_str());
}
//-----------------------------------------------------------------------------
