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
void GenericFile::operator>> (Mesh& mesh)
{
  read_not_impl("Mesh");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (GenericVector& x)
{
  read_not_impl("Vector");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (GenericMatrix& A)
{
  read_not_impl("Matrix");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (GenericDofMap& data)
{
  read_not_impl("GenericDofMap");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (LocalMeshData& data)
{
  read_not_impl("LocalMeshData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<int>& mesh_function)
{
  read_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<std::size_t>& mesh_function)
{
  read_not_impl("MeshFunction<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<double>& mesh_function)
{
  read_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<bool>& mesh_function)
{
  read_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshValueCollection<int>& mesh_markers)
{
  read_not_impl("MeshValueCollection<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshValueCollection<std::size_t>& mesh_markers)
{
  read_not_impl("MeshValueCollection<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshValueCollection<double>& mesh_markers)
{
  read_not_impl("MeshValueCollection<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshValueCollection<bool>& mesh_markers)
{
  read_not_impl("MeshValueCollection<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (Parameters& parameters)
{
  read_not_impl("Parameters");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (Table& table)
{
  read_not_impl("Table");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<int>& x)
{
  read_not_impl("std::vector<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<std::size_t>& x)
{
  read_not_impl("std::vector<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<double>& x)
{
  read_not_impl("std::vector<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t, int>& map)
{
  read_not_impl("std::map<std::size_t, int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t, std::size_t>& map)
{
  read_not_impl("std::map<std::size_t, std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t, double>& map)
{
  read_not_impl("std::map<std::size_t, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t,
                              std::vector<int>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<int>>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t,
                              std::vector<std::size_t>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<std::size_t>>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<std::size_t,
                              std::vector<double>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<double>>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (Function& u)
{
  read_not_impl("Function");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const Mesh & mesh)
{
  write_not_impl("Mesh");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const GenericVector& x)
{
  write_not_impl("Vector");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const GenericMatrix& A)
{
  write_not_impl("Matrix");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const GenericDofMap& dofmap)
{
  write_not_impl("GenericDofMap");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const LocalMeshData& data)
{
  write_not_impl("LocalMeshData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<int>& mesh_function)
{
  write_not_impl("MeshFunction<int>");
}
void GenericFile::operator<< (const MeshFunction<std::size_t>& mesh_function)
{
  write_not_impl("MeshFunction<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<double>& mesh_function)
{
  write_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<bool>& mesh_function)
{
  write_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshValueCollection<int>& mesh_markers)
{
  write_not_impl("MeshValueCollection<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshValueCollection<std::size_t>& mesh_markers)
{
  write_not_impl("MeshValueCollection<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshValueCollection<double>& mesh_markers)
{
  write_not_impl("MeshValueCollection<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshValueCollection<bool>& mesh_markers)
{
  write_not_impl("MeshValueCollection<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const Function& u)
{
  write_not_impl("Function");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const Mesh*, double> mesh)
{
  write_not_impl("std::pair<Mesh*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const MeshFunction<int>*, double> f)
{
  write_not_impl("std::pair<MeshFunction<int>*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const MeshFunction<std::size_t>*, double> f)
{
  write_not_impl("std::pair<MeshFunction<std::size_t>*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const MeshFunction<double>*, double> f)
{
  write_not_impl("std::pair<MeshFunction<double>*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const MeshFunction<bool>*, double> f)
{
  write_not_impl("std::pair<MeshFunction<bool>*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::pair<const Function*, double> u)
{
  write_not_impl("std::pair<Function*, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const Parameters& parameters)
{
  write_not_impl("Parameters");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const Table& table)
{
  write_not_impl("Table");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<int>& x)
{
  read_not_impl("std::vector<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<std::size_t>& x)
{
  read_not_impl("std::vector<std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<double>& x)
{
  read_not_impl("std::vector<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t, int>& map)
{
  read_not_impl("std::map<std::size_t, int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t, std::size_t>& map)
{
  read_not_impl("std::map<std::size_t, std::size_t>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t, double>& map)
{
  read_not_impl("std::map<std::size_t, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t,
                              std::vector<int>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<int>>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t,
                              std::vector<std::size_t>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<std::size_t>>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<std::size_t,
                              std::vector<double>>& array_map)
{
  read_not_impl("std::map<std::size_t, std::vector<double>>");
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
