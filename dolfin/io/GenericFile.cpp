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
// Modified by Niclas Jansson, 2008.
// Modified by Ola Skavhaug 2009.
// Modified by Garth N. Wells 2009.
//
// First added:  2002-11-12
// Last changed: 2011-09-13

#include <fstream>
#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "GenericFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(std::string filename,
                         std::string filetype)
 : filename(filename),
   filetype("<unknown>"),
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
void GenericFile::operator>> (MeshFunction<unsigned int>& mesh_function)
{
  read_not_impl("MeshFunction<unsigned int>");
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
void GenericFile::operator>> (MeshValueCollection<unsigned int>& mesh_markers)
{
  read_not_impl("MeshValueCollection<unsigned int>");
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
void GenericFile::operator>> (FunctionPlotData& data)
{
  read_not_impl("FunctionPlotData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<int>& x)
{
  read_not_impl("std::vector<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<uint>& x)
{
  read_not_impl("std::vector<uint>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::vector<double>& x)
{
  read_not_impl("std::vector<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, int>& map)
{
  read_not_impl("std::map<uint, int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, uint>& map)
{
  read_not_impl("std::map<uint, uint>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, double>& map)
{
  read_not_impl("std::map<uint, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, std::vector<int> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<int> >");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, std::vector<uint> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<uint> >");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (std::map<uint, std::vector<double> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<double> >");
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
void GenericFile::operator<< (const LocalMeshData& data)
{
  write_not_impl("LocalMeshData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<int>& mesh_function)
{
  write_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<unsigned int>& mesh_function)
{
  write_not_impl("MeshFunction<unsigned int>");
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
void GenericFile::operator<< (const MeshValueCollection<unsigned int>& mesh_markers)
{
  write_not_impl("MeshValueCollection<unsigned int>");
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
void GenericFile::operator<< (const std::pair<const Function*, double> u)
{
  write_not_impl("std::pair<Function*, double> Function");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const Parameters& parameters)
{
  write_not_impl("Parameters");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const FunctionPlotData& data)
{
  write_not_impl("FunctionPlotData");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<int>& x)
{
  read_not_impl("std::vector<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<uint>& x)
{
  read_not_impl("std::vector<uint>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::vector<double>& x)
{
  read_not_impl("std::vector<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint, int>& map)
{
  read_not_impl("std::map<uint, int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint, uint>& map)
{
  read_not_impl("std::map<uint, uint>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint, double>& map)
{
  read_not_impl("std::map<uint, double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint,
                              std::vector<int> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<int> >");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint,
                              std::vector<uint> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<uint> >");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const std::map<uint,
                              std::vector<double> >& array_map)
{
  read_not_impl("std::map<uint, std::vector<double> >");
}
//-----------------------------------------------------------------------------
void GenericFile::read()
{
  opened_read = true;
}
//-----------------------------------------------------------------------------
void GenericFile::write()
{
  // pvd files should only be cleared by one process
  if (filetype == "VTK" && MPI::process_number() > 0)
    opened_write = true;

  // Open file
  if (!opened_write)
  {
    // Clear file
    std::ofstream file(filename.c_str(), std::ios::trunc);
    if (!file.good())
      error("Unable to open file %s", filename.c_str());
    file.close();
  }
  opened_write = true;
}
//-----------------------------------------------------------------------------
void GenericFile::read_not_impl(const std::string object) const
{
  error("Unable to read objects of type %s from %s files.",
        object.c_str(), filetype.c_str());
}
//-----------------------------------------------------------------------------
void GenericFile::write_not_impl(const std::string object) const
{
  error("Unable to write objects of type %s to file of type %s.",
        object.c_str(), filetype.c_str());
}
//-----------------------------------------------------------------------------
