// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg
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
// Last changed: 2010-01-04

#include <fstream>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "GenericFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFile::GenericFile(const std::string filename) : filename(filename),
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
void GenericFile::operator>> (MeshFunction<int>& meshfunction)
{
  read_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<unsigned int>& meshfunction)
{
  read_not_impl("MeshFunction<unsigned int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<double>& meshfunction)
{
  read_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (MeshFunction<bool>& meshfunction)
{
  read_not_impl("MeshFunction<bool>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator>> (Sample& sample)
{
  read_not_impl("Sample");
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
void GenericFile::operator<< (const MeshFunction<int>& meshfunction)
{
  write_not_impl("MeshFunction<int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<unsigned int>& meshfunction)
{
  write_not_impl("MeshFunction<unsigned int>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<double>& meshfunction)
{
  write_not_impl("MeshFunction<double>");
}
//-----------------------------------------------------------------------------
void GenericFile::operator<< (const MeshFunction<bool>& meshfunction)
{
  write_not_impl("MeshFunction<bool>");
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
void GenericFile::operator<< (const Sample& sample)
{
  write_not_impl("Sample");
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
  if (type == "VTK" && MPI::process_number() > 0)
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
