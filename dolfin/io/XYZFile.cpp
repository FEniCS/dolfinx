// Copyright (C) 2005-2007 Garth N. Wells
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
// Modified by Nuno Lopes 2008
// Modified by Anders Logg 2011
//
// First added:  2008-07-02
// Last changed: 2011-11-21

#include <fstream>
#include <sstream>
#include <boost/scoped_array.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "XYZFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XYZFile::XYZFile(const std::string filename)
  : GenericFile(filename, "XYZ")
{
  // Do nothing
}
//----------------------------------------------------------------------------
XYZFile::~XYZFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XYZFile::operator<<(const Function& u)
{
  // Update xyz file name and clear file
  xyz_name_update();

  // Write results
  results_write(u);

  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << _filename << " in xd3d xyz format." << endl;
}
//----------------------------------------------------------------------------
void XYZFile::results_write(const Function& u) const
{
  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);
  if (!fp)
  {
    dolfin_error("XYZFile.cpp",
                 "write function to XYZ file"
                 "Unable to open file \"%s\"", _filename.c_str());
  }

  const std::size_t rank = u.function_space()->element()->value_rank();
  if (rank > 1)
  {
    dolfin_error("XYZFile.cpp",
                 "write function XYZ file",
                 "Only scalar functions can be saved in XYZ format");
  }

  // Get number of components
  std::size_t dim = 1;
  for (std::size_t i = 0; i < rank; i++)
    dim *= u.function_space()->element()->value_dimension(i);

  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Allocate memory for function values at vertices
  //const std::size_t size = mesh.num_vertices()*dim;
  std::vector<double> values;

  // Get function values at vertices
  u.compute_vertex_values(values, mesh);

  // Write function data at mesh vertices
  if (dim > 1)
  {
    dolfin_error("XYZFile.cpp",
                 "write function XYZ file",
                 "Only scalar functions can be saved in XYZ format");
  }

  std::ostringstream ss;
  ss << std::scientific;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    ss.str("");
    for (std::size_t i = 0; i < mesh.geometry().dim(); i++)
      ss << vertex->x(i) << " ";
    ss << values[vertex->index()];
    ss <<std::endl;
    fp << ss.str();
  }
}
//----------------------------------------------------------------------------
void XYZFile::xyz_name_update()
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(_filename, 0, _filename.find_last_of("."));
  extension.assign(_filename, _filename.find_last_of("."), _filename.size());

  fileid << counter;
  newfilename << filestart << fileid.str() << ".xyz";
  xyz_filename = newfilename.str();

  // Make sure file is empty
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::trunc);
  if (!fp)
  {
    dolfin_error("XYZFile.cpp",
                 "write data to XYZ file",
                 "Unable to open file \"%s\"", xyz_filename.c_str());
  }

  // Add to index file
  std::ofstream fp_index(_filename.c_str(), std::ios_base::trunc);
  if (!fp_index)
  {
    dolfin_error("XYZFile.cpp",
                 "write data to XYZ file",
                 "Unable to open file \"%s\"", _filename.c_str());
  }
  else
  {
    // File is cleared in GenericFile::write so we rewrite the whole
    // index every time, easier than messing with a flag to say that
    // the file should not be cleared. This whole file needs some
    // cleanup but we can sort that out later.
    for (std::size_t i = 0; i < counter + 1; i++)
    {
      std::ostringstream fileid0;
      fileid0.fill('0');
      fileid0.width(6);
      fileid0 << i;
      std::ostringstream f;
      f << filestart << fileid0.str() << ".xyz";
      fp_index << f.str() << "\n";
    }
  }
}
//----------------------------------------------------------------------------
template<typename T>
void XYZFile::mesh_function_write(T& meshfunction)
{
  // Update xyz file name and clear file
  xyz_name_update();

  Mesh& mesh = meshfunction.mesh();

  if (meshfunction.dim() != mesh.topology().dim())
  {
    dolfin_error("XYZFile.cpp",
                 "write mesh function to XYZ file",
                 "XYZ output of mesh functions is implemented for cell-based functions only");
  }

  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);
  if (!fp)
  {
    dolfin_error("XYZFile.cpp",
                 "write mesh function to XYZ file",
                 "Unable to open file \"%s\"", _filename.c_str());
  }

  fp << mesh.num_cells() << std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get(cell->index())  << std::endl;

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;
  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << _filename << " in XYZ format." << endl;
}
//----------------------------------------------------------------------------
