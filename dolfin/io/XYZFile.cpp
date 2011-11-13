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
// Modified by Nuno Lopes 2008.
//
// First added:  2008-07-02

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
  xyz_name_update(counter);

  // Write results
  results_write(u);

  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in xd3d xyz format." << endl;
}
//----------------------------------------------------------------------------
void XYZFile::results_write(const Function& u) const
{
  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  assert(u.function_space()->element());
  const uint rank = u.function_space()->element()->value_rank();
  if(rank > 1)
    error("Only scalar functions can be saved in xyz format.");

  // Get number of components
  uint dim = 1;
  for (uint i = 0; i < rank; i++)
    dim *= u.function_space()->element()->value_dimension(i);

  assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Allocate memory for function values at vertices
  const uint size = mesh.num_vertices()*dim;
  Array<double> values(size);

  // Get function values at vertices
  u.compute_vertex_values(values, mesh);

  // Write function data at mesh vertices
  if ( dim > 1 )
    error("Cannot handle XYZ file for non-scalar functions. ");
  std::ostringstream ss;
  ss << std::scientific;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    ss.str("");
    ss << vertex->x(0) << " " << vertex->x(1) << " " << values[ vertex->index() ];
    ss <<std::endl;
    fp << ss.str( );
  }
}
//----------------------------------------------------------------------------
void XYZFile::xyz_name_update(int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());

  fileid << counter;
  newfilename << filestart << fileid.str() << ".xyz";

  xyz_filename = newfilename.str();

  // Make sure file is empty
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::trunc);
  if (!fp)
    error("Unable to open file %s", filename.c_str());
}
//----------------------------------------------------------------------------
template<typename T>
void XYZFile::mesh_function_write(T& meshfunction)
{
  // Update xyz file name and clear file
  xyz_name_update(counter);

  Mesh& mesh = meshfunction.mesh();

  if( meshfunction.dim() != mesh.topology().dim() )
    error("XYZ output of mesh functions is implemenetd for cell-based functions only.");

  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  fp << mesh.num_cells() <<std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get( cell->index() )  << std::endl;

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;
  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in XYZ format." << endl;
}
//----------------------------------------------------------------------------
