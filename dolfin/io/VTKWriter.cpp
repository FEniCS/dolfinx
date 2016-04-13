// Copyright (C) 2010-2016 Garth N. Wells
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
// Modified by Anders Logg 2011
// Modified by Johannes Ring 2012

#include <cstdint>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <boost/detail/endian.hpp>

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include "Encoder.h"
#include "VTKWriter.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void VTKWriter::write_mesh(const Mesh& mesh, std::size_t cell_dim,
                           std::string filename, bool binary, bool compress)
{
  if (binary)
    write_base64_mesh(mesh, cell_dim, filename, compress);
  else
    write_ascii_mesh(mesh, cell_dim, filename);
}
//----------------------------------------------------------------------------
void VTKWriter::write_cell_data(const Function& u, std::string filename,
                                bool binary, bool compress)
{
  // For brevity
  dolfin_assert(u.function_space()->mesh());
  dolfin_assert(u.function_space()->dofmap());
  const Mesh& mesh = *u.function_space()->mesh();
  const GenericDofMap& dofmap = *u.function_space()->dofmap();
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().ghost_offset(tdim);

  std::string encode_string;
  if (!binary)
    encode_string = "ascii";
  else
    encode_string = "binary";

  // Get rank of Function
  const std::size_t rank = u.value_rank();
  if(rank > 2)
  {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
  }

  // Get number of components
  const std::size_t data_dim = u.value_size();

  // Open file
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Write headers
  if (rank == 0)
  {
    fp << "<CellData  Scalars=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  format=\"" << encode_string <<"\">";
  }
  else if (rank == 1)
  {
    if(!(data_dim == 2 || data_dim == 3))
    {
      dolfin_error("VTKWriter.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
    }
    fp << "<CellData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">";
  }
  else if (rank == 2)
  {
    if(!(data_dim == 4 || data_dim == 9))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle tensor function with dimension other than 4 or 9");
    }
    fp << "<CellData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">";
  }

  // Allocate memory for function values at cell centres
  const std::size_t size = num_cells*data_dim;

  // Build lists of dofs and create map
  std::vector<dolfin::la_index> dof_set;
  std::vector<std::size_t> offset(size + 1);
  std::vector<std::size_t>::iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Tabulate dofs
    const ArrayView<const dolfin::la_index>
      dofs = dofmap.cell_dofs(cell->index());
    for(std::size_t i = 0; i < dofmap.num_element_dofs(cell->index()); ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1)
      = *(cell_offset) + dofmap.num_element_dofs(cell->index());
    ++cell_offset;
  }

  // Get  values
  std::vector<double> values(dof_set.size());
  dolfin_assert(u.vector());
  u.vector()->get_local(values.data(), dof_set.size(), dof_set.data());

  // Get cell data
  if (!binary)
    fp << ascii_cell_data(mesh, offset, values, data_dim, rank);
  else
  {
    fp << base64_cell_data(mesh, offset, values, data_dim, rank, compress)
       << std::endl;
  }
  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------
std::string VTKWriter::ascii_cell_data(const Mesh& mesh,
                                       const std::vector<std::size_t>& offset,
                                       const std::vector<double>& values,
                                       std::size_t data_dim, std::size_t rank)
{
  std::ostringstream ss;
  ss << std::scientific;
  ss << std::setprecision(16);
  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (rank == 1 && data_dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      ss << values[*cell_offset] << "  " << values[*cell_offset + 1] << " "
         << 0.0;
    }
    else if (rank == 2 && data_dim == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for(std::size_t i = 0; i < 2; i++)
      {
        ss << values[*cell_offset + 2*i] << " ";
        ss << values[*cell_offset + 2*i + 1] << " ";
        ss << 0.0 << " ";
      }
      ss << 0.0 << " ";
      ss << 0.0 << " ";
      ss << 0.0;
    }
    else
    {
      // Write all components
      for (std::size_t i = 0; i < data_dim; i++)
        ss << values[*cell_offset + i] << " ";
    }
    ss << "  ";
    ++cell_offset;
  }

  return ss.str();
}
//----------------------------------------------------------------------------
std::string VTKWriter::base64_cell_data(const Mesh& mesh,
                                        const std::vector<std::size_t>& offset,
                                        const std::vector<double>& values,
                                        std::size_t data_dim, std::size_t rank,
                                        bool compress)
{
  const std::size_t num_cells = mesh.num_cells();

  // Number of zero paddings per point
  std::size_t padding_per_point = 0;
  if (rank == 1 && data_dim == 2)
    padding_per_point = 1;
  else if (rank == 2 && data_dim == 4)
    padding_per_point = 5;

  // Number of data entries per point and total number
  const std::size_t num_data_per_point = data_dim + padding_per_point;
  const std::size_t num_total_data_points = num_cells*num_data_per_point;

  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  std::vector<double> data(num_total_data_points, 0);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t index = cell->index();
    for(std::size_t i = 0; i < data_dim; i++)
      data[index*num_data_per_point + i] = values[*cell_offset + i];
    ++cell_offset;
  }

  return encode_stream(data, compress);
}
//----------------------------------------------------------------------------
void VTKWriter::write_ascii_mesh(const Mesh& mesh, std::size_t cell_dim,
                                 std::string filename)
{
  const std::size_t num_cells = mesh.topology().ghost_offset(cell_dim);
  const std::size_t num_cell_vertices = mesh.type().num_vertices(cell_dim);

  // Get VTK cell type
  const std::size_t _vtk_cell_type = vtk_cell_type(mesh, cell_dim);

  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  file.precision(16);
  if (!file.is_open())
  {
    dolfin_error("VTKWriter.cpp",
                 "write mesh to VTK file"
                 "Unable to open file \"%s\"", filename.c_str());
  }

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\""
       << "ascii" << "\">";
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    file << p.x() << " " << p.y() << " " <<  p.z() << "  ";
  }
  file << "</DataArray>" << std::endl <<  "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\""
       << "ascii" << "\">";

  std::unique_ptr<CellType>
    celltype(CellType::create(mesh.type().entity_type(cell_dim)));
  const std::vector<unsigned int> perm = celltype->vtk_mapping();
  for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
  {
    for (unsigned int i = 0; i != c->num_entities(0); ++i)
      file << c->entities(0)[perm[i]] << " ";
    file << " ";
  }
  file << "</DataArray>" << std::endl;

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\"" << "ascii"
       << "\">";
  for (std::size_t offsets = 1; offsets <= num_cells; offsets++)
    file << offsets*num_cell_vertices << " ";
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"" << "ascii"
       << "\">";
  for (std::size_t types = 0; types < num_cells; types++)
    file << _vtk_cell_type << " ";
  file  << "</DataArray>" << std::endl;
  file  << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//-----------------------------------------------------------------------------
void VTKWriter::write_base64_mesh(const Mesh& mesh, std::size_t cell_dim,
                                  std::string filename, bool compress)
{
  const std::size_t num_cells = mesh.topology().size(cell_dim);
  const std::size_t num_cell_vertices = mesh.type().num_vertices(cell_dim);

  // Get VTK cell type
  const std::uint8_t _vtk_cell_type = vtk_cell_type(mesh, cell_dim);

  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  file.precision(16);
  if ( !file.is_open() )
  {
    dolfin_error("VTKWriter.cpp",
                 "write mesh to VTK file",
                 "Unable to open file \"%s\"", filename.c_str());
  }

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\""
       << "binary" << "\">" << std::endl;
  std::vector<double> vertex_data(3*mesh.num_vertices());
  std::vector<double>::iterator vertex_entry = vertex_data.begin();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    *vertex_entry++ = p.x();
    *vertex_entry++ = p.y();
    *vertex_entry++ = p.z();
  }
  // Create encoded stream
  file <<  encode_stream(vertex_data, compress) << std::endl;
  file << "</DataArray>" << std::endl <<  "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\""
       << "binary" << "\">" << std::endl;
  const int size = num_cells*num_cell_vertices;
  std::vector<std::uint32_t> cell_data(size);
  std::vector<std::uint32_t>::iterator cell_entry = cell_data.begin();

  std::unique_ptr<CellType>
    celltype(CellType::create(mesh.type().entity_type(cell_dim)));
  const std::vector<unsigned int> perm = celltype->vtk_mapping();
  for (MeshEntityIterator c(mesh, cell_dim); !c.end(); ++c)
  {
    for (unsigned int i = 0; i != c->num_entities(0); ++i)
      *cell_entry++ = c->entities(0)[perm[i]];
  }

  // Create encoded stream
  file << encode_stream(cell_data, compress) << std::endl;
  file << "</DataArray>" << std::endl;

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\""
       << "binary" << "\">" << std::endl;
  std::vector<std::uint32_t> offset_data(num_cells*num_cell_vertices);
  std::vector<std::uint32_t>::iterator offset_entry = offset_data.begin();
  for (std::size_t offsets = 1; offsets <= num_cells; offsets++)
    *offset_entry++ = offsets*num_cell_vertices;

  // Create encoded stream
  file << encode_stream(offset_data, compress) << std::endl;
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"" << "binary"
       << "\">" << std::endl;
  std::vector<std::uint8_t> type_data(num_cells);
  std::vector<std::uint8_t>::iterator type_entry = type_data.begin();
  for (std::size_t types = 0; types < num_cells; types++)
    *type_entry++ = _vtk_cell_type;

  // Create encoded stream
  file << encode_stream(type_data, compress) << std::endl;

  file  << "</DataArray>" << std::endl;
  file  << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//----------------------------------------------------------------------------
std::uint8_t VTKWriter::vtk_cell_type(const Mesh& mesh,
                                      std::size_t cell_dim)
{
  // Get cell type
  CellType::Type cell_type = mesh.type().entity_type(cell_dim);

  // Determine VTK cell type
  std::uint8_t vtk_cell_type = 0;
  if (cell_type == CellType::tetrahedron)
    vtk_cell_type = 10;
  else if (cell_type == CellType::hexahedron)
    vtk_cell_type = 12;
  else if (cell_type == CellType::quadrilateral)
    vtk_cell_type = 9;
  else if (cell_type == CellType::triangle)
    vtk_cell_type = 5;
  else if (cell_type == CellType::interval)
    vtk_cell_type = 3;
  else if (cell_type == CellType::point)
    vtk_cell_type = 1;
  else
  {
    dolfin_error("VTKWriter.cpp",
                 "write data to VTK file",
                 "Unknown cell type (%d)", cell_type);
  }

  return vtk_cell_type;
}
//----------------------------------------------------------------------------
