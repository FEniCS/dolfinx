// Copyright (C) 2010-2016 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKWriter.h"
#include <boost/detail/endian.hpp>
#include <cstdint>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

using namespace dolfin;
using namespace dolfin::io;

//----------------------------------------------------------------------------
void VTKWriter::write_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
                           std::string filename)
{
  write_ascii_mesh(mesh, cell_dim, filename);
}
//----------------------------------------------------------------------------
void VTKWriter::write_cell_data(const function::Function& u,
                                std::string filename)
{
  // For brevity
  assert(u.function_space()->mesh());
  assert(u.function_space()->dofmap());
  const mesh::Mesh& mesh = *u.function_space()->mesh();
  const fem::GenericDofMap& dofmap = *u.function_space()->dofmap();
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().ghost_offset(tdim);

  std::string encode_string = "ascii";

  // Get rank of function::Function
  const std::size_t rank = u.value_rank();
  if (rank > 2)
  {
    log::dolfin_error("VTKFile.cpp", "write data to VTK file",
                 "Don't know how to handle vector function with dimension "
                 "other than 2 or 3");
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
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  format=\""
       << encode_string << "\">";
  }
  else if (rank == 1)
  {
    if (!(data_dim == 2 || data_dim == 3))
    {
      log::dolfin_error("VTKWriter.cpp", "write data to VTK file",
                   "Don't know how to handle vector function with dimension "
                   "other than 2 or 3");
    }
    fp << "<CellData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"3\" format=\"" << encode_string << "\">";
  }
  else if (rank == 2)
  {
    if (!(data_dim == 4 || data_dim == 9))
    {
      log::dolfin_error("VTKFile.cpp", "write data to VTK file",
                   "Don't know how to handle tensor function with dimension "
                   "other than 4 or 9");
    }
    fp << "<CellData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"9\" format=\"" << encode_string << "\">";
  }

  // Allocate memory for function values at cell centres
  const std::size_t size = num_cells * data_dim;

  // Build lists of dofs and create map
  std::vector<dolfin::la_index_t> dof_set;
  std::vector<std::size_t> offset(size + 1);
  std::vector<std::size_t>::iterator cell_offset = offset.begin();
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Tabulate dofs
    auto dofs = dofmap.cell_dofs(cell.index());
    for (std::size_t i = 0; i < dofmap.num_element_dofs(cell.index()); ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1) = *(cell_offset) + dofmap.num_element_dofs(cell.index());
    ++cell_offset;
  }

  // Get  values
  std::vector<double> values(dof_set.size());
  assert(u.vector());
  u.vector()->get_local(values.data(), dof_set.size(), dof_set.data());

  // Get cell data
  fp << ascii_cell_data(mesh, offset, values, data_dim, rank);
  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------
std::string VTKWriter::ascii_cell_data(const mesh::Mesh& mesh,
                                       const std::vector<std::size_t>& offset,
                                       const std::vector<double>& values,
                                       std::size_t data_dim, std::size_t rank)
{
  std::ostringstream ss;
  ss << std::scientific;
  ss << std::setprecision(16);
  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  for (std::uint32_t i = 0;
       i != mesh.topology().ghost_offset(mesh.topology().dim()); ++i)
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
      for (std::size_t i = 0; i < 2; i++)
      {
        ss << values[*cell_offset + 2 * i] << " ";
        ss << values[*cell_offset + 2 * i + 1] << " ";
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
void VTKWriter::write_ascii_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
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
    log::dolfin_error("VTKWriter.cpp", "write mesh to VTK file"
                                  "Unable to open file \"%s\"",
                 filename.c_str());
  }

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\""
       << "ascii"
       << "\">";
  for (auto& v : mesh::MeshRange<mesh::Vertex>(mesh))
  {
    geometry::Point p = v.point();
    file << p[0] << " " << p[1] << " " << p[2] << "  ";
  }
  file << "</DataArray>" << std::endl << "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\""
       << "ascii"
       << "\">";

  std::unique_ptr<mesh::CellType> celltype(
      mesh::CellType::create(mesh.type().entity_type(cell_dim)));
  const std::vector<std::int8_t> perm = celltype->vtk_mapping();
  for (auto& c : mesh::MeshRange<mesh::MeshEntity>(mesh, cell_dim))
  {
    for (unsigned int i = 0; i != c.num_entities(0); ++i)
      file << c.entities(0)[perm[i]] << " ";
    file << " ";
  }
  file << "</DataArray>" << std::endl;

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\""
       << "ascii"
       << "\">";
  for (std::size_t offsets = 1; offsets <= num_cells; offsets++)
    file << offsets * num_cell_vertices << " ";
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\""
       << "ascii"
       << "\">";
  for (std::size_t types = 0; types < num_cells; types++)
    file << _vtk_cell_type << " ";
  file << "</DataArray>" << std::endl;
  file << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//-----------------------------------------------------------------------------
std::uint8_t VTKWriter::vtk_cell_type(const mesh::Mesh& mesh,
                                      std::size_t cell_dim)
{
  // Get cell type
  mesh::CellType::Type cell_type = mesh.type().entity_type(cell_dim);

  // Determine VTK cell type
  std::uint8_t vtk_cell_type = 0;
  if (cell_type == mesh::CellType::Type::tetrahedron)
    vtk_cell_type = 10;
  else if (cell_type == mesh::CellType::Type::hexahedron)
    vtk_cell_type = 12;
  else if (cell_type == mesh::CellType::Type::quadrilateral)
    vtk_cell_type = 9;
  else if (cell_type == mesh::CellType::Type::triangle)
    vtk_cell_type = 5;
  else if (cell_type == mesh::CellType::Type::interval)
    vtk_cell_type = 3;
  else if (cell_type == mesh::CellType::Type::point)
    vtk_cell_type = 1;
  else
  {
    log::dolfin_error("VTKWriter.cpp", "write data to VTK file",
                 "Unknown cell type (%d)", cell_type);
  }

  return vtk_cell_type;
}
//----------------------------------------------------------------------------
