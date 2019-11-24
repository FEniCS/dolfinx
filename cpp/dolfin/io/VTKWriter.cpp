// Copyright (C) 2010-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKWriter.h"
#include "cells.h"
#include <boost/detail/endian.hpp>
#include <cstdint>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

using namespace dolfin;
using namespace dolfin::io;

namespace
{
//-----------------------------------------------------------------------------
// Get VTK cell type
std::int8_t get_vtk_cell_type(const mesh::Mesh& mesh, std::size_t cell_dim,
                              std::size_t cell_order)
{
  // Get cell type
  mesh::CellType cell_type = mesh::cell_entity_type(mesh.cell_type(), cell_dim);

  // Determine VTK cell type
  switch (cell_type)
  {
  case mesh::CellType::tetrahedron:
    switch (cell_order)
    {
    case 1:
      return 10;
    default:
      return 71;
    }

  case mesh::CellType::hexahedron:
    switch (cell_order)
    {
    case 1:
      return 12;
    default:
      return 72;
    }
  case mesh::CellType::quadrilateral:
  {
    switch (cell_order)
    {
    case 1:
      return 9;
    default:
      return 70;
    }
  }
  case mesh::CellType::triangle:
  {
    switch (cell_order)
    {
    case 1:
      return 5;
    default:
      return 69;
    }
  }
  case mesh::CellType::interval:
    return 3;
  case mesh::CellType::point:
    return 1;
  default:
    throw std::runtime_error("Unknown cell type");
    return 0;
  }
}
//----------------------------------------------------------------------------
// Write cell data (ascii)
std::string ascii_cell_data(const mesh::Mesh& mesh,
                            const std::vector<std::size_t>& offset,
                            const std::vector<PetscScalar>& values,
                            std::size_t data_dim, std::size_t rank)
{
  std::ostringstream ss;
  ss << std::scientific;
  ss << std::setprecision(16);
  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  for (int i = 0; i < mesh.topology().ghost_offset(mesh.topology().dim()); ++i)
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
// mesh::Mesh writer (ascii)
void write_ascii_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
                      std::string filename)
{
  // FIXME: 'mesh.topology().ghost_offset' is plain confusing
  const int num_cells = mesh.topology().ghost_offset(cell_dim);
  const int element_degree = mesh.degree();

  // Get VTK cell type
  const std::int8_t vtk_cell_type
      = get_vtk_cell_type(mesh, cell_dim, element_degree);

  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  file.precision(16);
  if (!file.is_open())
  {
    throw std::runtime_error("Unable to open file:" + filename);
  }

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\""
       << "ascii"
       << "\">";
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> points
      = mesh.geometry().points();
  for (int i = 0; i < points.rows(); ++i)
    file << points(i, 0) << " " << points(i, 1) << " " << points(i, 2) << "  ";
  file << "</DataArray>" << std::endl << "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\""
       << "ascii"
       << "\">";

  int num_nodes;
  const int tdim = mesh.topology().dim();
  if (cell_dim == 0)
  {
    // Special case when only points should be visualized
    for (int i = 0; i < points.rows(); ++i)
      file << i << " ";
    file << "</DataArray>" << std::endl;
    num_nodes = 1;
  }
  else if (cell_dim == tdim)
  {
    // Special case where the cells are visualized (Supports higher order
    // elements)
    const mesh::Connectivity& connectivity_g
        = mesh.coordinate_dofs().entity_points();
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_connections
        = connectivity_g.connections();
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
        = connectivity_g.entity_positions();
    num_nodes = connectivity_g.size(0);

    const std::vector<std::uint8_t> perm
        = io::cells::dolfin_to_vtk(mesh.cell_type(), num_nodes);
    for (int j = 0; j < mesh.num_entities(mesh.topology().dim()); ++j)
    {
      for (int i = 0; i < num_nodes; ++i)
        file << cell_connections(pos_g(j) + perm[i]) << " ";
      file << " ";
    }
    file << "</DataArray>" << std::endl;
  }
  else
  {
    const int degree = mesh.degree();
    if (degree > 1)
    {
      throw std::runtime_error("MeshFunction of lower degree than the "
                               "topological dimension is not implemented");
    }
    mesh::CellType e_type = mesh::cell_entity_type(mesh.cell_type(), cell_dim);
    // FIXME : Need to implement permutations for higher order
    // geometries (aka line segments). CoordinateDofs needs to be
    // extended to have connections to facets.
    const int num_vertices = mesh::num_cell_vertices(e_type);
    const std::vector<std::uint8_t> perm
        = io::cells::dolfin_to_vtk(e_type, num_vertices);
    auto vertex_connectivity = mesh.topology().connectivity(cell_dim, 0);
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& vertex_connections
        = vertex_connectivity->connections();
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_vertex
        = vertex_connectivity->entity_positions();
    for (int j = 0; j < mesh.num_entities(cell_dim); ++j)
    {
      for (int i = 0; i < num_vertices; ++i)
        file << vertex_connections(pos_vertex(j) + perm[i]) << " ";
      file << " ";
    }
    file << "</DataArray>" << std::endl;
    // Change number of nodes to fix offset
    num_nodes = num_vertices;
  }

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\""
       << "ascii"
       << "\">";
  for (int offsets = 1; offsets <= num_cells; offsets++)
    file << offsets * num_nodes << " ";
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\""
       << "ascii"
       << "\">";
  for (int types = 0; types < num_cells; types++)
    file << vtk_cell_type << " ";
  file << "</DataArray>" << std::endl;
  file << "</Cells>" << std::endl;

  // Close file
  file.close();
} // namespace
//-----------------------------------------------------------------------------

} // namespace

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
  const fem::DofMap& dofmap = *u.function_space()->dofmap();
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().ghost_offset(tdim);

  std::string encode_string = "ascii";

  // Get rank of function::Function
  const std::size_t rank = u.value_rank();
  if (rank > 2)
  {
    throw std::runtime_error("Don't know how to handle vector function with "
                             "dimension other than 2 or 3");
  }

  // Get number of components
  const std::size_t data_dim = u.value_size();

  // Open file
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Write headers
  if (rank == 0)
  {
    fp << "<CellData  Scalars=\""
       << "u"
       << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\""
       << "u"
       << "\"  format=\"" << encode_string << "\">";
  }
  else if (rank == 1)
  {
    if (!(data_dim == 2 || data_dim == 3))
    {
      throw std::runtime_error(
          "Don't know how to handle vector function with dimension  "
          "other than 2 or 3");
    }
    fp << "<CellData  Vectors=\""
       << "u"
       << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\""
       << "u"
       << "\"  NumberOfComponents=\"3\" format=\"" << encode_string << "\">";
  }
  else if (rank == 2)
  {
    if (!(data_dim == 4 || data_dim == 9))
    {
      throw std::runtime_error("Don't know how to handle tensor function with "
                               "dimension other than 4 or 9");
    }
    fp << "<CellData  Tensors=\""
       << "u"
       << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\""
       << "u"
       << "\"  NumberOfComponents=\"9\" format=\"" << encode_string << "\">";
  }

  // Allocate memory for function values at cell centres
  const std::size_t size = num_cells * data_dim;

  // Build lists of dofs and create map
  std::vector<PetscInt> dof_set;
  std::vector<std::size_t> offset(size + 1);
  std::vector<std::size_t>::iterator cell_offset = offset.begin();
  assert(dofmap.element_dof_layout);
  const int num_dofs_cell = dofmap.element_dof_layout->num_dofs();
  for (auto& cell : mesh::MeshRange(mesh, tdim))
  {
    // Tabulate dofs
    auto dofs = dofmap.cell_dofs(cell.index());
    for (int i = 0; i < num_dofs_cell; ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1) = *(cell_offset) + num_dofs_cell;
    ++cell_offset;
  }

  // Get  values
  std::vector<PetscScalar> values(dof_set.size());
  la::VecReadWrapper u_wrapper(u.vector().vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x
      = u_wrapper.x;
  for (std::size_t i = 0; i < dof_set.size(); ++i)
    values[i] = _x[dof_set[i]];

  // Get cell data
  fp << ascii_cell_data(mesh, offset, values, data_dim, rank);
  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------
