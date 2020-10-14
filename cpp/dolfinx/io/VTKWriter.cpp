// Copyright (C) 2010-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKWriter.h"
#include "cells.h"
#include "pugixml.hpp"
#include <boost/filesystem.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------
// Get VTK cell type
std::int8_t get_vtk_cell_type(const mesh::Mesh& mesh, int cell_dim)
{

  // Get cell type
  mesh::CellType cell_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim);

  // Determine VTK cell type (Using arbitrary Lagrange elements)
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell_type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    return 72;
  default:
    throw std::runtime_error("Unknown cell type");
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
  auto cell_offset = offset.begin();
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  for (int i = 0; i < num_cells; ++i)
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
void write_ascii_mesh(const mesh::Mesh& mesh, int cell_dim,
                      std::string filename)
{

  // Get mesh geoemtry and number of cells
  const int num_cells = mesh.topology().index_map(cell_dim)->size_local();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> points
      = mesh.geometry().x();

  // Output arrays for XML
  std::int32_t num_nodes_per_cell;
  std::vector<std::int32_t> mesh_topology;

  // Get cell topology and map it from dolfin-X to VTK ordering
  const int tdim = mesh.topology().dim();
  if (cell_dim == 0)
  {
    for (std::int32_t i = 0; i < points.rows(); ++i)
      mesh_topology.push_back(i);
    num_nodes_per_cell = 1;
  }
  else if (cell_dim == tdim)
  {
    // Special case where the cells are visualized (Supports higher order
    // elements)
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh.geometry().dofmap();
    // FIXME: Use better way to get number of nods
    num_nodes_per_cell = x_dofmap.num_links(0);

    // Get map from VTK index i to DOLFIN index j
    std::vector map = io::cells::transpose(
        io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes_per_cell));

    // TODO: Remove when when paraview issue 19433 is resolved
    // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
    if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
        and num_nodes_per_cell == 27)
    {
      map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
             22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
    }

    for (int c = 0; c < x_dofmap.num_nodes(); ++c)
    {
      auto x_dofs = x_dofmap.links(c);
      for (int i = 0; i < x_dofs.rows(); ++i)
        mesh_topology.push_back(x_dofs(map[i]));
    }
  }
  else
  {
    // Create map from topology of cell of sub dimension to geometry entries
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> cell_indices(num_cells);
    for (std::int32_t i = 0; i < num_cells; ++i)
      cell_indices[i] = i;
    mesh::CellType cell_type
        = mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim);
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        sub_topology
        = mesh::entities_to_geometry(mesh, cell_dim, cell_indices, false);
    num_nodes_per_cell = sub_topology.cols();
    std::vector map = io::cells::transpose(
        io::cells::perm_vtk(cell_type, num_nodes_per_cell));
    for (int c = 0; c < sub_topology.rows(); ++c)
    {
      auto x_dofs = sub_topology.row(c);
      for (int i = 0; i < x_dofs.cols(); ++i)
        mesh_topology.push_back(x_dofs(map[i]));
    }
  }

  // Open file (Header should already have been created)
  if (!boost::filesystem::exists(filename))
    throw std::runtime_error("File " + filename + " does not exist");
  pugi::xml_document file;
  pugi::xml_parse_result result = file.load_file(filename.c_str());
  assert(result);

  // Select mesh node, Note: Could be done with xpath in the future
  pugi::xml_node node
      = file.select_node("/VTKFile/UnstructuredGrid/Piece").node();
  if (!node)
    throw std::runtime_error("XML node VTKFile/Unstructured/Piece not found.");

  // Add mesh geometry
  pugi::xml_node points_node = node.append_child("Points");
  pugi::xml_node data_item_node = points_node.append_child("DataArray");
  data_item_node.append_attribute("type") = "Float64";
  data_item_node.append_attribute("NumberOfComponents") = 3;
  data_item_node.append_attribute("format") = "ascii";

  // Flatten mesh geometry
  std::vector<double> x(points.rows() * points.cols(), 0.0);
  std::copy(points.data(), points.data() + points.rows() * points.cols(),
            x.begin());
  data_item_node.append_child(pugi::node_pcdata)
      .set_value(common::container_to_string(x, " ", 16, 3).c_str());

  // Save cell topology
  pugi::xml_node cells_node = node.append_child("Cells");
  pugi::xml_node connectivity_item_node = cells_node.append_child("DataArray");
  connectivity_item_node.append_attribute("type") = "Int32";
  connectivity_item_node.append_attribute("Name") = "connectivity";
  connectivity_item_node.append_attribute("format") = "ascii";
  connectivity_item_node.append_child(pugi::node_pcdata)
      .set_value(common::container_to_string(mesh_topology, " ", 16,
                                             num_nodes_per_cell)
                     .c_str());

  // Compute and add topology offsets
  std::vector<std::int32_t> offsets(num_cells);
  for (int i = 1; i <= num_cells; i++)
    offsets[i - 1] = i * num_nodes_per_cell;
  pugi::xml_node offsets_item_node = cells_node.append_child("DataArray");
  offsets_item_node.append_attribute("type") = "Int32";
  offsets_item_node.append_attribute("Name") = "offsets";
  offsets_item_node.append_attribute("format") = "ascii";
  offsets_item_node.append_child(pugi::node_pcdata)
      .set_value(common::container_to_string(offsets, " ", 16, 0).c_str());

  // Get VTK cell type
  const std::int8_t vtk_cell_type = get_vtk_cell_type(mesh, cell_dim);
  std::vector<std::int8_t> cell_types(num_cells, vtk_cell_type);
  pugi::xml_node types_item_node = cells_node.append_child("DataArray");
  types_item_node.append_attribute("type") = "Int8";
  types_item_node.append_attribute("Name") = "types";
  types_item_node.append_attribute("format") = "ascii";
  types_item_node.append_child(pugi::node_pcdata)
      .set_value(common::container_to_string(cell_types, " ", 8, 0).c_str());

  // Close file
  file.save_file(filename.c_str(), "  ");

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
void VTKWriter::write_cell_data(const function::Function<PetscScalar>& u,
                                std::string filename)
{
  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local();
  std::string encode_string = "ascii";

  // Get rank of function::Function
  const int rank = u.function_space()->element()->value_rank();
  if (rank > 2)
  {
    throw std::runtime_error("Don't know how to handle vector function with "
                             "dimension other than 2 or 3");
  }

  // Get number of components
  const int data_dim = u.function_space()->element()->value_size();

  // Open file
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Write headers
  if (rank == 0)
  {
    fp << "<CellData  Scalars=\""
       << "u"
       << "\"> " << std::endl;
    fp << R"(<DataArray  type="Float64"  Name=")"
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
    fp << R"(<DataArray  type="Float64"  Name=")"
       << "u"
       << R"("  NumberOfComponents="3" format=")" << encode_string << "\">";
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
    fp << R"(<DataArray  type="Float64"  Name=")"
       << "u"
       << R"("  NumberOfComponents="9" format=")" << encode_string << "\">";
  }

  // Allocate memory for function values at cell centres
  const std::size_t size = num_cells * data_dim;

  // Build lists of dofs and create map
  std::vector<std::int32_t> dof_set;
  std::vector<std::size_t> offset(size + 1);
  auto cell_offset = offset.begin();
  assert(dofmap->element_dof_layout);
  const int num_dofs_cell = dofmap->element_dof_layout->num_dofs();
  for (int c = 0; c < num_cells; ++c)
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(c);
    for (int i = 0; i < num_dofs_cell; ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1) = *(cell_offset) + num_dofs_cell;
    ++cell_offset;
  }

  // Get  values
  std::vector<PetscScalar> values(dof_set.size());
  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& _x = u.x()->array();
  for (std::size_t i = 0; i < dof_set.size(); ++i)
    values[i] = _x[dof_set[i]];

  // Get cell data
  fp << ascii_cell_data(*mesh, offset, values, data_dim, rank);
  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------
