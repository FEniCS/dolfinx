// Copyright (C) 2010-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKWriter.h"
#include "cells.h"
#include <Eigen/Core>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
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
// Write cell data (ascii)
template <typename Scalar>
std::string ascii_cell_data(const mesh::Mesh& mesh,
                            const std::vector<std::size_t>& offset,
                            const std::vector<Scalar>& values,
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
  const int num_cells = mesh.topology().index_map(cell_dim)->size_local();

  // Get VTK cell type
  const std::int8_t vtk_cell_type
      = io::cells::get_vtk_cell_type(mesh, cell_dim);

  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  file.precision(16);
  if (!file.is_open())
  {
    throw std::runtime_error("Unable to open file:" + filename);
  }

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << R"(<DataArray  type="Float64"  NumberOfComponents="3"  format=")"
       << "ascii"
       << "\">";

  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      points(mesh.geometry().x().data(), mesh.geometry().x().shape[0],
             mesh.geometry().x().shape[1]);

  for (int i = 0; i < points.rows(); ++i)
    file << points(i, 0) << " " << points(i, 1) << " " << points(i, 2) << "  ";
  file << "</DataArray>" << std::endl << "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << R"(<DataArray  type="Int32"  Name="connectivity"  format=")"
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
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh.geometry().dofmap();
    // FIXME: Use better way to get number of nods
    num_nodes = x_dofmap.num_links(0);

    // Get map from VTK index i to DOLFIN index j
    std::vector map = io::cells::transpose(
        io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes));

    // TODO: Remove when when paraview issue 19433 is resolved
    // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
    if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
        and num_nodes == 27)
    {
      map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
             22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
    }

    for (int c = 0; c < x_dofmap.num_nodes(); ++c)
    {
      auto x_dofs = x_dofmap.links(c);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
        file << x_dofs[map[i]] << " ";
      file << " ";
    }
    file << "</DataArray>" << std::endl;
  }
  else
  {
    throw std::runtime_error(
        "VTK outout for mesh_entities for dim<tdim is not implemented yet.");

    // Build a map from topology to geometry
    auto c_to_v = mesh.topology().connectivity(tdim, 0);
    assert(c_to_v);
    auto map = mesh.topology().index_map(0);
    assert(map);
    const std::int32_t num_mesh_vertices
        = map->size_local() + map->num_ghosts();

    auto x_dofmap = mesh.geometry().dofmap();
    std::vector<std::int32_t> vertex_to_node(num_mesh_vertices);
    for (int c = 0; c < c_to_v->num_nodes(); ++c)
    {
      auto vertices = c_to_v->links(c);
      auto x_dofs = x_dofmap.links(c);
      for (std::size_t i = 0; i < vertices.size(); ++i)
        vertex_to_node[vertices[i]] = x_dofs[i];
    }

    const mesh::CellType e_type
        = mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim);
    // FIXME : Need to implement re-mapping for higher order
    // geometries (aka line segments). CoordinateDofs needs to be
    // extended to have connections to facets.
    const int num_vertices = mesh::num_cell_vertices(e_type);
    const std::vector map_vtk
        = io::cells::transpose(io::cells::perm_vtk(e_type, num_vertices));
    auto e_to_v = mesh.topology().connectivity(cell_dim, 0);
    assert(e_to_v);
    for (int e = 0; e < e_to_v->num_nodes(); ++e)
    {
      auto vertices = e_to_v->links(e);
      for (int i = 0; i < num_vertices; ++i)
        file << vertex_to_node[vertices[map_vtk[i]]] << " ";
      file << " ";
    }
    file << "</DataArray>" << std::endl;
    // Change number of nodes to fix offset
    num_nodes = num_vertices;
  }

  // Write offset into connectivity array for the end of each cell
  file << R"(<DataArray  type="Int32"  Name="offsets"  format=")"
       << "ascii"
       << "\">";
  for (int offsets = 1; offsets <= num_cells; offsets++)
    file << offsets * num_nodes << " ";
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << R"(<DataArray  type="Int8"  Name="types"  format=")"
       << "ascii"
       << "\">";
  for (int types = 0; types < num_cells; types++)
    file << std::to_string(vtk_cell_type) << " ";
  file << "</DataArray>" << std::endl;
  file << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//-----------------------------------------------------------------------------
template <typename Scalar>
void _write_cell_data(const fem::Function<Scalar>& u, std::string filename)
{
  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local();
  std::string encode_string = "ascii";

  // Get rank of fem::Function
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
  std::vector<Scalar> values(dof_set.size());
  const std::vector<Scalar>& _x = u.x()->array();
  for (std::size_t i = 0; i < dof_set.size(); ++i)
    values[i] = _x[dof_set[i]];

  // Get cell data
  fp << ascii_cell_data(*mesh, offset, values, data_dim, rank);
  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
void VTKWriter::write_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
                           std::string filename)
{
  write_ascii_mesh(mesh, cell_dim, filename);
}
//----------------------------------------------------------------------------
void VTKWriter::write_cell_data(const fem::Function<double>& u,
                                std::string filename)
{
  _write_cell_data(u, filename);
}
//----------------------------------------------------------------------------
void VTKWriter::write_cell_data(const fem::Function<std::complex<double>>& u,
                                std::string filename)
{
  _write_cell_data(u, filename);
}
//----------------------------------------------------------------------------
