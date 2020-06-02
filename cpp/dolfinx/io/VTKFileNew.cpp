// Copyright (C) 2005-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFileNew.h"
#include "cells.h"
#include <boost/filesystem.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <sstream>
#include <string>

// #include "VTKWriter.h"
// #include "pugixml.hpp"
// #include <boost/cstdint.hpp>
// #include <boost/detail/endian.hpp>
// #include <dolfinx/common/IndexMap.h>
// #include <dolfinx/common/MPI.h>
// #include <dolfinx/common/Timer.h>
// #include <dolfinx/common/log.h>
// #include <dolfinx/fem/DofMap.h>
// #include <dolfinx/fem/FiniteElement.h>
// #include <dolfinx/function/Function.h>
// #include <dolfinx/function/FunctionSpace.h>
// #include <dolfinx/la/PETScVector.h>
// #include <dolfinx/mesh/Geometry.h>
// #include <dolfinx/mesh/Mesh.h>
// #include <dolfinx/mesh/MeshEntity.h>
// #include <iomanip>
// #include <ostream>
// #include <vector>

using namespace dolfinx;

namespace
{

/// Get counter string to include in filename
std::string get_counter(const pugi::xml_node& node, const std::string& name)
{
  // Count number of entries
  const size_t n = std::distance(node.children(name.c_str()).begin(),
                                 node.children(name.c_str()).end());

  // Compute counter string
  const int num_digits = 6;
  std::string counter = std::to_string(n);
  return std::string(num_digits - counter.size(), '0').append(counter);
}

/// Get the VTK cell type integer
std::int8_t get_vtk_cell_type(mesh::CellType cell, int dim)
{
  // Get cell type
  mesh::CellType cell_type = mesh::cell_entity_type(cell, dim);

  // Determine VTK cell type (arbitrary Lagrange elements)
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

/// Convert and Eigen array/matrix to a std::string
template <typename T>
std::string eigen_to_string(const T& x, int precision)
{
  std::stringstream s;
  s.precision(precision);
  for (Eigen::Index i = 0; i < x.size(); ++i)
    s << x.data()[i] << " ";
  return s.str();
}

void add_pvtu_mesh(pugi::xml_node& node)
{
  pugi::xml_node vertex_data_node = node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";

  // pugi::xml_node cell_data_node = node.append_child("PCellData");
  // data_node = cell_data_node.append_child("PDataArray");
  // data_node.append_attribute("type") = "Int32";
  // data_node.append_attribute("Name") = "connectivity";
  // data_node = cell_data_node.append_child("PDataArray");
  // data_node.append_attribute("type") = "Int32";
  // data_node.append_attribute("Name") = "offsets";
  // data_node = cell_data_node.append_child("PDataArray");
  // data_node.append_attribute("type") = "Int8";
  // data_node.append_attribute("Name") = "types";
}

//----------------------------------------------------------------------------
/// At mesh point data to a pugixml node.
void add_point_data(const function::Function& u, pugi::xml_node& node)
{
  const int rank = u.value_rank();
  const int dim = u.value_size();
  if (rank == 1)
  {
    if (!(dim == 2 or dim == 3))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. Don't know how to handle vector "
          "function with dimension other than 2 or 3");
    }
  }
  else if (rank == 2)
  {
    if (!(dim == 4 or dim == 9))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. Don't know how to handle tensor "
          "function with dimension other than 4 or 9");
    }
  }
  else if (rank > 2)
  {
    throw std::runtime_error(
        "Cannot write data to VTK file. "
        "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  pugi::xml_node pointdata_node = node.append_child("PointData");

  pugi::xml_node field_node = pointdata_node.append_child("DataArray");
  field_node.append_attribute("type") = "Float64";
  field_node.append_attribute("Name") = u.name.c_str();
  field_node.append_attribute("format") = "ascii";

  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      values = u.compute_point_values();
  if (rank == 0)
  {
    pointdata_node.append_attribute("Scalars") = u.name.c_str();
    field_node.append_child(pugi::node_pcdata)
        .set_value(eigen_to_string(values, 16).c_str());
  }
  else if (rank == 1)
  {
    pointdata_node.append_attribute("Vectors") = u.name.c_str();
    field_node.append_attribute("NumberOfComponents") = 3;
    if (dim == 2)
    {
      assert(values.cols() == 2);
      std::stringstream ss;
      for (int i = 0; i < values.rows(); ++i)
      {
        for (int j = 0; j < 2; ++j)
          ss << values(i, j) << " ";
        ss << 0.0 << " ";
      }
      field_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
    }
    else
    {
      assert(values.cols() == 3);
      field_node.append_child(pugi::node_pcdata)
          .set_value(eigen_to_string(values, 16).c_str());
    }
  }
  else if (rank == 2)
  {
    pointdata_node.append_attribute("Tensors") = u.name.c_str();
    field_node.append_attribute("NumberOfComponents") = 9;
    if (dim == 2)
    {
      // Pad 2D tensors with 0.0 to make them 3D
      std::stringstream ss;
      for (int i = 0; i < values.rows(); ++i)
      {
        for (int j = 0; j < 2; ++j)
        {
          ss << values(i, (2 * j + 0)) << " ";
          ss << values(i, (2 * j + 1)) << " ";
          ss << 0.0 << " ";
        }
        ss << 0.0 << " ";
        ss << 0.0 << " ";
        ss << 0.0 << "  ";
      }
      field_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
    }
    else
    {
      field_node.append_child(pugi::node_pcdata)
          .set_value(eigen_to_string(values, 16).c_str());
    }
  }
}

//----------------------------------------------------------------------------
/// At mesh geometry and topology data to a pugixml node. The function /
/// adds the Points and Cells nodes to the input node/
void add_mesh(const mesh::Mesh& mesh, pugi::xml_node& piece_node)
{
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const int tdim = topology.dim();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  // Add geometry (points)

  pugi::xml_node points_node = piece_node.append_child("Points");
  pugi::xml_node x_node = points_node.append_child("DataArray");
  x_node.append_attribute("type") = "Float64";
  x_node.append_attribute("NumberOfComponents") = "3";
  x_node.append_attribute("format") = "ascii";
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x
      = geometry.x();
  x_node.append_child(pugi::node_pcdata)
      .set_value(eigen_to_string(x, 16).c_str());

  // Add topology(cells)

  pugi::xml_node cells_node = piece_node.append_child("Cells");
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  pugi::xml_node connectivity_node = cells_node.append_child("DataArray");
  connectivity_node.append_attribute("type") = "Int32";
  connectivity_node.append_attribute("Name") = "connectivity";
  connectivity_node.append_attribute("format") = "ascii";

  // Get map from VTK index i to DOLFIN index j
  int num_nodes = geometry.cmap().dof_layout().num_dofs();

  std::vector<std::uint8_t> map = io::cells::transpose(
      io::cells::perm_vtk(topology.cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (topology.cell_type() == mesh::CellType::hexahedron and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  std::stringstream ss;
  for (int c = 0; c < x_dofmap.num_nodes(); ++c)
  {
    auto cell = x_dofmap.links(c);
    for (int i = 0; i < cell.size(); ++i)
      ss << cell(map[i]) << " ";
  }
  connectivity_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());

  pugi::xml_node offsets_node = cells_node.append_child("DataArray");
  offsets_node.append_attribute("type") = "Int32";
  offsets_node.append_attribute("Name") = "offsets";
  offsets_node.append_attribute("format") = "ascii";
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets
      = x_dofmap.offsets();
  offsets_node.append_child(pugi::node_pcdata)
      .set_value(eigen_to_string(offsets.tail(offsets.size() - 1), 0).c_str());

  pugi::xml_node type_node = cells_node.append_child("DataArray");
  type_node.append_attribute("type") = "Int8";
  type_node.append_attribute("Name") = "types";
  type_node.append_attribute("format") = "ascii";
  int celltype = get_vtk_cell_type(topology.cell_type(), tdim);
  std::stringstream s;
  for (std::int32_t c = 0; c < num_cells; ++c)
    s << celltype << " ";
  type_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
}
} // namespace

//----------------------------------------------------------------------------
io::VTKFileNew::VTKFileNew(MPI_Comm comm, const std::string filename,
                           const std::string)
    : _filename(filename), _comm(comm)
{
  pugi::xml_node vtk_node = _pvd_xml.append_child("VTKFile");
  vtk_node.append_attribute("type") = "Collection";
  vtk_node.append_attribute("version") = "0.1";
  vtk_node.append_child("Collection");
}
//----------------------------------------------------------------------------
io::VTKFileNew::~VTKFileNew()
{
  if (MPI::rank(_comm.comm()) == 0)
    _pvd_xml.save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void io::VTKFileNew::close()
{
  if (MPI::rank(_comm.comm()) == 0)
    _pvd_xml.save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void io::VTKFileNew::flush()
{
  if (MPI::rank(_comm.comm()) == 0)
    _pvd_xml.save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void io::VTKFileNew::write(const mesh::Mesh& mesh, double time)
{
  const int mpi_rank = MPI::rank(_comm.comm());
  boost::filesystem::path p(_filename);

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = _pvd_xml.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Get mesh data for this rank
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const int tdim = topology.dim();
  const std::int32_t num_points
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  // Create a VTU XML object
  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;

  // Add mesh data to "Piece" node
  add_mesh(mesh, piece_node);

  // Save VTU XML to file
  boost::filesystem::path p_vtu = p.stem();
  p_vtu += "_p" + std::to_string(mpi_rank) + "_" + counter_str;
  p_vtu.replace_extension("vtu");
  xml_vtu.save_file(p_vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  boost::filesystem::path p_pvtu = p.stem();
  p_pvtu += counter_str;
  p_pvtu.replace_extension("pvtu");
  if (mpi_rank == 0)
  {
    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "0.1";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 0;

    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);

    // Add data for each process to the PVTU object
    const int mpi_size = MPI::size(_comm.comm());
    for (int i = 0; i < mpi_size; ++i)
    {
      boost::filesystem::path vtu = p.stem();
      vtu += "_p" + std::to_string(i) + "_" + counter_str;
      vtu.replace_extension("vtu");
      pugi::xml_node piece_node = grid_node.append_child("Piece");
      piece_node.append_attribute("Source") = vtu.c_str();
    }

    // Write PVTU file
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.c_str();
}
//----------------------------------------------------------------------------
void io::VTKFileNew::write(const function::Function& u, double time)
{
  const int mpi_rank = MPI::rank(_comm.comm());
  boost::filesystem::path p(_filename);

  // Get the mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = _pvd_xml.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Get mesh data for this rank
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();
  const int tdim = topology.dim();
  const std::int32_t num_points
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  // Create a VTU XML object
  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;

  // Add mesh data to "Piece" node
  add_mesh(*mesh, piece_node);

  // Add cell/point data to VTU node
  add_point_data(u, piece_node);

  // Save VTU XML to file
  boost::filesystem::path vtu = p.stem();
  vtu += "_p" + std::to_string(mpi_rank) + "_" + counter_str;
  vtu.replace_extension("vtu");
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  boost::filesystem::path p_pvtu = p.stem();
  p_pvtu += counter_str;
  p_pvtu.replace_extension("pvtu");
  if (mpi_rank == 0)
  {
    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "0.1";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 0;

    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);

    // Add field data
    pugi::xml_node pointdata_pnode = grid_node.append_child("PPointData");
    pointdata_pnode.append_attribute("Scalars") = u.name.c_str();
    pugi::xml_node data_node = pointdata_pnode.append_child("PDataArray");
    data_node.append_attribute("type") = "Float64";
    data_node.append_attribute("Name") = u.name.c_str();
    data_node.append_attribute("NumberOfComponents") = 0;

    // Add data for each process to the PVTU object
    const int mpi_size = MPI::size(_comm.comm());
    for (int i = 0; i < mpi_size; ++i)
    {
      boost::filesystem::path vtu = p.stem();
      vtu += "_p" + std::to_string(i) + "_" + counter_str;
      vtu.replace_extension("vtu");
      pugi::xml_node piece_node = grid_node.append_child("Piece");
      piece_node.append_attribute("Source") = vtu.c_str();
    }

    // Write PVTU file
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.c_str();
}
//----------------------------------------------------------------------------
