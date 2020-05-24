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
// #include <sstream>
// #include <vector>

using namespace dolfinx;
// using namespace dolfinx::io;

namespace
{
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

template <typename T>
std::string eigen_to_string(const T& x, int precision)
{
  std::stringstream s;
  s.precision(precision);
  for (Eigen::Index i = 0; i < x.size(); ++i)
    s << x.data()[i] << " ";
  return s.str();
}

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
//, _counter(0)
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
void io::VTKFileNew::write(const mesh::Mesh& mesh, double time)
{
  const int rank = MPI::rank(_comm.comm());
  boost::filesystem::path p(_filename);

  // Write to the PVD file
  pugi::xml_node xml_collections
      = _pvd_xml.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Count number of data sets
  const size_t n = std::distance(xml_collections.children("DataSet").begin(),
                                 xml_collections.children("DataSet").end());

  // Compute counter string
  const int num_digits = 6;
  std::string counter_str = std::to_string(n);
  counter_str
      = std::string(num_digits - counter_str.size(), '0').append(counter_str);

  // Write VTU file
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const int tdim = topology.dim();
  const std::int32_t num_points
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;
  add_mesh(mesh, piece_node);

  boost::filesystem::path p_vtu = p.stem();
  p_vtu += "_p" + std::to_string(rank) + "_" + counter_str;
  p_vtu.replace_extension("vtu");
  xml_vtu.save_file(p_vtu.c_str(), "  ");

  // Write PVTU file
  boost::filesystem::path p_pvtu = p.stem();
  p_pvtu += counter_str;
  p_pvtu.replace_extension("pvtu");

  pugi::xml_document xml_pvtu;
  pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // Mesh geometry data
  pugi::xml_node vertex_data_node = grid_node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";

  pugi::xml_node cell_data_node = grid_node.append_child("PCellData");
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "connectivity";
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "offsets";
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int8";
  data_node.append_attribute("Name") = "types";

  const int mpi_size = MPI::size(_comm.comm());
  for (int i = 0; i < mpi_size; ++i)
  {
    boost::filesystem::path p_vtu = p.stem();
    p_vtu += "_p" + std::to_string(i) + "_" + counter_str;
    p_vtu.replace_extension("vtu");
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = p_vtu.c_str();
  }

  if (MPI::rank(_comm.comm()) == 0)
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.c_str();
}
// //----------------------------------------------------------------------------
void io::VTKFileNew::write(const function::Function& u, double time)
{
  const int rank = MPI::rank(_comm.comm());
  boost::filesystem::path p(_filename);

  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  // Write to the PVD file
  pugi::xml_node xml_collections
      = _pvd_xml.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Count number of data sets
  const size_t n = std::distance(xml_collections.children("DataSet").begin(),
                                 xml_collections.children("DataSet").end());

  // Compute counter string
  const int num_digits = 6;
  std::string counter_str = std::to_string(n);
  counter_str
      = std::string(num_digits - counter_str.size(), '0').append(counter_str);

  // Prepare VTU node
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();
  const int tdim = topology.dim();
  const std::int32_t num_points
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;

  // Add mesh to VTU node
  add_mesh(*mesh, piece_node);

  // Add cell/point data to VTU node
  pugi::xml_node pointdata_node = piece_node.append_child("PointData");
  pointdata_node.append_attribute("Scalars") = u.name.c_str();
  pugi::xml_node field_node = pointdata_node.append_child("DataArray");
  field_node.append_attribute("type") = "Float64";
  field_node.append_attribute("Name") = u.name.c_str();
  field_node.append_attribute("format") = "ascii";

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values = u.compute_point_values();

  field_node.append_child(pugi::node_pcdata)
      .set_value(eigen_to_string(values, 16).c_str());

  boost::filesystem::path p_vtu = p.stem();
  p_vtu += "_p" + std::to_string(rank) + "_" + counter_str;
  p_vtu.replace_extension("vtu");
  xml_vtu.save_file(p_vtu.c_str(), "  ");

  // Write PVTU file
  boost::filesystem::path p_pvtu = p.stem();
  p_pvtu += counter_str;
  p_pvtu.replace_extension("pvtu");

  pugi::xml_document xml_pvtu;
  pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // Mesh geometry data
  pugi::xml_node vertex_data_node = grid_node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";

  pugi::xml_node cell_data_node = grid_node.append_child("PCellData");
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "connectivity";
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "offsets";
  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int8";
  data_node.append_attribute("Name") = "types";

  pugi::xml_node pointdata_pnode = grid_node.append_child("PPointData");
  pointdata_pnode.append_attribute("Scalars") = u.name.c_str();
  data_node = pointdata_pnode.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("Name") = u.name.c_str();
  data_node.append_attribute("NumberOfComponents") = 0;

  const int mpi_size = MPI::size(_comm.comm());
  for (int i = 0; i < mpi_size; ++i)
  {
    boost::filesystem::path p_vtu = p.stem();
    p_vtu += "_p" + std::to_string(i) + "_" + counter_str;
    p_vtu.replace_extension("vtu");
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = p_vtu.c_str();
  }

  if (MPI::rank(_comm.comm()) == 0)
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.c_str();
}
//----------------------------------------------------------------------------
