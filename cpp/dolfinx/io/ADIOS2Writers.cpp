// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2Writers.h"
#include "cells.h"
#include "vtk_utils.h"
#include <adios2.h>
#include <algorithm>
#include <array>
#include <complex>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <pugixml.hpp>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
/// String suffix for real and complex components of a vector-valued
/// field
constexpr std::array field_ext = {"_real", "_imag"};

template <class... Ts>
struct overload : Ts...
{
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>; // line not needed in C++20...

//-----------------------------------------------------------------------------
/// Safe definition of an attribute. First check if it has already been
/// defined and return it. If not defined create new attribute.
template <class T>
adios2::Attribute<T> define_attribute(adios2::IO& io, std::string name,
                                      const T& value, std::string var_name = "",
                                      std::string separator = "/")
{
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
    return attr;
  else
    return io.DefineAttribute<T>(name, value, var_name, separator);
}
//-----------------------------------------------------------------------------

/// Safe definition of a variable. First check if it has already been
/// defined and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> define_variable(adios2::IO& io, std::string name,
                                    const adios2::Dims& shape = adios2::Dims(),
                                    const adios2::Dims& start = adios2::Dims(),
                                    const adios2::Dims& count = adios2::Dims())
{
  if (adios2::Variable<T> v = io.InquireVariable<T>(name); v)
  {
    if (v.Count() != count and v.ShapeID() == adios2::ShapeID::LocalArray)
      v.SetSelection({start, count});
    return v;
  }
  else
    return io.DefineVariable<T>(name, shape, start, count);
}
//-----------------------------------------------------------------------------

/// Given a Function, write the coefficient to file using ADIOS2
/// @note Only supports (discontinuous) Lagrange functions.
/// @note For a complex function, the coefficient is split into a real
/// and imaginary function
/// @note Data is padded to be three dimensional if vector and 9
/// dimensional if tensor
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function
template <typename T>
void vtx_write_data(adios2::IO& io, adios2::Engine& engine,
                    const fem::Function<T>& u)
{
  // Get function data array and information about layout
  assert(u.x());
  std::span<const T> u_vector = u.x()->array();
  const int rank = u.function_space()->element()->value_shape().size();
  const std::uint32_t num_comp = std::pow(3, rank);
  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  std::shared_ptr<const common::IndexMap> index_map = dofmap->index_map;
  assert(index_map);
  const int index_map_bs = dofmap->index_map_bs();
  const int dofmap_bs = dofmap->bs();
  const std::uint32_t num_dofs
      = index_map_bs * (index_map->size_local() + index_map->num_ghosts())
        / dofmap_bs;

  if constexpr (std::is_scalar_v<T>)
  {
    // ---- Real
    std::vector<T> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = u_vector[i * index_map_bs + j];

    adios2::Variable<T> output
        = define_variable<T>(io, u.name, {}, {}, {num_dofs, num_comp});
    engine.Put<T>(output, data.data(), adios2::Mode::Sync);
  }
  else
  {
    // ---- Complex
    using U = typename T::value_type;

    std::vector<U> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::real(u_vector[i * index_map_bs + j]);

    adios2::Variable<U> output_real = define_variable<U>(
        io, u.name + field_ext[0], {}, {}, {num_dofs, num_comp});
    engine.Put<U>(output_real, data.data(), adios2::Mode::Sync);

    std::fill(data.begin(), data.end(), 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::imag(u_vector[i * index_map_bs + j]);
    adios2::Variable<U> output_imag = define_variable<U>(
        io, u.name + field_ext[1], {}, {}, {num_dofs, num_comp});
    engine.Put<U>(output_imag, data.data(), adios2::Mode::Sync);
  }
}
//-----------------------------------------------------------------------------

/// Write mesh to file using VTX format
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] mesh The mesh
void vtx_write_mesh(adios2::IO& io, adios2::Engine& engine,
                    const mesh::Mesh& mesh)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  // "Put" geometry
  std::shared_ptr<const common::IndexMap> x_map = geometry.index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = define_variable<double>(io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, geometry.x().data());

  // Put number of nodes. The mesh data is written with local indices,
  // therefore we need the ghost vertices.
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  const auto [vtkcells, shape] = io::extract_vtk_connectivity(mesh);

  // Add cell metadata
  const int tdim = topology.dim();
  adios2::Variable<std::uint32_t> cell_variable
      = define_variable<std::uint32_t>(io, "NumberOfCells",
                                       {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_variable, shape[0]);
  adios2::Variable<std::uint32_t> celltype_variable
      = define_variable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(
      celltype_variable, cells::get_vtk_cell_type(topology.cell_type(), tdim));

  // Pack mesh 'nodes'. Output is written as [N0, v0_0,...., v0_N0, N1,
  // v1_0,...., v1_N1,....], where N is the number of cell nodes and v0,
  // etc, is the node index
  std::vector<std::int64_t> cells(shape[0] * (shape[1] + 1), shape[1]);
  for (std::size_t c = 0; c < shape[0]; ++c)
  {
    std::span vtkcell(vtkcells.data() + c * shape[1], shape[1]);
    std::span cell(cells.data() + c * (shape[1] + 1), shape[1] + 1);
    std::copy(vtkcell.begin(), vtkcell.end(), std::next(cell.begin()));
  }

  // Put topology (nodes)
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {shape[0], shape[1] + 1});
  engine.Put<std::int64_t>(local_topology, cells.data());

  // Vertex global ids and ghost markers
  adios2::Variable<std::int64_t> orig_id = define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {num_vertices});
  engine.Put<std::int64_t>(orig_id, geometry.input_global_indices().data());

  std::vector<std::uint8_t> x_ghost(num_vertices, 0);
  std::fill(std::next(x_ghost.begin(), x_map->size_local()), x_ghost.end(), 1);
  adios2::Variable<std::uint8_t> ghost = define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put<std::uint8_t>(ghost, x_ghost.data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates. Writes the topology and geometry using ADIOS2 in
/// VTX format.
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function
void vtx_write_mesh_from_space(adios2::IO& io, adios2::Engine& engine,
                               const fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Get a VTK mesh with points at the 'nodes'
  const auto [x, xshape, x_id, x_ghost, vtk, vtkshape]
      = io::vtk_mesh_from_space(V);

  std::uint32_t num_dofs = xshape[0];

  // -- Pack mesh 'nodes'. Output is written as [N0, v0_0,...., v0_N0, N1,
  // v1_0,...., v1_N1,....], where N is the number of cell nodes and v0,
  // etc, is the node index.

  // Create vector, setting all entries to nodes per cell (vtk.shape(1))
  std::vector<std::int64_t> cells(vtkshape[0] * (vtkshape[1] + 1), vtkshape[1]);

  // Set the [v0_0,...., v0_N0, v1_0,...., v1_N1,....] data
  for (std::size_t c = 0; c < vtkshape[0]; ++c)
  {
    std::span vtkcell(vtk.data() + c * vtkshape[1], vtkshape[1]);
    std::span cell(cells.data() + c * (vtkshape[1] + 1), vtkshape[1] + 1);
    std::copy(vtkcell.begin(), vtkcell.end(), std::next(cell.begin()));
  }

  // Define ADIOS2 variables for geometry, topology, celltypes and
  // corresponding VTK data
  adios2::Variable<double> local_geometry
      = define_variable<double>(io, "geometry", {}, {}, {num_dofs, 3});
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {vtkshape[0], vtkshape[1] + 1});
  adios2::Variable<std::uint32_t> cell_type
      = define_variable<std::uint32_t>(io, "types");
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = define_variable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});

  // Write mesh information to file
  engine.Put<std::uint32_t>(vertices, num_dofs);
  engine.Put<std::uint32_t>(elements, vtkshape[0]);
  engine.Put<std::uint32_t>(
      cell_type, cells::get_vtk_cell_type(mesh->topology().cell_type(), tdim));
  engine.Put<double>(local_geometry, x.data());
  engine.Put<std::int64_t>(local_topology, cells.data());

  // Node global ids
  adios2::Variable<std::int64_t> orig_id = define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {x_id.size()});
  engine.Put<std::int64_t>(orig_id, x_id.data());
  adios2::Variable<std::uint8_t> ghost = define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put<std::uint8_t>(ghost, x_ghost.data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

/// Extract name of functions and split into real and imaginary component
std::vector<std::string> extract_function_names(const ADIOS2Writer::U& u)
{
  std::vector<std::string> names;
  using T = decltype(names);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{[](const std::shared_ptr<const ADIOS2Writer::Fd32>& u) -> T
                 { return {u->name}; },
                 [](const std::shared_ptr<const ADIOS2Writer::Fd64>& u) -> T
                 { return {u->name}; },
                 [](const std::shared_ptr<const ADIOS2Writer::Fc64>& u) -> T {
                   return {u->name + field_ext[0], u->name + field_ext[1]};
                 },
                 [](const std::shared_ptr<const ADIOS2Writer::Fc128>& u) -> T {
                   return {u->name + field_ext[0], u->name + field_ext[1]};
                 }},
        v);
    names.insert(names.end(), n.begin(), n.end());
  };

  return names;
}
//-----------------------------------------------------------------------------

/// Create VTK xml scheme to be interpreted by the VTX reader
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
std::stringstream create_vtk_schema(const std::vector<std::string>& point_data,
                                    const std::vector<std::string>& cell_data)
{
  // Create XML
  pugi::xml_document xml_schema;
  pugi::xml_node vtk_node = xml_schema.append_child("VTKFile");
  vtk_node.append_attribute("type") = "UnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node unstructured = vtk_node.append_child("UnstructuredGrid");

  // -- Piece

  pugi::xml_node piece = unstructured.append_child("Piece");

  // Add mesh attributes
  piece.append_attribute("NumberOfPoints") = "NumberOfNodes";
  piece.append_attribute("NumberOfCells") = "NumberOfCells";

  // -- Points

  // Add point information
  pugi::xml_node xml_geometry = piece.append_child("Points");
  pugi::xml_node xml_vertices = xml_geometry.append_child("DataArray");
  xml_vertices.append_attribute("Name") = "geometry";

  // -- Cells

  pugi::xml_node xml_topology = piece.append_child("Cells");
  xml_topology.append_child("DataArray").append_attribute("Name")
      = "connectivity";
  xml_topology.append_child("DataArray").append_attribute("Name") = "types";

  // -- PointData

  pugi::xml_node xml_pointdata = piece.append_child("PointData");

  // Stepping info for time dependency
  pugi::xml_node item_time = xml_pointdata.append_child("DataArray");
  item_time.append_attribute("Name") = "TIME";
  item_time.append_child(pugi::node_pcdata).set_value("step");

  pugi::xml_node item_idx = xml_pointdata.append_child("DataArray");
  item_idx.append_attribute("Name") = "vtkOriginalPointIds";
  pugi::xml_node item_ghost = xml_pointdata.append_child("DataArray");
  item_ghost.append_attribute("Name") = "vtkGhostType";
  for (auto& name : point_data)
  {
    pugi::xml_node item = xml_pointdata.append_child("DataArray");
    item.append_attribute("Name") = name.c_str();
  }

  // -- CellData

  if (!cell_data.empty())
  {
    pugi::xml_node xml_celldata = piece.append_child("CellData");
    for (auto& name : cell_data)
    {
      pugi::xml_node item = xml_celldata.append_child("DataArray");
      item.append_attribute("Name") = name.c_str();
    }
  }

  std::stringstream ss;
  xml_schema.save(ss, "  ");
  return ss;
}
//-----------------------------------------------------------------------------

/// Convert DOLFINx CellType to Fides CellType
/// https://gitlab.kitware.com/vtk/vtk-m/-/blob/master/vtkm/CellShape.h#L30-53
/// @param[in] type The DOLFInx cell
/// @return The Fides cell string
std::string to_fides_cell(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return "vertex";
  case mesh::CellType::interval:
    return "line";
  case mesh::CellType::triangle:
    return "triangle";
  case mesh::CellType::tetrahedron:
    return "tetrahedron";
  case mesh::CellType::quadrilateral:
    return "quad";
  case mesh::CellType::pyramid:
    return "pyramid";
  case mesh::CellType::prism:
    return "wedge";
  case mesh::CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------

/// Pack Function data at vertices. The mesh and the function must both
/// be 'P1'
template <typename T>
std::vector<T> pack_function_data(const fem::Function<T>& u)
{
  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  auto mesh = V->mesh();
  assert(mesh);
  const mesh::Geometry& geometry = mesh->geometry();
  const mesh::Topology& topology = mesh->topology();

  // The Function and the mesh must have identical element_dof_layouts
  // (up to the block size)
  assert(dofmap->element_dof_layout() == geometry.cmap().create_dof_layout());

  const int tdim = topology.dim();
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();

  auto vertex_map = topology.index_map(0);
  assert(vertex_map);
  const std::uint32_t num_vertices
      = vertex_map->size_local() + vertex_map->num_ghosts();

  const int rank = V->element()->value_shape().size();
  const std::uint32_t num_components = std::pow(3, rank);

  // Get dof array and pack into array (padded where appropriate)
  const graph::AdjacencyList<std::int32_t>& dofmap_x = geometry.dofmap();
  const int bs = dofmap->bs();
  const auto& u_data = u.x()->array();
  std::vector<T> data(num_vertices * num_components, 0);
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    auto dofs_x = dofmap_x.links(c);
    assert(dofs.size() == dofs_x.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int j = 0; j < bs; ++j)
        data[num_components * dofs_x[i] + j] = u_data[bs * dofs[i] + j];
  }
  return data;
}
//-----------------------------------------------------------------------------

/// Write a first order Lagrange function (real or complex) using ADIOS2
/// in Fides format. Data is padded to be three dimensional if vector
/// and 9 dimensional if tensor.
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function to write
template <typename T>
void fides_write_data(adios2::IO& io, adios2::Engine& engine,
                      const fem::Function<T>& u)
{
  // FIXME: There is an implicit assumptions that u and the mesh have
  // the same ElementDoflayout
  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  auto mesh = V->mesh();
  assert(mesh);
  const int gdim = mesh->geometry().dim();

  // Vectors and tensor need padding in gdim < 3
  const int rank = V->element()->value_shape().size();
  const bool need_padding = rank > 0 and gdim != 3 ? true : false;

  // Get vertex data. If the mesh and function dofmaps are the same we
  // can work directly with the dof array.
  std::span<const T> data;
  std::vector<T> _data;
  if (mesh->geometry().dofmap() == dofmap->list() and !need_padding)
    data = u.x()->array();
  else
  {
    _data = pack_function_data(u);
    data = std::span<const T>(_data);
  }

  auto vertex_map = mesh->topology().index_map(0);
  assert(vertex_map);
  const std::uint32_t num_vertices
      = vertex_map->size_local() + vertex_map->num_ghosts();

  // Write each real and imaginary part of the function
  const std::uint32_t num_components = std::pow(3, rank);
  assert(data.size() % num_components == 0);
  if constexpr (std::is_scalar_v<T>)
  {
    // ---- Real
    adios2::Variable<T> local_output = define_variable<T>(
        io, u.name, {}, {}, {num_vertices, num_components});

    // To reuse out_data, we use sync mode here
    engine.Put<T>(local_output, data.data());
    engine.PerformPuts();
  }
  else
  {
    // ---- Complex
    using U = typename T::value_type;

    std::vector<U> data_real(data.size()), data_imag(data.size());

    adios2::Variable<U> local_output_r = define_variable<U>(
        io, u.name + field_ext[0], {}, {}, {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_real.begin(),
                   [](auto& x) -> U { return std::real(x); });
    engine.Put<U>(local_output_r, data_real.data());

    adios2::Variable<U> local_output_c = define_variable<U>(
        io, u.name + field_ext[1], {}, {}, {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_imag.begin(),
                   [](auto& x) -> U { return std::imag(x); });
    engine.Put<U>(local_output_c, data_imag.data());
    engine.PerformPuts();
  }
}
//-----------------------------------------------------------------------------

/// Write mesh geometry and connectivity (topology) for Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] engine The ADIOS2 engine
/// @param[in] mesh The mesh
void fides_write_mesh(adios2::IO& io, adios2::Engine& engine,
                      const mesh::Mesh& mesh)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  // "Put" geometry data
  auto x_map = geometry.index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = define_variable<double>(io, "points", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, geometry.x().data());

  // TODO: The DOLFINx and VTK topology are the same for some cell types
  // - no need to repack via extract_vtk_connectivity in these cases

  // Get topological dimenson, number of cells and number of 'nodes' per
  // cell, and compute 'VTK' connectivity
  const int tdim = topology.dim();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local();
  const int num_nodes = geometry.cmap().dim();
  const auto [cells, shape] = io::extract_vtk_connectivity(mesh);

  // "Put" topology data in the result in the ADIOS2 file
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {std::size_t(num_cells * num_nodes)});
  engine.Put<std::int64_t>(local_topology, cells.data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

/// Initialize mesh related attributes for the ADIOS2 file used in Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] mesh The mesh
void fides_initialize_mesh_attributes(adios2::IO& io, const mesh::Mesh& mesh)
{
  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  // Check that mesh is first order mesh
  const int num_dofs_g = geometry.cmap().dim();
  const int num_vertices_per_cell
      = mesh::cell_num_entities(topology.cell_type(), 0);
  if (num_dofs_g != num_vertices_per_cell)
    throw std::runtime_error("Fides only supports lowest-order meshes.");

  // NOTE: If we start using mixed element types, we can change
  // data-model to "unstructured"
  define_attribute<std::string>(io, "Fides_Data_Model", "unstructured_single");

  // Define FIDES attributes pointing to ADIOS2 Variables for geometry
  // and topology
  define_attribute<std::string>(io, "Fides_Coordinates_Variable", "points");
  define_attribute<std::string>(io, "Fides_Connecticity_Variable",
                                "connectivity");

  std::string cell_type = to_fides_cell(topology.cell_type());
  define_attribute<std::string>(io, "Fides_Cell_Type", cell_type);

  define_attribute<std::string>(io, "Fides_Time_Variable", "step");
}
//-----------------------------------------------------------------------------

/// Initialize function related attributes for the ADIOS2 file used in
/// Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] functions The list of functions
void fides_initialize_function_attributes(adios2::IO& io,
                                          const ADIOS2Writer::U& u)
{
  // Array of function (name, cell association types) for each function
  // added to the file
  std::vector<std::array<std::string, 2>> u_data;
  using T = decltype(u_data);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{[](const std::shared_ptr<const ADIOS2Writer::Fd32>& u) -> T {
                   return {{u->name, "points"}};
                 },
                 [](const std::shared_ptr<const ADIOS2Writer::Fd64>& u) -> T {
                   return {{u->name, "points"}};
                 },
                 [](const std::shared_ptr<const ADIOS2Writer::Fc64>& u) -> T
                 {
                   return {{u->name + field_ext[0], "points"},
                           {u->name + field_ext[1], "points"}};
                 },
                 [](const std::shared_ptr<const ADIOS2Writer::Fc128>& u) -> T
                 {
                   return {{u->name + field_ext[0], "points"},
                           {u->name + field_ext[1], "points"}};
                 }},
        v);
    u_data.insert(u_data.end(), n.begin(), n.end());
  }

  // Write field associations to file
  if (adios2::Attribute<std::string> assc
      = io.InquireAttribute<std::string>("Fides_Variable_Associations");
      !assc)
  {
    std::vector<std::string> u_type;
    std::transform(u_data.cbegin(), u_data.cend(), std::back_inserter(u_type),
                   [](auto& f) { return f[1]; });
    io.DefineAttribute<std::string>("Fides_Variable_Associations",
                                    u_type.data(), u_type.size());
  }

  // Write field pointers to file
  if (adios2::Attribute<std::string> fields
      = io.InquireAttribute<std::string>("Fides_Variable_List");
      !fields)
  {
    std::vector<std::string> names;
    std::transform(u_data.cbegin(), u_data.cend(), std::back_inserter(names),
                   [](auto& f) { return f[0]; });
    io.DefineAttribute<std::string>("Fides_Variable_List", names.data(),
                                    names.size());
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
                           std::string tag,
                           std::shared_ptr<const mesh::Mesh> mesh, const U& u)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, adios2::Mode::Write))),
      _mesh(mesh), _u(u)
{
  _io->SetEngine("BPFile");
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
                           std::string tag,
                           std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, tag, mesh, {})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
                           std::string tag, const U& u)
    : ADIOS2Writer(comm, filename, tag, nullptr, u)
{
  // Extract mesh from first function
  assert(!u.empty());
  _mesh = std::visit([](const auto& u) { return u->function_space()->mesh(); },
                     u.front());
  assert(_mesh);

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    std::visit(
        [&mesh = _mesh](const auto& u)
        {
          if (mesh != u->function_space()->mesh())
          {
            throw std::runtime_error(
                "ADIOS2Writer only supports functions sharing the same mesh");
          }
        },
        v);
  }
}
//-----------------------------------------------------------------------------
ADIOS2Writer::~ADIOS2Writer() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2Writer::close()
{
  assert(_engine);
  // This looks odd because ADIOS2 uses `operator bool()` to test if the
  // engine is open
  if (*_engine)
    _engine->Close();
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
                         std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, "Fides mesh writer", mesh),
      _mesh_reuse_policy(MeshPolicy::update)
{
  assert(_io);
  assert(mesh);
  fides_initialize_mesh_attributes(*_io, *mesh);
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
                         const ADIOS2Writer::U& u, MeshPolicy policy)
    : ADIOS2Writer(comm, filename, "Fides function writer", u),
      _mesh_reuse_policy(policy)
{
  if (u.empty())
    throw std::runtime_error("FidesWriter fem::Function list is empty");

  // Extract Mesh from first function
  auto mesh = std::visit(
      [](const auto& u) { return u->function_space()->mesh(); }, u.front());
  assert(mesh);

  // Extract element from first function
  const fem::FiniteElement* element0 = std::visit(
      [](const auto& e) { return e->function_space()->element().get(); },
      u.front());
  assert(element0);

  // Check if function is mixed
  if (element0->is_mixed())
    throw std::runtime_error("Mixed functions are not supported by VTXWriter");

  // Check if function is DG 0
  if (element0->space_dimension() / element0->block_size() == 1)
  {
    throw std::runtime_error(
        "Piecewise constants are not (yet) supported by VTXWriter");
  }

  // FIXME: is the below check adequate for detecting a
  // Lagrange element? Check that element is Lagrange
  if (!element0->interpolation_ident())
  {
    throw std::runtime_error("Only Lagrange functions are "
                             "supported. Interpolate Functions before output.");
  }

  // Check that all functions are first order Lagrange
  int num_vertices_per_cell
      = mesh::cell_num_entities(mesh->topology().cell_type(), 0);
  for (auto& v : _u)
  {
    std::visit(
        [&](const auto& u)
        {
          auto element = u->function_space()->element();
          assert(element);
          if (*element != *element0)
          {
            throw std::runtime_error(
                "All functions in FidesWriter must have the same element type");
          }
          auto dof_layout = u->function_space()->dofmap()->element_dof_layout();
          int num_vertex_dofs = dof_layout.num_entity_dofs(0);
          int num_dofs = element->space_dimension() / element->block_size();
          if (num_dofs != num_vertices_per_cell or num_vertex_dofs != 1)
          {
            throw std::runtime_error("Only first order Lagrange spaces are "
                                     "supported by FidesWriter");
          }
        },
        v);
  }

  fides_initialize_mesh_attributes(*_io, *mesh);
  fides_initialize_function_attributes(*_io, u);
}
//-----------------------------------------------------------------------------
void FidesWriter::write(double t)
{
  assert(_io);
  assert(_engine);

  _engine->BeginStep();
  adios2::Variable<double> var_step = define_variable<double>(*_io, "step");
  _engine->Put<double>(var_step, t);

  if (auto v = _io->InquireVariable<std::int64_t>("connectivity");
      !v or _mesh_reuse_policy == MeshPolicy::update)
  {
    fides_write_mesh(*_io, *_engine, *_mesh);
  }

  for (auto& v : _u)
    std::visit([&](const auto& u) { fides_write_data(*_io, *_engine, *u); }, v);

  _engine->EndStep();
}
//-----------------------------------------------------------------------------
VTXWriter::VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
                     std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, "VTX mesh writer", mesh)
{
  // Define VTK scheme attribute for mesh
  std::string vtk_scheme = create_vtk_schema({}, {}).str();
  define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
}
//-----------------------------------------------------------------------------
VTXWriter::VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
                     const ADIOS2Writer::U& u)
    : ADIOS2Writer(comm, filename, "VTX function writer", u)
{
  if (u.empty())
    throw std::runtime_error("VTXWriter fem::Function list is empty");

  // Extract element from first function
  const fem::FiniteElement* element0 = std::visit(
      [](const auto& u) { return u->function_space()->element().get(); },
      u.front());
  assert(element0);

  // Check if function is mixed
  if (element0->is_mixed())
    throw std::runtime_error("Mixed functions are not supported by VTXWriter");

  // Check if function is DG 0
  if (element0->space_dimension() / element0->block_size() == 1)
  {
    throw std::runtime_error(
        "VTK does not support cell-wise fields. See "
        "https://gitlab.kitware.com/vtk/vtk/-/issues/18458.");
  }

  // FIXME: is the below check adequate for detecting a Lagrange
  // element?
  // Check that element is Lagrange
  if (!element0->interpolation_ident())
  {
    throw std::runtime_error("Only (discontinuous) Lagrange functions are "
                             "supported. Interpolate Functions before output.");
  }

  // Check that all functions come from same element type
  for (auto& v : _u)
  {
    std::visit(
        [element0](const auto& u)
        {
          auto element = u->function_space()->element();
          assert(element);
          if (*element != *element0)
          {
            throw std::runtime_error(
                "All functions in VTXWriter must have the same element type");
          }
        },
        v);
  }

  // Define VTK scheme attribute for set of functions
  std::vector<std::string> names = extract_function_names(u);
  std::string vtk_scheme = create_vtk_schema(names, {}).str();
  define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
}
//-----------------------------------------------------------------------------
void VTXWriter::write(double t)
{
  assert(_io);
  assert(_engine);
  adios2::Variable<double> var_step = define_variable<double>(*_io, "step");

  _engine->BeginStep();
  _engine->Put<double>(var_step, t);

  // If we have no functions write the mesh to file
  if (_u.empty())
    vtx_write_mesh(*_io, *_engine, *_mesh);
  else
  {
    // Write a single mesh for functions as they share finite element
    std::visit(
        [&](const auto& u)
        { vtx_write_mesh_from_space(*_io, *_engine, *u->function_space()); },
        _u[0]);

    // Write function data for each function to file
    for (auto& v : _u)
      std::visit([&](const auto& u) { vtx_write_data(*_io, *_engine, *u); }, v);
  }

  _engine->EndStep();
}
//-----------------------------------------------------------------------------

#endif