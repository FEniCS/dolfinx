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
template <typename T>
std::shared_ptr<const mesh::Mesh<T>>
extract_common_mesh(const typename ADIOS2Writer<T>::U& u)
{
  // Extract mesh from first function
  assert(!u.empty());
  auto mesh = std::visit(
      [](const auto& u) { return u->function_space()->mesh(); }, u.front());
  assert(mesh);

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    std::visit(
        [&mesh](const auto& u)
        {
          if (mesh != u->function_space()->mesh())
          {
            throw std::runtime_error(
                "ADIOS2Writer only supports functions sharing the same mesh");
          }
        },
        v);
  }

  return mesh;
}
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

/// Extract name of functions and split into real and imaginary component
template <typename T>
std::vector<std::string>
extract_function_names(const typename ADIOS2Writer<T>::U& u)
{
  std::vector<std::string> names;
  using X = decltype(names);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd32>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd64>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc64>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc128>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            }},
        v);
    names.insert(names.end(), n.begin(), n.end());
  };

  return names;
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
template <typename T, typename U>
std::vector<T> pack_function_data(const fem::Function<T, U>& u)
{
  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  auto mesh = V->mesh();
  assert(mesh);
  const mesh::Geometry<U>& geometry = mesh->geometry();
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
template <typename T, typename U>
void fides_write_data(adios2::IO& io, adios2::Engine& engine,
                      const fem::Function<T, U>& u)
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
    using X = typename T::value_type;

    std::vector<X> data_real(data.size()), data_imag(data.size());

    adios2::Variable<X> local_output_r = define_variable<X>(
        io, u.name + field_ext[0], {}, {}, {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_real.begin(),
                   [](auto& x) -> X { return std::real(x); });
    engine.Put<X>(local_output_r, data_real.data());

    adios2::Variable<X> local_output_c = define_variable<X>(
        io, u.name + field_ext[1], {}, {}, {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_imag.begin(),
                   [](auto& x) -> X { return std::imag(x); });
    engine.Put<X>(local_output_c, data_imag.data());
    engine.PerformPuts();
  }
}
//-----------------------------------------------------------------------------

/// Write mesh geometry and connectivity (topology) for Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] engine The ADIOS2 engine
/// @param[in] mesh The mesh
template <typename T>
void fides_write_mesh(adios2::IO& io, adios2::Engine& engine,
                      const mesh::Mesh<T>& mesh)
{
  const mesh::Geometry<T>& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  // "Put" geometry data
  auto x_map = geometry.index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<T> local_geometry
      = define_variable<T>(io, "points", {}, {}, {num_vertices, 3});
  engine.Put<T>(local_geometry, geometry.x().data());

  // TODO: The DOLFINx and VTK topology are the same for some cell types
  // - no need to repack via extract_vtk_connectivity in these cases

  // Get topological dimenson, number of cells and number of 'nodes' per
  // cell, and compute 'VTK' connectivity
  const int tdim = topology.dim();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local();
  const int num_nodes = geometry.cmap().dim();
  const auto [cells, shape] = io::extract_vtk_connectivity(
      mesh.geometry().dofmap(), mesh.topology().cell_type());

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
template <typename T>
void fides_initialize_mesh_attributes(adios2::IO& io, const mesh::Mesh<T>& mesh)
{
  const mesh::Geometry<T>& geometry = mesh.geometry();
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
template <typename T>
void fides_initialize_function_attributes(adios2::IO& io,
                                          const typename ADIOS2Writer<T>::U& u)
{
  // Array of function (name, cell association types) for each function
  // added to the file
  std::vector<std::array<std::string, 2>> u_data;
  using X = decltype(u_data);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd32>& u)
                -> X {
              return {{u->name, "points"}};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd64>& u)
                -> X {
              return {{u->name, "points"}};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc64>& u)
                -> X
            {
              return {{u->name + field_ext[0], "points"},
                      {u->name + field_ext[1], "points"}};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc128>& u)
                -> X
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

/// Create VTK xml scheme to be interpreted by the VTX reader
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
std::stringstream
io::impl_vtx::create_vtk_schema(const std::vector<std::string>& point_data,
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

/// Extract name of functions and split into real and imaginary component
template <typename T>
std::vector<std::string>
extract_function_names(const typename ADIOS2Writer<T>::U& u)
{
  std::vector<std::string> names;
  using X = decltype(names);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd32>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd64>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc64>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc128>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            }},
        v);
    names.insert(names.end(), n.begin(), n.end());
  };

  return names;
}

//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
                         std::shared_ptr<const mesh::Mesh<double>> mesh)
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
    : ADIOS2Writer(comm, filename, "Fides function writer",
                   extract_common_mesh<double>(u), u),
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
  fides_initialize_function_attributes<double>(*_io, u);
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

#endif