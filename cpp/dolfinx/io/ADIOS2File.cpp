// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2File.h"
#include "dolfinx/fem/FiniteElement.h"
#include "dolfinx/fem/FunctionSpace.h"
#include "dolfinx/io/cells.h"
#include "dolfinx/mesh/utils.h"
#include "pugixml.hpp"
#include <adios2.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
// @todo Add description
std::string create_vtk_schema(const std::set<std::string>& point_data,
                              const std::set<std::string>& cell_data,
                              bool _time_dep)
{
  // Create XML SCHEMA by using pugi_xml
  pugi::xml_document xml_schema;
  pugi::xml_node vtk_node = xml_schema.append_child("VTKFile");
  vtk_node.append_attribute("type") = "UnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  vtk_node.append_attribute("byte_order") = "LittleEndian";
  pugi::xml_node unstructured = vtk_node.append_child("UnstructuredGrid");
  pugi::xml_node piece = unstructured.append_child("Piece");

  // Create VTK schema for mesh
  piece.append_attribute("NumberOfPoints") = "NumberOfNodes";
  piece.append_attribute("NumberOfCells") = "NumberOfEntities";
  pugi::xml_node geometry = piece.append_child("Points");
  pugi::xml_node vertices = geometry.append_child("DataArray");
  vertices.append_attribute("Name") = "geometry";

  pugi::xml_node topology = piece.append_child("Cells");
  {
    std::vector<std::string> topology_data = {"connectivity", "types"};
    for (auto data : topology_data)
    {
      pugi::xml_node item = topology.append_child("DataArray");
      item.append_attribute("Name") = data.c_str();
    }
  }

  {
    pugi::xml_node data = piece.append_child("PointData");

    // If we have any point data to write to file
    // Stepping info for time dependency
    if (_time_dep)
    {
      pugi::xml_node item = data.append_child("DataArray");
      item.append_attribute("Name") = "TIME";
      item.append_child(pugi::node_pcdata).set_value("step");
    }
    // Append point data to VTK Schema
    for (auto name : point_data)
    {
      pugi::xml_node item = data.append_child("DataArray");
      item.append_attribute("Name") = name.c_str();
    }
  }
  {
    pugi::xml_node data = piece.append_child("CellData");
    // Append point data to VTK Schema
    for (auto name : cell_data)
    {
      pugi::xml_node item = data.append_child("DataArray");
      item.append_attribute("Name") = name.c_str();
    }
  }

  std::stringstream ss;
  xml_schema.save(ss, "  ");
  return ss.str();
}
} // namespace
//-----------------------------------------------------------------------------

namespace
{
//-----------------------------------------------------------------------------
// Safe definition of an attribute (required for time dependent problems)
template <class T>
adios2::Attribute<T>
DefineAttribute(adios2::IO& io, const std::string& attr_name, const T& value,
                const std::string& var_name = "",
                const std::string& separator = "/")
{
  adios2::Attribute<T> attribute = io.InquireAttribute<T>(attr_name);
  if (attribute)
    return attribute;
  else
    return io.DefineAttribute<T>(attr_name, value, var_name, separator);
}
//-----------------------------------------------------------------------------
// Safe definition of a variable (required for time dependent problems)
template <class T>
adios2::Variable<T> DefineVariable(adios2::IO& io, const std::string& var_name,
                                   const adios2::Dims& shape = adios2::Dims(),
                                   const adios2::Dims& start = adios2::Dims(),
                                   const adios2::Dims& count = adios2::Dims())
{
  adios2::Variable<T> variable = io.InquireVariable<T>(var_name);
  if (variable)
  {
    if (variable.Count() != count
        and variable.ShapeID() == adios2::ShapeID::LocalArray)
    {
      variable.SetSelection({start, count});
    }
  }
  else
    variable = io.DefineVariable<T>(var_name, shape, start, count);

  return variable;
}
//-----------------------------------------------------------------------------
adios2::Mode string_to_mode(const std::string& mode)
{
  if (mode == "w")
    return adios2::Mode::Write;
  else if (mode == "a")
    return adios2::Mode::Append;
  else if (mode == "r")
    return adios2::Mode::Read;
  else
    throw std::runtime_error("Unknown mode for ADIOS2: " + mode);
}
//-----------------------------------------------------------------------------
bool is_cellwise_constant(const fem::FiniteElement& element)
{
  std::string family = element.family();
  int num_nodes_per_dim = element.space_dimension() / element.block_size();
  return num_nodes_per_dim == 1;
}
//-----------------------------------------------------------------------------
void _write_mesh(adios2::IO& io, adios2::Engine& engine, const mesh::Mesh& mesh)
{
  // assert(_engine);
  // // ADIOS should handle mode checks, and if we need to we should get it
  // // from ADIOS - DOLFINx should not store the state
  // if (_engine->OpenMode() == adios2::Mode::Append)
  // {
  //   throw std::runtime_error(
  //       "Cannot append functions to previously created file.");
  // }

  std::shared_ptr<const common::IndexMap> x_map = mesh.geometry().index_map();
  const int tdim = mesh.topology().dim();
  const std::uint32_t num_cells = mesh.topology().index_map(tdim)->size_local();

  // Add number of nodes (mesh data is written with local indices we need the
  // ghost vertices)
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  // Add cell metadata
  adios2::Variable<std::uint32_t> cell_variable = DefineVariable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_variable, num_cells);

  adios2::Variable<std::uint32_t> celltype_variable
      = DefineVariable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(celltype_variable,
                            cells::get_vtk_cell_type(mesh, tdim));

  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      cells::perm_vtk(mesh.topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  // Extract mesh 'nodes'
  // Output is written as [N0 v0_0 .... v0_N0 N1 v1_0 .... v1_N1 ....]
  std::vector<std::uint64_t> vtk_topology;
  vtk_topology.reserve(num_cells * (num_nodes + 1));
  for (size_t c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    vtk_topology.push_back(x_dofs.size());
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      vtk_topology.push_back(x_dofs[map[i]]);
  }

  // Start topology (node) writer
  adios2::Variable<std::uint64_t> local_topology
      = DefineVariable<std::uint64_t>(io, "connectivity", {}, {},
                                      {num_cells, num_nodes + 1});
  engine.Put<std::uint64_t>(local_topology, vtk_topology.data());

  // Start geometry writer
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh.geometry().x().data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------
template <typename Scalar>
void _write_lagrange_function(adios2::IO& io, adios2::Engine& engine,
                              const fem::Function<Scalar>& u, double /*t*/,
                              bool time_dep)
{
  std::set<std::string> point_data;

  auto V = u.function_space();
  auto mesh = u.function_space()->mesh();
  std::string family = V->element()->family();

  array2d<double> geometry = V->tabulate_dof_coordinates(false);
  const std::uint32_t num_dofs = geometry.shape[0];
  const std::uint32_t num_elements
      = mesh->topology().index_map(mesh->topology().dim())->size_local();

  // Create permutation from DOLFINx dof ordering to VTK
  std::shared_ptr<const fem::DofMap> dofmap = V->dofmap();
  const std::uint32_t num_nodes = dofmap->cell_dofs(0).size();
  std::vector<std::uint8_t> map;
  if (family == "Lagrange")
  {
    map = dolfinx::io::cells::transpose(
        io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));
  }
  else
    map = dolfinx::io::cells::perm_discontinuous(mesh->topology().cell_type(),
                                                 num_nodes);

  // Extract topology for all local cells as
  // [N0 v0_0 .... v0_N0 N1 v1_0 .... v1_N1 ....]
  std::vector<std::uint64_t> vtk_topology(num_elements * (num_nodes + 1));
  int topology_offset = 0;

  for (size_t c = 0; c < num_elements; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    vtk_topology[topology_offset++] = dofs.size();
    for (std::size_t i = 0; i < dofs.size(); ++i)
      vtk_topology[topology_offset++] = dofs[map[i]];
  }

  // Define ADIOS2 variables for geometry, topology, celltypes and
  // corresponding VTK data
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "geometry", {}, {}, {num_dofs, 3});
  adios2::Variable<std::uint64_t> local_topology
      = DefineVariable<std::uint64_t>(io, "connectivity", {}, {},
                                      {num_elements, num_nodes + 1});
  adios2::Variable<std::uint32_t> cell_type
      = DefineVariable<std::uint32_t>(io, "types");
  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = DefineVariable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});

  // Start writer and insert mesh information
  engine.BeginStep();
  engine.Put<std::uint32_t>(vertices, num_dofs);
  engine.Put<std::uint32_t>(elements, num_elements);
  engine.Put<std::uint32_t>(cell_type, dolfinx::io::cells::get_vtk_cell_type(
                                           *mesh, mesh->topology().dim()));
  engine.Put<double>(local_geometry, geometry.data());
  engine.Put<std::uint64_t>(local_topology, vtk_topology.data());

  // Get function data array and information about layout
  std::shared_ptr<const la::Vector<Scalar>> function_vector = u.x();
  const std::vector<Scalar>& function_data = function_vector->array();
  const int rank = u.function_space()->element()->value_rank();
  const std::uint32_t num_components = std::pow(3, rank);
  const std::uint32_t local_size = geometry.shape[0];
  const std::uint32_t block_size = dofmap->index_map_bs();
  std::vector<double> out_data(num_components * local_size);

  // Write each real and imaginary part of the function
  std::vector<std::string> parts = {""};
  if constexpr (!std::is_scalar<Scalar>::value)
    parts = {"real", "imag"};
  for (const auto& part : parts)
  {
    std::string function_name = u.name;
    if (part != "")
      function_name += "_" + part;
    // Extract real or imaginary part
    for (size_t i = 0; i < local_size; ++i)
    {
      for (size_t j = 0; j < block_size; ++j)
      {
        if (part == "imag")
        {
          out_data[i * num_components + j]
              = std::imag(function_data[i * block_size + j]);
        }
        else
        {
          out_data[i * num_components + j]
              = std::real(function_data[i * block_size + j]);
        }
      }

      // Pad data to 3D if vector or tensor data
      for (size_t j = block_size; j < num_components; ++j)
        out_data[i * num_components + j] = 0;
    }
    point_data.insert(function_name);

    // To reuse out_data, we use sync mode here
    adios2::Variable<double> local_output = DefineVariable<double>(
        io, function_name, {}, {}, {local_size, num_components});
    engine.Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
  }

  // Check if VTKScheme exists, and if so, check that we are only adding
  // values already existing
  std::string vtk_scheme = create_vtk_schema({point_data}, {}, time_dep);

  // If writing to file set vtk scheme as current
  // if (_vtk_scheme.empty())
  //   _vtk_scheme = vtk_scheme;

  // if (vtk_scheme != _vtk_scheme)
  // {
  //   throw std::runtime_error(
  //       "Have to write the same functions to file for each "
  //       "time step");
  // }

  DefineAttribute<std::string>(io, "vtk.xml", vtk_scheme);
  engine.EndStep();
}
//-----------------------------------------------------------------------------
template <typename Scalar>
void _write_function_at_nodes(
    adios2::IO& io, adios2::Engine& engine,
    const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
    double t, bool time_dep)
{
  auto mesh = u[0].get().function_space()->mesh();
  engine.BeginStep();

  // Write time step information
  // _time_dep = true;
  adios2::Variable<double> time = DefineVariable<double>(io, "step");
  engine.Put<double>(time, t);

  // Write mesh to file
  _write_mesh(io, engine, *mesh);

  // Extract and write function data
  std::set<std::string> point_data;
  for (auto u_ : u)
  {
    assert(mesh == u_.get().function_space()->mesh());

    // NOTE: Currently CG-1 interpolation of data.
    auto function_data = u_.get().compute_point_values();
    std::uint32_t local_size = function_data.shape[0];
    std::uint32_t block_size = function_data.shape[1];
    // Extract real and imaginary parts
    std::vector<std::string> parts = {""};
    if constexpr (!std::is_scalar<Scalar>::value)
      parts = {"real", "imag"};

    // Write each real and imaginary part of the function
    const int rank = u_.get().function_space()->element()->value_rank();
    const std::uint32_t num_components = std::pow(3, rank);
    std::vector<double> out_data(num_components * local_size);
    for (const auto& part : parts)
    {
      std::string function_name = u_.get().name;
      if (part != "")
        function_name += "_" + part;
      adios2::Variable<double> local_output = DefineVariable<double>(
          io, function_name, {}, {}, {local_size, num_components});

      // Loop over components of each real and imaginary part
      for (size_t i = 0; i < local_size; ++i)
      {
        for (size_t j = 0; j < block_size; ++j)
        {
          if (part == "imag")
          {
            out_data[i * num_components + j]
                = std::imag(function_data.row(i)[j]);
          }
          else
          {
            out_data[i * num_components + j]
                = std::real(function_data.row(i)[j]);
          }
        }

        // Pad data to 3D if vector or tensor data
        for (size_t j = block_size; j < num_components; ++j)
          out_data[i * num_components + j] = 0;
      }
      point_data.insert(function_name);

      // To reuse out_data, we use sync mode here
      engine.Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
    }
  }

  // Check if VTKScheme exists, and if so, check that we are only adding
  // values already existing
  std::string vtk_scheme = create_vtk_schema(point_data, {}, time_dep);
  // If writing to file set vtk scheme as current
  // if (_vtk_scheme.empty())
  //   _vtk_scheme = vtk_scheme;
  // if (vtk_scheme != _vtk_scheme)
  // {
  //   throw std::runtime_error(
  //       "Have to write the same functions to file for each "
  //       "time step");
  // }

  DefineAttribute<std::string>(io, "vtk.xml", vtk_scheme);
  engine.EndStep();
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
ADIOS2File::ADIOS2File(MPI_Comm comm, const std::string& filename,
                       const std::string& mode)
    : _adios(std::make_unique<adios2::ADIOS>(comm))
{
  _io = std::make_unique<adios2::IO>(_adios->DeclareIO("ADIOS2 DOLFINx IO"));
  _io->SetEngine("BPFile");

  if (mode == "a")
  {
    // FIXME: Remove this when is resolved
    // https://github.com/ornladios/ADIOS2/issues/2482
    _io->SetParameter("AggregatorRatio", "1");
  }

  adios2::Mode file_mode = string_to_mode(mode);
  _engine = std::make_unique<adios2::Engine>(_io->Open(filename, file_mode));
}
//-----------------------------------------------------------------------------
ADIOS2File::~ADIOS2File() { close(); };
//-----------------------------------------------------------------------------
void ADIOS2File::close()
{
  assert(_engine);

  // This looks a bit odd because ADIOS2 uses `operator bool()` to test
  // of the engine is open
  if (*_engine)
  {
    // Add create_vtk_schema if it hasn't been previously added (for
    // instance when writing meshes)
    if (_vtk_scheme.empty())
    {
      DefineAttribute<std::string>(*_io, "vtk.xml",
                                   create_vtk_schema({}, {}, _time_dep));
    }
    _engine->Close();
  }
}
//-----------------------------------------------------------------------------
void ADIOS2File::write_meshtags(const mesh::MeshTags<std::int32_t>& meshtag)
{
  // NOTE: CellData cannot be visualized, see:
  // https://gitlab.kitware.com/vtk/vtk/-/merge_requests/7401

  /// Get topology of geometry
  const int dim = meshtag.dim();
  auto mesh = meshtag.mesh();

  array2d<std::int32_t> geometry_entities
      = entities_to_geometry(*mesh, dim, meshtag.indices(), false);
  auto x_map = mesh->geometry().index_map();
  const std::uint32_t num_elements = meshtag.indices().size();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();

  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      *_io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = DefineVariable<std::uint32_t>(
      *_io, "NumberOfEntities", {adios2::LocalValueDim});

  std::vector<int32_t> cells(num_elements);
  std::iota(cells.begin(), cells.end(), 0);
  const std::uint32_t num_nodes = geometry_entities.shape[1];
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(*_io, "geometry", {}, {}, {num_vertices, 3});
  mesh::CellType cell_type
      = mesh::cell_entity_type(mesh->topology().cell_type(), dim);
  std::vector<std::uint8_t> map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(cell_type, num_nodes));

  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (cell_type == dolfinx::mesh::CellType::hexahedron and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }
  adios2::Variable<std::uint64_t> local_topology
      = DefineVariable<std::uint64_t>(*_io, "connectivity", {}, {},
                                      {num_elements, num_nodes + 1});
  std::vector<std::uint64_t> vtk_topology(num_elements * (num_nodes + 1));
  int topology_offset = 0;
  for (size_t c = 0; c < geometry_entities.shape[0]; ++c)
  {
    auto x_dofs = geometry_entities.row(c);
    vtk_topology[topology_offset++] = x_dofs.size();
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      vtk_topology[topology_offset++] = x_dofs[map[i]];
  }

  // Add element cell types
  const uint32_t vtk_type = dolfinx::io::cells::get_vtk_cell_type(*mesh, dim);
  adios2::Variable<std::uint32_t> cell_types
      = DefineVariable<std::uint32_t>(*_io, "types");

  // Create attribute for meshtags data
  adios2::Variable<std::int32_t> mesh_tags
      = DefineVariable<std::int32_t>(*_io, "MeshTag", {}, {}, {num_elements});
  _engine->Put<std::int32_t>(mesh_tags, meshtag.values().data());
  std::set<std::string> cell_data = {"MeshTag"};

  // Start writer for given function
  _engine->Put<std::uint32_t>(vertices, num_vertices);
  _engine->Put<std::uint32_t>(elements, num_elements);
  _engine->Put<std::uint32_t>(cell_types, vtk_type);
  _engine->Put<double>(local_geometry, mesh->geometry().x().data());
  _engine->Put<std::uint64_t>(local_topology, vtk_topology.data());
  _engine->PerformPuts();
  std::string _vtk_scheme = create_vtk_schema({}, cell_data, _time_dep);
  DefineAttribute<std::string>(*_io, "vtk.xml", _vtk_scheme);
}
//-----------------------------------------------------------------------------
void ADIOS2File::write_mesh(const mesh::Mesh& mesh)
{
  assert(_io);
  assert(_engine);
  _write_mesh(*_io, *_engine, mesh);
}
//-----------------------------------------------------------------------------
void ADIOS2File::write_function(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double t)
{
  bool compute_at_nodes = false;

  // Can only write one mesh to file at the time if using higher order
  // visualization
  if (u.size() > 1)
    compute_at_nodes = true;
  else
  {
    std::shared_ptr<const fem::FiniteElement> element
        = u[0].get().function_space()->element();
    assert(element);
    if (is_cellwise_constant(*element))
      throw std::runtime_error("Cell-wise constants not currently supported");

    std::array<std::string, 2> elements
        = {"Lagrange", "Discontinuous Lagrange"};
    if (std::find(elements.begin(), elements.end(), element->family())
        == elements.end())
    {
      compute_at_nodes = true;
    }
  }

  if (compute_at_nodes)
    _write_function_at_nodes<double>(*_io, *_engine, u, t, _time_dep);
  else
    _write_lagrange_function<double>(*_io, *_engine, u[0], t, _time_dep);
}
//-----------------------------------------------------------------------------
void ADIOS2File::write_function(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
    double t)
{
  bool compute_at_nodes = false;

  // Can only write one mesh to file at the time if using higher order
  // visualization
  if (u.size() > 1)
    compute_at_nodes = true;
  else
  {
    std::shared_ptr<const fem::FiniteElement> element
        = u[0].get().function_space()->element();
    assert(element);
    if (is_cellwise_constant(*element))
      throw std::runtime_error("Cell-wise constants not currently supported");

    std::array<std::string, 2> elements
        = {"Lagrange", "Discontinuous Lagrange"};
    if (std::find(elements.begin(), elements.end(), element->family())
        == elements.end())
    {
      compute_at_nodes = true;
    }
  }

  if (compute_at_nodes)
    _write_function_at_nodes<std::complex<double>>(*_io, *_engine, u, t,
                                                   _time_dep);
  else
    _write_lagrange_function<std::complex<double>>(*_io, *_engine, u[0], t,
                                                   _time_dep);
}
//-----------------------------------------------------------------------------
#endif