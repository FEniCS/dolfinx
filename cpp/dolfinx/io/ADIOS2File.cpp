// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2File.h"
#include "dolfinx/io/cells.h"
#include "dolfinx/mesh/utils.h"
#include "pugixml.hpp"
#include <adios2.h>
using namespace dolfinx;
using namespace dolfinx::io;
namespace
{
// Safe definition of an attribute (required for time dependent problems)
template <class T>
adios2::Attribute<T>
DefineAttribute(std::shared_ptr<adios2::IO> io, const std::string& attr_name,
                const T& value, const std::string& var_name = "",
                const std::string separator = "/")
{
  adios2::Attribute<T> attribute = io->InquireAttribute<T>(attr_name);
  if (attribute)
    return attribute;
  return io->DefineAttribute<T>(attr_name, value, var_name, separator);
}

// Safe definition of a variable (required for time dependent problems)
template <class T>
adios2::Variable<T> DefineVariable(std::shared_ptr<adios2::IO> io,
                                   const std::string& var_name,
                                   const adios2::Dims& shape = adios2::Dims(),
                                   const adios2::Dims& start = adios2::Dims(),
                                   const adios2::Dims& count = adios2::Dims())
{
  adios2::Variable<T> variable = io->InquireVariable<T>(var_name);
  if (variable)
  {
    if (variable.Count() != count
        && variable.ShapeID() == adios2::ShapeID::LocalArray)
      variable.SetSelection({start, count});
  }
  else
    variable = io->DefineVariable<T>(var_name, shape, start, count);

  return variable;
}

adios2::Mode string_to_mode(std::string mode)
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

} // namespace
//-----------------------------------------------------------------------------

ADIOS2File::ADIOS2File(MPI_Comm comm, std::string filename, std::string mode)
    : _adios(), _io(), _engine(), _vtk_scheme(), _mode(mode)
{
  _adios = std::make_shared<adios2::ADIOS>(comm);
  adios2::Mode file_mode = string_to_mode(mode);
  _io = std::make_shared<adios2::IO>(_adios->DeclareIO("ADIOS2 DOLFINx IO"));
  _io->SetEngine("BPFile");

  if (mode == "a")
  {
    // FIXME: Remove this when is resolved
    // https://github.com/ornladios/ADIOS2/issues/2482
    _io->SetParameter("AggregatorRatio", "1");
  }

  _engine = std::make_shared<adios2::Engine>(_io->Open(filename, file_mode));
}
//-----------------------------------------------------------------------------

ADIOS2File::~ADIOS2File() { close(); };
//-----------------------------------------------------------------------------

void ADIOS2File::close()
{
  if (*_engine)
  {
    // Add VTKSchema if it hasn't been previously added (for instance when
    // writing meshes)
    if (_vtk_scheme.empty())
    {
      DefineAttribute<std::string>(_io, "vtk.xml", VTKSchema({}, {}));
    }
    _engine->Close();
  }
}
//-----------------------------------------------------------------------------
/// Write meshtags to file
/// @param[in] meshtags
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
      _io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = DefineVariable<std::uint32_t>(
      _io, "NumberOfEntities", {adios2::LocalValueDim});

  std::vector<int32_t> cells(num_elements);
  std::iota(cells.begin(), cells.end(), 0);
  const std::uint32_t num_nodes = geometry_entities.shape[1];
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(_io, "geometry", {}, {}, {num_vertices, 3});
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
      = DefineVariable<std::uint64_t>(_io, "connectivity", {}, {},
                                      {num_elements, num_nodes + 1});
  std::vector<std::uint64_t> vtk_topology(num_elements * (num_nodes + 1));
  int topology_offset = 0;
  std::stringstream cc;

  for (size_t c = 0; c < geometry_entities.shape[0]; ++c)
  {
    auto x_dofs = geometry_entities.row(c);
    vtk_topology[topology_offset++] = x_dofs.size();
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      vtk_topology[topology_offset++] = x_dofs[map[i]];
    }
  }

  // Add element cell types
  const uint32_t vtk_type = dolfinx::io::cells::get_vtk_cell_type(*mesh, dim);
  adios2::Variable<std::uint32_t> cell_types
      = DefineVariable<std::uint32_t>(_io, "types");

  // Create attribute for meshtags data
  adios2::Variable<std::int32_t> mesh_tags
      = DefineVariable<std::int32_t>(_io, "MeshTag", {}, {}, {num_elements});
  _engine->Put<std::int32_t>(mesh_tags, meshtag.values().data());
  std::set<std::string> cell_data = {"MeshTag"};

  // Start writer for given function
  _engine->Put<std::uint32_t>(vertices, num_vertices);
  _engine->Put<std::uint32_t>(elements, num_elements);
  _engine->Put<std::uint32_t>(cell_types, vtk_type);
  _engine->Put<double>(local_geometry, mesh->geometry().x().data());
  _engine->Put<std::uint64_t>(local_topology, vtk_topology.data());
  _engine->PerformPuts();
  DefineAttribute<std::string>(_io, "vtk.xml", VTKSchema({}, cell_data));
}
//-----------------------------------------------------------------------------

void ADIOS2File::write_mesh(const mesh::Mesh& mesh)
{
  if (_mode == "a")
    throw std::runtime_error(
        "Cannot append functions to previously created file.");

  // Get some data about mesh
  auto top = mesh.topology();
  auto x_map = mesh.geometry().index_map();
  const int tdim = top.dim();

  // As the mesh data is written with local indices we need the ghost vertices
  const std::uint32_t num_elements = top.index_map(tdim)->size_local();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      _io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = DefineVariable<std::uint32_t>(
      _io, "NumberOfEntities", {adios2::LocalValueDim});

  // Extract geometry for all local cells
  std::vector<int32_t> cells(num_elements);
  std::iota(cells.begin(), cells.end(), 0);
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(_io, "geometry", {}, {}, {num_vertices, 3});

  // Get DOLFINx to VTK permuation
  // FIXME: Use better way to get number of nods
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector<std::uint8_t> map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  // Extract topology for all local cells
  // Output is written as [N0 v0_0 .... v0_N0 N1 v1_0 .... v1_N1 ....]
  adios2::Variable<std::uint64_t> local_topology
      = DefineVariable<std::uint64_t>(_io, "connectivity", {}, {},
                                      {num_elements, num_nodes + 1});
  std::vector<std::uint64_t> vtk_topology(num_elements * (num_nodes + 1));
  int topology_offset = 0;
  std::stringstream cc;

  for (size_t c = 0; c < num_elements; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    vtk_topology[topology_offset++] = x_dofs.size();
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      vtk_topology[topology_offset++] = x_dofs[map[i]];
    }
  }

  // Add element cell types
  adios2::Variable<std::uint32_t> cell_type
      = DefineVariable<std::uint32_t>(_io, "types");
  // Start writer for given function
  _engine->Put<std::uint32_t>(vertices, num_vertices);
  _engine->Put<std::uint32_t>(elements, num_elements);
  _engine->Put<std::uint32_t>(
      cell_type, dolfinx::io::cells::get_vtk_cell_type(mesh, tdim));
  _engine->Put<double>(local_geometry, mesh.geometry().x().data());
  _engine->Put<std::uint64_t>(local_topology, vtk_topology.data());
  _engine->PerformPuts();
}
//-----------------------------------------------------------------------------

template <typename Scalar>
void ADIOS2File::_write_function(
    const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
    double t)
{
  auto mesh = u[0].get().function_space()->mesh();
  _engine->BeginStep();

  // Write time step information
  _time_dep = true;
  adios2::Variable<double> time = DefineVariable<double>(_io, "step");
  _engine->Put<double>(time, t);

  // Write mesh to file
  write_mesh(*mesh);

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
          _io, function_name, {}, {}, {local_size, num_components});
      // Loop over components of each real and imaginary part
      for (size_t i = 0; i < local_size; ++i)
      {
        for (size_t j = 0; j < block_size; ++j)
          if (part == "imag")
            out_data[i * num_components + j]
                = std::imag(function_data.row(i)[j]);
          else
            out_data[i * num_components + j]
                = std::real(function_data.row(i)[j]);

        // Pad data to 3D if vector or tensor data
        for (size_t j = block_size; j < num_components; ++j)
        {
          out_data[i * num_components + j] = 0;
        }
      }
      point_data.insert(function_name);

      // To reuse out_data, we use sync mode here
      _engine->Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
    }
  }
  // Check if VTKScheme exists, and if so, check that we are only adding values
  // already existing
  std::string vtk_scheme = VTKSchema(point_data, {});
  // If writing to file set vtk scheme as current
  if (_vtk_scheme.empty())
    _vtk_scheme = vtk_scheme;
  if (vtk_scheme != _vtk_scheme)
  {
    throw std::runtime_error(
        "Have to write the same functions to file for each "
        "time step");
  }
  DefineAttribute<std::string>(_io, "vtk.xml", vtk_scheme);
  _engine->EndStep();
}
//-----------------------------------------------------------------------------

void ADIOS2File::write_function(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double t)
{
  _write_function<double>(u, t);
}
//-----------------------------------------------------------------------------

void ADIOS2File::write_function(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
    double t)
{
  _write_function<std::complex<double>>(u, t);
}
//-----------------------------------------------------------------------------

std::string ADIOS2File::VTKSchema(std::set<std::string> point_data,
                                  std::set<std::string> cell_data)
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

#endif