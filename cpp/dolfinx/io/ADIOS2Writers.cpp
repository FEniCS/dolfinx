// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2Writers.h"
#include "pugixml.hpp"
#include <adios2.h>
#include <algorithm>
#include <complex>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
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
adios2::Attribute<T> define_attribute(adios2::IO& io, const std::string& name,
                                      const T& value,
                                      const std::string& var_name = "",
                                      const std::string& separator = "/")
{
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
    return attr;
  else
    return io.DefineAttribute<T>(name, value, var_name, separator);
}

/// Safe definition of a variable. First check if it has already been
/// defined and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> define_variable(adios2::IO& io, const std::string& name,
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

/// Write function (real or complex) ADIOS2. Data is padded to be three
/// dimensional if vector and 9 dimensional if tensor
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function to write
template <typename Scalar>
void write_function_at_nodes(adios2::IO& io, adios2::Engine& engine,
                             const fem::Function<Scalar>& u)
{
  auto function_data = u.compute_point_values();
  std::uint32_t local_size = function_data.shape(0);
  std::uint32_t block_size = function_data.shape(1);

  // Extract real and imaginary parts
  std::vector<std::string> parts = {""};
  if constexpr (!std::is_scalar<Scalar>::value)
    parts = {"real", "imag"};

  // Write each real and imaginary part of the function
  const int rank = u.function_space()->element()->value_rank();
  const std::uint32_t num_components = std::pow(3, rank);
  std::vector<double> out_data(num_components * local_size);
  for (const auto& part : parts)
  {
    std::string function_name = u.name;
    if (part != "")
      function_name += "_" + part;

    adios2::Variable<double> local_output = define_variable<double>(
        io, function_name, {}, {}, {local_size, num_components});

    // Loop over components of each real and imaginary part
    for (size_t i = 0; i < local_size; ++i)
    {
      auto data_row = xt::row(function_data, i);
      for (size_t j = 0; j < block_size; ++j)
      {
        if (part == "imag")
          out_data[i * num_components + j] = std::imag(data_row[j]);
        else
          out_data[i * num_components + j] = std::real(data_row[j]);
      }

      // Pad data to 3D if vector or tensor data
      for (size_t j = block_size; j < num_components; ++j)
        out_data[i * num_components + j] = 0;
    }

    // To reuse out_data, we use sync mode here
    engine.Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
  }
}
//-----------------------------------------------------------------------------
template <typename Scalar>
void write_lagrange_function(adios2::IO& io, adios2::Engine& engine,
                             const fem::Function<Scalar>& u)
{
  auto V = u.function_space();
  auto mesh = u.function_space()->mesh();
  std::string family = V->element()->family();

  xt::xtensor<double, 2> geometry = V->tabulate_dof_coordinates(false);
  const std::uint32_t num_dofs = geometry.shape(0);
  const std::uint32_t num_elements
      = mesh->topology().index_map(mesh->topology().dim())->size_local();

  // Create permutation from DOLFINx dof ordering to VTK
  std::shared_ptr<const fem::DofMap> dofmap = V->dofmap();
  const std::uint32_t num_nodes = dofmap->cell_dofs(0).size();
  std::vector<std::uint8_t> map = dolfinx::io::cells::transpose(
      io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));

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
      = define_variable<double>(io, "geometry", {}, {}, {num_dofs, 3});
  adios2::Variable<std::uint64_t> local_topology
      = define_variable<std::uint64_t>(io, "connectivity", {}, {},
                                       {num_elements, num_nodes + 1});
  adios2::Variable<std::uint32_t> cell_type
      = define_variable<std::uint32_t>(io, "types");
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = define_variable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});

  // Start writer and insert mesh information
  engine.Put<std::uint32_t>(vertices, num_dofs);
  engine.Put<std::uint32_t>(elements, num_elements);
  engine.Put<std::uint32_t>(cell_type, dolfinx::io::cells::get_vtk_cell_type(
                                           *mesh, mesh->topology().dim()));
  engine.Put<double>(local_geometry, geometry.data());
  engine.Put<std::uint64_t>(local_topology, vtk_topology.data());
  engine.PerformPuts();

  // Get function data array and information about layout
  std::shared_ptr<const la::Vector<Scalar>> function_vector = u.x();
  auto function_data = xt::adapt(function_vector->array());
  const int rank = u.function_space()->element()->value_rank();
  const std::uint32_t num_components = std::pow(3, rank);
  const std::uint32_t local_size = geometry.shape(0);
  const std::uint32_t block_size = dofmap->index_map_bs();
  std::vector<double> out_data(num_components * local_size);

  // Write each real and imaginary part of the function
  std::vector<std::string> parts = {""};
  if constexpr (!std::is_scalar<Scalar>::value)
    parts = {"real", "imag"};
  for (const auto& part : parts)
  {
    // Extract real or imaginary part
    xt::xtensor<double, 1> part_data;
    if (part == "imag")
      part_data = xt::imag(function_data);
    else
      part_data = xt::real(function_data);

    std::string function_name = u.name;
    if (part != "")
      function_name += "_" + part;
    for (size_t i = 0; i < local_size; ++i)
    {
      for (size_t j = 0; j < block_size; ++j)
        out_data[i * num_components + j] = part_data[i * block_size + j];

      // Pad data to 3D if vector or tensor data
      for (size_t j = block_size; j < num_components; ++j)
        out_data[i * num_components + j] = 0;
    }

    // To reuse out_data, we use sync mode here
    adios2::Variable<double> local_output = define_variable<double>(
        io, function_name, {}, {}, {local_size, num_components});
    engine.Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
  }
}
//-----------------------------------------------------------------------------
void _write_mesh(adios2::IO& io, adios2::Engine& engine, const mesh::Mesh& mesh)
{
  // Put geometry
  std::shared_ptr<const common::IndexMap> x_map = mesh.geometry().index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = define_variable<double>(io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh.geometry().x().data());

  // Put number of nodes. The mesh data is written with local indices. Therefore
  // we need the ghost vertices
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  // Add cell metadata
  const int tdim = mesh.topology().dim();
  const std::uint32_t num_cells = mesh.topology().index_map(tdim)->size_local();
  adios2::Variable<std::uint32_t> cell_variable
      = define_variable<std::uint32_t>(io, "NumberOfCells",
                                       {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_variable, num_cells);

  // Add cell-type
  adios2::Variable<std::uint32_t> celltype_variable
      = define_variable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(celltype_variable,
                            cells::get_vtk_cell_type(mesh, tdim));

  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);

  // // Extract mesh 'nodes'
  // Output is written as [N0 v0_0 .... v0_N0 N1 v1_0 .... v1_N1 ....]
  xt::xtensor<std::uint64_t, 2> topology({num_cells, num_nodes + 1});
  xt::view(topology, xt::all(), xt::xrange(std::size_t(1), topology.shape(1)))
      = io::extract_vtk_connectivity(mesh);
  xt::view(topology, xt::all(), 0) = num_nodes;

  // Put topology (nodes)
  adios2::Variable<std::uint64_t> local_topology
      = define_variable<std::uint64_t>(io, "connectivity", {}, {},
                                       {num_cells, num_nodes + 1});
  engine.Put<std::uint64_t>(local_topology, topology.data());
  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

// Extract name of functions and split into real and imaginary component
std::vector<std::string> extract_function_names(
    const std::vector<std::variant<std::shared_ptr<const ADIOS2Writer::U0>,
                                   std::shared_ptr<const ADIOS2Writer::U1>>>& u)
{
  std::vector<std::string> names;
  for (auto& v : u)
  {
    std::visit(
        overload{[&names](const std::shared_ptr<const ADIOS2Writer::U0>& u)
                 { names.push_back(u->name); },
                 [&names](const std::shared_ptr<const ADIOS2Writer::U1>& u)
                 { names.push_back(u->name); }},
        v);
  };

  return names;
}
//-----------------------------------------------------------------------------
bool is_cellwise_constant(const fem::FiniteElement& element)
{
  std::string family = element.family();
  int num_nodes_per_dim = element.space_dimension() / element.block_size();
  return num_nodes_per_dim == 1;
}
//-----------------------------------------------------------------------------

// Create VTK xml scheme to be interpreted by the Paraview VTKWriter
// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
std::string create_vtk_schema(const std::vector<std::string>& point_data,
                              const std::vector<std::string>& cell_data)
{
  // Create XML
  pugi::xml_document xml_schema;
  pugi::xml_node vtk_node = xml_schema.append_child("VTKFile");
  vtk_node.append_attribute("type") = "UnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  vtk_node.append_attribute("byte_order") = "LittleEndian";
  pugi::xml_node unstructured = vtk_node.append_child("UnstructuredGrid");
  pugi::xml_node piece = unstructured.append_child("Piece");

  // Add mesh attributes
  piece.append_attribute("NumberOfPoints") = "NumberOfNodes";
  piece.append_attribute("NumberOfCells") = "NumberOfCells";

  // Add point information
  pugi::xml_node xml_geometry = piece.append_child("Points");
  pugi::xml_node xml_vertices = xml_geometry.append_child("DataArray");
  xml_vertices.append_attribute("Name") = "geometry";

  // Add topology pointers
  pugi::xml_node xml_topology = piece.append_child("Cells");
  xml_topology.append_child("DataArray").append_attribute("Name")
      = "connectivity";
  xml_topology.append_child("DataArray").append_attribute("Name") = "types";

  // If we have any point data to write to file
  pugi::xml_node xml_pointdata = piece.append_child("PointData");

  // Stepping info for time dependency
  pugi::xml_node item = xml_pointdata.append_child("DataArray");
  item.append_attribute("Name") = "TIME";
  item.append_child(pugi::node_pcdata).set_value("step");

  // Append point data to VTK Schema
  for (auto name : point_data)
  {
    pugi::xml_node item = xml_pointdata.append_child("DataArray");
    item.append_attribute("Name") = name.c_str();
  }

  // Append cell data
  pugi::xml_node xml_celldata = piece.append_child("CellData");
  for (auto& name : cell_data)
  {
    pugi::xml_node item = xml_celldata.append_child("DataArray");
    item.append_attribute("Name") = name.c_str();
  }

  std::stringstream ss;
  xml_schema.save(ss, "  ");
  return ss.str();
}
//-----------------------------------------------------------------------------

/// Convert DOLFINx CellType to FIDES CellType
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

/// Put mesh geometry and connectivity for FIDES
/// @param[in] io The ADIOS2 IO
/// @param[in] engine The ADIOS2 engine
/// @param[in] mesh The mesh
void write_fides_mesh(adios2::IO& io, adios2::Engine& engine,
                      const mesh::Mesh& mesh)
{
  // "Put" geometry data
  std::shared_ptr<const common::IndexMap> x_map = mesh.geometry().index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = define_variable<double>(io, "points", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh.geometry().x().data());

  // Get topological dimenson, number of cells and number of 'nodes' per
  // cell
  // FIXME: Use better way to get number of nodes
  const int tdim = mesh.topology().dim();
  const std::int32_t num_cells = mesh.topology().index_map(tdim)->size_local();
  const int num_nodes = mesh.geometry().dofmap().num_links(0);

  // Compute the mesh 'VTK' connectivity  and "put" result in the ADIOS2
  // file
  xt::xtensor<std::int64_t, 2> topology = extract_vtk_connectivity(mesh);
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {std::size_t(num_cells * num_nodes)});
  engine.Put<std::int64_t>(local_topology, topology.data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

/// Initialize mesh related attributes for the ADIOS2 file used in FIDES
/// @param[in] io The ADIOS2 IO
/// @param[in] mesh The mesh
void initialize_mesh_attributes(adios2::IO& io, const mesh::Mesh& mesh)
{
  // NOTE: If we start using mixed element types, we can change
  // data-model to "unstructured"
  define_attribute<std::string>(io, "Fides_Data_Model", "unstructured_single");

  // Define FIDES attributes pointing to ADIOS2 Variables for geometry
  // and topology
  define_attribute<std::string>(io, "Fides_Coordinates_Variable", "points");
  define_attribute<std::string>(io, "Fides_Connecticity_Variable",
                                "connectivity");

  std::string cell_type = to_fides_cell(mesh.topology().cell_type());
  define_attribute<std::string>(io, "Fides_Cell_Type", cell_type);
}
//-----------------------------------------------------------------------------

/// Initialize function related attributes for the ADIOS2 file used in
/// FIDES
/// @param[in] io The ADIOS2 IO
/// @param[in] functions The list of functions
void initialize_function_attributes(adios2::IO& io, const ADIOS2Writer::U& u)
{
  // Array of function (name, cell association types) for each function added to
  // the file
  std::vector<std::array<std::string, 2>> u_data;
  for (auto& _u : u)
  {
    if (auto v = std::get_if<std::shared_ptr<const fem::Function<double>>>(&_u))
      u_data.push_back({(*v)->name, "points"});
    else if (auto v = std::get_if<
                 std::shared_ptr<const fem::Function<std::complex<double>>>>(
                 &_u))
    {
      for (auto part : {"real", "imag"})
        u_data.push_back({(*v)->name + "_" + part, "points"});
    }
    else
    {
      throw std::runtime_error("Unsupported function.");
    }
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
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::string& filename,
                           const std::string& tag,
                           const std::shared_ptr<const mesh::Mesh>& mesh,
                           const U& u)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, adios2::Mode::Write))),
      _mesh(mesh), _u(u)
{
  _io->SetEngine("BPFile");
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::string& filename,
                           const std::string& tag,
                           std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, tag, mesh, {})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(
    MPI_Comm comm, const std::string& filename, const std::string& tag,
    const std::vector<std::variant<std::shared_ptr<const ADIOS2Writer::U0>,
                                   std::shared_ptr<const ADIOS2Writer::U1>>>& u)
    : ADIOS2Writer(comm, filename, tag, nullptr, u)
{
  // Extract mesh from first function
  assert(!u.empty());
  if (auto v = std::get_if<std::shared_ptr<const U0>>(&u[0]))
    _mesh = (*v)->function_space()->mesh();
  else if (auto v = std::get_if<std::shared_ptr<const U1>>(&u[0]))
    _mesh = (*v)->function_space()->mesh();

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    if (auto _v = std::get_if<std::shared_ptr<const U0>>(&v))
    {
      assert(_mesh == (*_v)->function_space()->mesh());
    }
    else if (auto _v = std::get_if<std::shared_ptr<const U1>>(&v))
    {
      assert(_mesh == (*_v)->function_space()->mesh());
    }
    else
      throw std::runtime_error("Unsupported function.");
  }
}
//-----------------------------------------------------------------------------
ADIOS2Writer::~ADIOS2Writer() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2Writer::close()
{
  assert(_engine);
  // This looks a bit odd because ADIOS2 uses `operator bool()` to
  // test if the engine is open
  if (*_engine)
    _engine->Close();
}
//-----------------------------------------------------------------------------
xt::xtensor<std::int64_t, 2>
io::extract_vtk_connectivity(const mesh::Mesh& mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }
  // Extract mesh 'nodes'
  const int tdim = mesh.topology().dim();
  const std::uint32_t num_cells = mesh.topology().index_map(tdim)->size_local();

  // Write mesh connectivity
  xt::xtensor<std::int64_t, 2> topology({num_cells, num_nodes});
  for (size_t c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      topology(c, i) = x_dofs[map[i]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::string& filename,
                         std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, "Fides mesh writer", mesh)
{
  assert(_io);
  assert(mesh);
  initialize_mesh_attributes(*_io, *mesh);
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::string& filename,
                         const ADIOS2Writer::U& u)
    : ADIOS2Writer(comm, filename, "Fides function writer", u)
{
  assert(!u.empty());
  const mesh::Mesh* mesh = nullptr;
  if (auto v = std::get_if<std::shared_ptr<const ADIOS2Writer::U0>>(&u[0]))
    mesh = (*v)->function_space()->mesh().get();
  else if (auto v = std::get_if<std::shared_ptr<const ADIOS2Writer::U1>>(&u[0]))
    mesh = (*v)->function_space()->mesh().get();
  else
    throw std::runtime_error("Unsupported function.");

  assert(mesh);
  initialize_mesh_attributes(*_io, *mesh);
  initialize_function_attributes(*_io, u);
}
//-----------------------------------------------------------------------------
void FidesWriter::write(double t)
{
  assert(_io);
  assert(_engine);

  _engine->BeginStep();
  adios2::Variable<double> var_step = define_variable<double>(*_io, "step");
  _engine->Put<double>(var_step, t);

  write_fides_mesh(*_io, *_engine, *_mesh);

  for (auto& v : _u)
  {
    std::visit(overload{[&](const std::shared_ptr<const ADIOS2Writer::U0>& u) {
                          write_function_at_nodes<double>(*_io, *_engine, *u);
                        },
                        [&](const std::shared_ptr<const ADIOS2Writer::U1>& u) {
                          write_function_at_nodes<std::complex<double>>(
                              *_io, *_engine, *u);
                        }},
               v);
  };

  _engine->EndStep();
}
//-----------------------------------------------------------------------------
VTXWriter::VTXWriter(MPI_Comm comm, const std::string& filename,
                     std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, "VTX mesh writer", mesh)
{

  // Define VTK scheme attribute for mesh
  std::string vtk_scheme = create_vtk_schema({}, {});
  define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);

  // Set compute at nodes true since we want to write mesh at each timestep
  _write_mesh_data = true;
}
//-----------------------------------------------------------------------------
VTXWriter::VTXWriter(
    MPI_Comm comm, const std::string& filename,
    const std::vector<std::variant<
        std::shared_ptr<const fem::Function<double>>,
        std::shared_ptr<const fem::Function<std::complex<double>>>>>& u)
    : ADIOS2Writer(comm, filename, "VTX function writer", u)
{
  // Can only write one mesh to file at the time if using higher order
  // visualization

  // Only Lagrange and discontinuous Lagrange can be written at function space
  // dof coordinates
  _write_mesh_data = false;
  if (u.size() > 1)
    _write_mesh_data = true;
  else
  {
    // Extract element from first function
    assert(!u.empty());
    std::shared_ptr<const fem::FiniteElement> element;
    if (auto v = std::get_if<std::shared_ptr<const U0>>(&u[0]))
      element = (*v)->function_space()->element();
    else if (auto v = std::get_if<std::shared_ptr<const U1>>(&u[0]))
      element = (*v)->function_space()->element();
    assert(element);
    if (is_cellwise_constant(*element))
      throw std::runtime_error("Cell-wise constants not currently supported");

    // FIXME: Should not use string checks
    std::array<std::string, 3> elements
        = {"Lagrange", "Discontinuous Lagrange", "DQ"};
    if (std::find(elements.begin(), elements.end(), element->family())
        == elements.end())
    {
      _write_mesh_data = true;
    }
  }

  // Define VTK scheme attribute for set of functions
  std::vector<std::string> names = extract_function_names(u);
  std::string vtk_scheme = create_vtk_schema(names, {});
  define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
}
//-----------------------------------------------------------------------------
void VTXWriter::write(double t)
{
  assert(_io);
  assert(_engine);
  adios2::Variable<double> var_step = define_variable<double>(*_io, "step");

  _engine->BeginStep();
  if (_write_mesh_data)
    _write_mesh(*_io, *_engine, *_mesh);

  _engine->Put<double>(var_step, t);

  // Write functions to file
  for (auto& v : _u)
  {
    std::visit(overload{[&](const std::shared_ptr<const ADIOS2Writer::U0>& u)
                        {
                          if (_write_mesh_data)
                          {
                            write_function_at_nodes<double>(*_io, *_engine, *u);
                          }
                          else
                            write_lagrange_function<double>(*_io, *_engine, *u);
                        },
                        [&](const std::shared_ptr<const ADIOS2Writer::U1>& u)
                        {
                          if (_write_mesh_data)
                          {
                            write_function_at_nodes<std::complex<double>>(
                                *_io, *_engine, *u);
                          }
                          else
                            write_lagrange_function<std::complex<double>>(
                                *_io, *_engine, *u);
                        }},
               v);
  };

  _engine->EndStep();
}

#endif