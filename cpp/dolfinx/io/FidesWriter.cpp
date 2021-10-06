// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "FidesWriter.h"
#include "adios2_utils.h"
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
      = adios2_utils::define_variable<double>(io, "points", {}, {},
                                              {num_vertices, 3});
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
  adios2::Variable<std::int64_t> local_topology
      = adios2_utils::define_variable<std::int64_t>(
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
  adios2_utils::define_attribute<std::string>(io, "Fides_Data_Model",
                                              "unstructured_single");

  // Define FIDES attributes pointing to ADIOS2 Variables for geometry
  // and topology
  adios2_utils::define_attribute<std::string>(io, "Fides_Coordinates_Variable",
                                              "points");
  adios2_utils::define_attribute<std::string>(io, "Fides_Connecticity_Variable",
                                              "connectivity");

  std::string cell_type = to_fides_cell(mesh.topology().cell_type());
  adios2_utils::define_attribute<std::string>(io, "Fides_Cell_Type", cell_type);
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
                           const std::string& tag)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, adios2::Mode::Write)))
{
  _io->SetEngine("BPFile");
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::string& filename,
                           const std::string& tag,
                           std::shared_ptr<const mesh::Mesh> mesh)
    : ADIOS2Writer(comm, filename, tag)
{
  _mesh = mesh;
}
//-----------------------------------------------------------------------------
ADIOS2Writer::ADIOS2Writer(
    MPI_Comm comm, const std::string& filename, const std::string& tag,
    const std::vector<std::variant<std::shared_ptr<const ADIOS2Writer::U0>,
                                   std::shared_ptr<const ADIOS2Writer::U1>>>& u)
    : ADIOS2Writer(comm, filename, tag)
{
  // Extract mesh from first function
  assert(!u.empty());
  if (auto v = std::get_if<std::shared_ptr<const U0>>(&u[0]))
    _mesh = (*v)->function_space()->mesh();
  else if (auto v = std::get_if<std::shared_ptr<const U1>>(&u[0]))
    _mesh = (*v)->function_space()->mesh();

  _u = u;
  for (auto& v : u)
  {
    if (auto _v = std::get_if<std::shared_ptr<const U0>>(&v))
    {
      assert(_mesh == (*_v)->function_space()->mesh());
      Ur.push_back(**_v);
    }
    else if (auto _v = std::get_if<std::shared_ptr<const U1>>(&v))
    {
      assert(_mesh == (*_v)->function_space()->mesh());
      Ur.push_back(**_v);
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
  adios2::Variable<double> var_step
      = adios2_utils::define_variable<double>(*_io, "step");
  _engine->Put<double>(var_step, t);

  write_fides_mesh(*_io, *_engine, *_mesh);

  for (auto& v : _u)
  {
    std::visit(
        overload{[&](const std::shared_ptr<const ADIOS2Writer::U0>& u) {
                   adios2_utils::write_function_at_nodes<double>(*_io, *_engine,
                                                                 *u);
                 },
                 [&](const std::shared_ptr<const ADIOS2Writer::U1>& u)
                 {
                   adios2_utils::write_function_at_nodes<std::complex<double>>(
                       *_io, *_engine, *u);
                 }},
        v);
  };

  _engine->EndStep();
}
//-----------------------------------------------------------------------------

#endif