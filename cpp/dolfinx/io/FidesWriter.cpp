// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "FidesWriter.h"
#include "VTXWriter.h"
#include "adios2_utils.h"
#include <adios2.h>
#include <algorithm>
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
                      std::shared_ptr<const mesh::Mesh> mesh)
{
  // "Put" geometry data
  std::shared_ptr<const common::IndexMap> x_map = mesh->geometry().index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = adios2_utils::define_variable<double>(io, "points", {}, {},
                                              {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh->geometry().x().data());

  // Get topological dimenson, number of cells and number of 'nodes' per
  // cell
  // FIXME: Use better way to get number of nodes
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local();
  const int num_nodes = mesh->geometry().dofmap().num_links(0);

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
template <typename T>
void initialize_function_attributes(
    adios2::IO& io,
    const std::vector<std::shared_ptr<const fem::Function<T>>>& u)
{
  // Array of function (name, cell association types) for each function added to
  // the file
  std::vector<std::array<std::string, 2>> u_data;
  if constexpr (std::is_scalar<T>::value)
  {
    std::for_each(u.begin(), u.end(),
                  [&](std::shared_ptr<const fem::Function<T>> u) {
                    u_data.push_back({u->name, "points"});
                  });
  }
  else
  {
    const std::array<std::string, 2> parts = {"real", "imag"};
    std::for_each(u.begin(), u.end(),
                  [&](std::shared_ptr<const fem::Function<T>> u)
                  {
                    for (auto part : parts)
                      u_data.push_back({u->name + "_" + part, "points"});
                  });
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
Adios2Writer::Adios2Writer(MPI_Comm comm, const std::string& filename,
                           const std::string& tag)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, adios2::Mode::Write)))
{
  _io->SetEngine("BPFile");
}

//-----------------------------------------------------------------------------
Adios2Writer::Adios2Writer(MPI_Comm comm, const std::string& filename,
                           const std::string& tag,
                           std::shared_ptr<const mesh::Mesh> mesh)
    : Adios2Writer(comm, filename, tag)
{
  _mesh = mesh;
}

//-----------------------------------------------------------------------------
Adios2Writer::Adios2Writer(
    MPI_Comm comm, const std::string& filename, const std::string& tag,
    const std::vector<std::shared_ptr<const fem::Function<double>>>& u)
    : Adios2Writer(comm, filename, tag)
{
  _u = u;
}

//-----------------------------------------------------------------------------
Adios2Writer::~Adios2Writer() { close(); }
//-----------------------------------------------------------------------------
void Adios2Writer::close()
{
  assert(_engine);
  // This looks a bit odd because ADIOS2 uses `operator bool()` to
  // test if the engine is open
  if (*_engine)
    _engine->Close();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::string& filename,
                         std::shared_ptr<const mesh::Mesh> mesh)
    : Adios2Writer(comm, filename, "Fides mesh writer", mesh)
{
  assert(_io);
  assert(mesh);
  initialize_mesh_attributes(*_io, *mesh);
}
//-----------------------------------------------------------------------------
/// Initialize a FIDES writer for writing a list of functions to file
/// @param[in] comm The MPI communicator
/// @param[in] filename The filename of the output file
/// @param[in] functions The list of functions
FidesWriter::FidesWriter(
    MPI_Comm comm, const std::string& filename,
    const std::vector<std::shared_ptr<const fem::Function<double>>>& u)
    : Adios2Writer(comm, filename, "Fides function writer", u)
{
  assert(!u.empty());
  // _mesh = functions[0]->function_space()->mesh();

  // Check that mesh is the same for all functions
  // for (std::size_t i = 1; i < functions.size(); i++)
  //   assert(_mesh == functions[i]->function_space()->mesh());

  auto mesh = u[0]->function_space()->mesh();
  assert(mesh);
  initialize_mesh_attributes(*_io, *mesh);
  initialize_function_attributes<double>(*_io, u);
}
// //-----------------------------------------------------------------------------
// /// Initialize a FIDES writer for writing a list of functions to file
// /// @param[in] comm The MPI communicator
// /// @param[in] filename The filename of the output file
// /// @param[in] functions The list of functions
// FidesWriter::FidesWriter(
//     MPI_Comm comm, const std::string& filename,
//     const std::vector<
//         std::shared_ptr<const fem::Function<std::complex<double>>>>&
//         functions)
//     : _adios(std::make_unique<adios2::ADIOS>(comm)),
//       _io(std::make_unique<adios2::IO>(
//           _adios->DeclareIO("Fides function writer"))),
//       _engine(std::make_unique<adios2::Engine>(
//           _io->Open(filename, adios2::Mode::Write))),
//       _mesh(), _functions(), _complex_functions(functions)
// {
//   _io->SetEngine("BPFile");

//   // Check that mesh is the same for all functions
//   assert(functions.size() >= 1);
//   _mesh = functions[0]->function_space()->mesh();
//   for (std::size_t i = 1; i < functions.size(); i++)
//     assert(_mesh == functions[i]->function_space()->mesh());
//   _initialize_mesh_attributes(*_io, _mesh);
//   _initialize_function_attributes<std::complex<double>>(*_io,
//                                                         _complex_functions);
// }

//-----------------------------------------------------------------------------
void FidesWriter::write(double t)
{
  assert(_io);
  assert(_engine);

  _engine->BeginStep();
  adios2::Variable<double> var_step
      = adios2_utils::define_variable<double>(*_io, "step");
  _engine->Put<double>(var_step, t);

  // TODO: clarify the below 'note'. What impact does it have on code/user?
  // NOTE: Mesh can only be written to file once

  if (_mesh)
  {
    assert(_u.empty());
    write_fides_mesh(*_io, *_engine, _mesh);
  }
  else
  {
    throw std::runtime_error(
        "Function output to Fides/ADIOS2 not yet supported.");
  }

  // // Write real valued functions to file
  // std::for_each(
  //     _functions.begin(), _functions.end(),
  //     [&](std::shared_ptr<const fem::Function<double>> u)
  //     { adios2_utils::write_function_at_nodes<double>(*_io, *_engine, u); });

  // // Write complex valued functions to file
  // std::for_each(
  //     _complex_functions.begin(), _complex_functions.end(),
  //     [&](std::shared_ptr<const fem::Function<std::complex<double>>> u)
  //     {
  //       adios2_utils::write_function_at_nodes<std::complex<double>>(
  //           *_io, *_engine, u);
  //     });

  _engine->EndStep();
}
//-----------------------------------------------------------------------------

#endif