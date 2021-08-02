// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "FidesWriter.h"
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
// Safe definition of an attribute. First check if it has already been defined
// and return it. If not defined create new attribute.
template <class T>
adios2::Attribute<T> DefineAttribute(adios2::IO& io, const std::string& name,
                                     const T& value,
                                     const std::string& var_name = "",
                                     const std::string& separator = "/")
{
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
    return attr;
  else
    return io.DefineAttribute<T>(name, value, var_name, separator);
}
//-----------------------------------------------------------------------------
// Safe definition of a variable. First check if it has already been defined
// and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> DefineVariable(adios2::IO& io, const std::string& name,
                                   const adios2::Dims& shape = adios2::Dims(),
                                   const adios2::Dims& start = adios2::Dims(),
                                   const adios2::Dims& count = adios2::Dims())
{
  adios2::Variable<T> v = io.InquireVariable<T>(name);
  if (v)
  {
    if (v.Count() != count and v.ShapeID() == adios2::ShapeID::LocalArray)
      v.SetSelection({start, count});
  }
  else
    v = io.DefineVariable<T>(name, shape, start, count);

  return v;
}
//-----------------------------------------------------------------------------
constexpr adios2::Mode dolfinx_to_adios_mode(io::mode mode)
{
  switch (mode)
  {
  case io::mode::write:
    return adios2::Mode::Write;
  case io::mode::append:
    return adios2::Mode::Append;
  case io::mode::read:
    throw std::runtime_error("Unsupported file mode");
  //   return adios2::Mode::Read;
  default:
    throw std::runtime_error("Unknown file mode");
  }
}
//-----------------------------------------------------------------------------
void _write_mesh(adios2::IO& io, adios2::Engine& engine,
                 std::shared_ptr<const mesh::Mesh> mesh)
{
  // assert(_engine);
  // // ADIOS should handle mode checks, and if we need to we should get it
  // // from ADIOS - DOLFINx should not store the state
  // if (_engine->OpenMode() == adios2::Mode::Append)
  // {
  //   throw std::runtime_error(
  //       "Cannot append functions to previously created file.");
  // }

  // Put geometry
  std::shared_ptr<const common::IndexMap> x_map = mesh->geometry().index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "points", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh->geometry().x().data());

  // Extract topology (CG 1)
  const int tdim = mesh->topology().dim();
  const std::uint32_t num_cells
      = mesh->topology().index_map(tdim)->size_local();

  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      cells::perm_vtk(mesh->topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh->topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  // Extract mesh 'nodes'
  std::vector<std::int64_t> topology;
  topology.reserve(num_cells * num_nodes);
  for (size_t c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    std::transform(map.cbegin(), map.cend(), std::back_inserter(topology),
                   [&x_dofs](auto index) { return x_dofs[index]; });
  }

  // Put topology (nodes)
  adios2::Variable<std::int64_t> local_topology = DefineVariable<std::int64_t>(
      io, "connectivity", {}, {}, {num_cells * num_nodes});
  engine.Put<std::int64_t>(local_topology, topology.data());
  // Perform puts before going out of scope
  engine.PerformPuts();
}
//-----------------------------------------------------------------------------
template <typename T>
void _write_function(adios2::IO& io, adios2::Engine& engine,
                     std::reference_wrapper<const fem::Function<T>> u)
{

  const int rank = u.get().function_space()->element()->value_rank();
  const std::size_t value_size
      = u.get().function_space()->element()->value_size();
  if (rank > 1)
    throw std::runtime_error("Tensor output not implemented");

  // Determine number of components (1 for scalars, 3 for vectors, 9 for
  // tensors)
  const std::uint32_t num_components = std::pow(3, rank);

  // Create output array
  auto x_map = u.get().function_space()->mesh()->geometry().index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();

  // Compute point values
  xt::xtensor<T, 2> values = u.get().compute_point_values();

  // Pad vector data out to 3D if required
  if (rank == 1 and value_size == 2)
  {
    xt::xtensor<T, 2> values_pad
        = xt::zeros<T>({std::size_t(num_vertices), std::size_t(3)});
    xt::view(values_pad, xt::all(), xt::range(0, 2)) = values;
    values = values_pad;
  }
  else if (rank != 0 and !(rank == 1 and value_size == 3))
    throw std::runtime_error("Unsupported function type");

  // 'Put' data into ADIOS file
  if constexpr (std::is_scalar<T>::value)
  {
    // 'Put' array  (real)
    adios2::Variable<T> _u = DefineVariable<T>(io, u.get().name, {}, {},
                                               {num_vertices, num_components});
    engine.Put<T>(_u, values.data(), adios2::Mode::Sync);
  }
  else
  {
    // 'Put' array (imaginary)
    using Q = typename T::value_type;
    const std::array<std::string, 2> parts = {"real", "imag"};
    xt::xtensor<Q, 2> _values;
    for (auto part : parts)
    {
      // Extract real/imaginary parts
      if (part == "real")
        _values = xt::real(values);
      else if (part == "imag")
        _values = xt::imag(values);

      adios2::Variable<Q> _u
          = DefineVariable<Q>(io, u.get().name + "_" + part, {}, {},
                              {num_vertices, num_components});
      engine.Put<Q>(_u, _values.data(), adios2::Mode::Sync);
    }
  }
}

//-----------------------------------------------------------------------------
void _initialize_mesh_attributes(adios2::IO& io,
                                 std::shared_ptr<const mesh::Mesh> mesh)
{
  // NOTE: If we start using mixed element types, we can change
  // data-model to "unstructured"
  DefineAttribute<std::string>(io, "Fides_Data_Model", "unstructured_single");

  // Define FIDES attributes pointing to ADIOS2 Variables for geometry
  // and topology
  DefineAttribute<std::string>(io, "Fides_Coordinates_Variable", "points");
  DefineAttribute<std::string>(io, "Fides_Connecticity_Variable",
                               "connectivity");

  std::string cell_type = to_fides_cell(mesh->topology().cell_type());
  DefineAttribute<std::string>(io, "Fides_Cell_Type", cell_type);
}
//-----------------------------------------------------------------------------
template <typename T>
void _initialize_function_attributes(
    adios2::IO& io,
    const std::vector<std::reference_wrapper<const fem::Function<T>>>&
        functions)
{
  // Array of function (name, cell association types) for each function added to
  // the file
  std::vector<std::array<std::string, 2>> function_data;
  if constexpr (std::is_scalar<T>::value)
  {
    std::for_each(functions.begin(), functions.end(),
                  [&](const fem::Function<T>& u) {
                    function_data.push_back({u.name, "points"});
                  });
  }
  else
  {
    const std::array<std::string, 2> parts = {"real", "imag"};
    std::for_each(functions.begin(), functions.end(),
                  [&](const fem::Function<T>& u)
                  {
                    for (auto part : parts)
                      function_data.push_back({u.name + "_" + part, "points"});
                  });
  }
  // Write field associations to file
  if (adios2::Attribute<std::string> assc
      = io.InquireAttribute<std::string>("Fides_Variable_Associations");
      !assc)
  {
    std::vector<std::string> u_type;
    std::transform(function_data.cbegin(), function_data.cend(),
                   std::back_inserter(u_type), [](auto& f) { return f[1]; });
    io.DefineAttribute<std::string>("Fides_Variable_Associations",
                                    u_type.data(), u_type.size());
  }

  // Write field pointers to file
  if (adios2::Attribute<std::string> fields
      = io.InquireAttribute<std::string>("Fides_Variable_List");
      !fields)
  {
    std::vector<std::string> names;
    std::transform(function_data.cbegin(), function_data.cend(),
                   std::back_inserter(names), [](auto& f) { return f[0]; });
    io.DefineAttribute<std::string>("Fides_Variable_List", names.data(),
                                    names.size());
  }
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(MPI_Comm comm, const std::string& filename,
                         io::mode mode, std::shared_ptr<const mesh::Mesh> mesh)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO("Fides mesh writer"))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, dolfinx_to_adios_mode(mode)))),
      _mesh(mesh), _functions(), _complex_functions(), _mesh_written(false)
{
  _io->SetEngine("BPFile");

  // FIXME: Remove when https://github.com/ornladios/ADIOS2/issues/2482
  // is resolved
  if (mode == io::mode::append)
    _io->SetParameter("AggregatorRatio", "1");
  _initialize_mesh_attributes(*_io, mesh);
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(
    MPI_Comm comm, const std::string& filename, io::mode mode,
    const std::vector<std::reference_wrapper<const fem::Function<double>>>&
        functions)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(
          _adios->DeclareIO("Fides function writer"))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, dolfinx_to_adios_mode(mode)))),
      _mesh(), _functions(functions), _complex_functions(), _mesh_written(false)
{
  _io->SetEngine("BPFile");

  // FIXME: Remove when https://github.com/ornladios/ADIOS2/issues/2482
  // is resolved
  if (mode == io::mode::append)
    _io->SetParameter("AggregatorRatio", "1");

  // Check that mesh is the same for all functions
  assert(functions.size() >= 1);
  _mesh = functions[0].get().function_space()->mesh();
  for (std::size_t i = 1; i < functions.size(); i++)
    assert(_mesh == functions[i].get().function_space()->mesh());
  _initialize_mesh_attributes(*_io, _mesh);
  _initialize_function_attributes<double>(*_io, _functions);
}
//-----------------------------------------------------------------------------
FidesWriter::FidesWriter(
    MPI_Comm comm, const std::string& filename, io::mode mode,
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>&
        functions)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(
          _adios->DeclareIO("Fides function writer"))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, dolfinx_to_adios_mode(mode)))),
      _mesh(), _functions(), _complex_functions(functions)
{
  _io->SetEngine("BPFile");

  // FIXME: Remove when https://github.com/ornladios/ADIOS2/issues/2482
  // is resolved
  if (mode == io::mode::append)
    _io->SetParameter("AggregatorRatio", "1");

  // Check that mesh is the same for all functions
  assert(functions.size() >= 1);
  _mesh = functions[0].get().function_space()->mesh();
  for (std::size_t i = 1; i < functions.size(); i++)
    assert(_mesh == functions[i].get().function_space()->mesh());
  _initialize_mesh_attributes(*_io, _mesh);
  _initialize_function_attributes<std::complex<double>>(*_io,
                                                        _complex_functions);
}

//-----------------------------------------------------------------------------
FidesWriter::~FidesWriter() { close(); }
//-----------------------------------------------------------------------------
void FidesWriter::close()
{
  assert(_engine);

  // This looks a bit odd because ADIOS2 uses `operator bool()` to test
  // if the engine is open
  if (*_engine)
    _engine->Close();
}
//-----------------------------------------------------------------------------
void FidesWriter::write(double t)
{
  assert(_io);
  assert(_engine);
  _engine->BeginStep();
  adios2::Variable<double> var_step = DefineVariable<double>(*_io, "step");
  _engine->Put<double>(var_step, t);
  // NOTE: Mesh can only be written to file once
  if (!_mesh_written)
  {
    _write_mesh(*_io, *_engine, _mesh);
    // _mesh_written = true;
  }

  // Write real valued functions to file
  std::for_each(_functions.begin(), _functions.end(),
                [&](const fem::Function<double>& u)
                { _write_function<double>(*_io, *_engine, u); });

  // Write complex valued functions to file
  std::for_each(_complex_functions.begin(), _complex_functions.end(),
                [&](const fem::Function<std::complex<double>>& u)
                { _write_function<std::complex<double>>(*_io, *_engine, u); });
  _engine->EndStep();
}
//-----------------------------------------------------------------------------

#endif