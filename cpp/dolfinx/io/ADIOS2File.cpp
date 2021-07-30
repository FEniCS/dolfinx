// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2File.h"
#include <adios2.h>
#include <algorithm>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/Mesh.h>

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
// Safe definition of an attribute (required for time dependent problems)
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
// Safe definition of a variable (required for time dependent problems)
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
    return adios2::Mode::Read;
  }
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

  // NOTE: If we start using mixed element types, we can change
  // data-model to "unstructured"
  DefineAttribute<std::string>(io, "Fides_Data_Model", "unstructured_single");

  // Define FIDES attributes pointing to ADIOS2 Variables for geometry
  // and topology
  DefineAttribute<std::string>(io, "Fides_Coordinates_Variable", "points");
  DefineAttribute<std::string>(io, "Fides_Connecticity_Variable",
                               "connectivity");

  std::string cell_type = to_fides_cell(mesh.topology().cell_type());
  DefineAttribute<std::string>(io, "Fides_Cell_Type", cell_type);

  std::shared_ptr<const common::IndexMap> x_map = mesh.geometry().index_map();
  const int tdim = mesh.topology().dim();
  const std::uint32_t num_cells = mesh.topology().index_map(tdim)->size_local();

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

  // Put geometry
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "points", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh.geometry().x().data());

  engine.PerformPuts();
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
void _write_function(adios2::IO& io, adios2::Engine& engine,
                     std::reference_wrapper<const fem::Function<ScalarType>> u)
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
  xt::xtensor<ScalarType, 2> values = u.get().compute_point_values();

  // Pad vector data out to 3D if required
  if (rank == 1 and value_size == 2)
  {
    xt::xtensor<ScalarType, 2> values_pad
        = xt::zeros<ScalarType>({std::size_t(num_vertices), std::size_t(3)});
    xt::view(values_pad, xt::all(), xt::range(0, 2)) = values;
    values = values_pad;
  }
  else if (rank != 0 or !(rank == 1 and value_size == 3))
    throw std::runtime_error("Unsupported function type");

  // 'Put' data into ADIOS file
  if constexpr (std::is_scalar<ScalarType>::value)
  {
    // 'Put' array  (real)
    adios2::Variable<ScalarType> _u = DefineVariable<ScalarType>(
        io, u.get().name, {}, {}, {num_vertices, num_components});
    engine.Put<ScalarType>(_u, values.data());
    engine.PerformPuts();
  }
  else
  {
    // 'Put' array (imaginary)
    using T = typename ScalarType::value_type;
    const std::array<std::string, 2> parts = {"real", "imag"};
    xt::xtensor<T, 2> _values;
    for (auto part : parts)
    {
      // Extract real/imaginary parts
      if (part == "real")
        _values = xt::real(values);
      else if (part == "imag")
        _values = xt::imag(values);

      adios2::Variable<T> _u
          = DefineVariable<T>(io, u.get().name + "_" + part, {}, {},
                              {num_vertices, num_components});
      engine.Put<T>(_u, _values.data());
    }
  }
  engine.PerformPuts();
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
ADIOS2File::ADIOS2File(MPI_Comm comm, const std::string& filename,
                       io::mode mode)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(
          _adios->DeclareIO("ADIOS2-FIDES DOLFINx IO"))),
      _engine(std::make_unique<adios2::Engine>(
          _io->Open(filename, dolfinx_to_adios_mode(mode))))
{
  _io->SetEngine("BPFile");

  // FIXME: Remove when https://github.com/ornladios/ADIOS2/issues/2482
  // is resolved
  if (mode == io::mode::append)
  {
    _io->SetParameter("AggregatorRatio", "1");
  }
}
//-----------------------------------------------------------------------------
ADIOS2File::~ADIOS2File() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2File::close()
{
  assert(_engine);

  // This looks a bit odd because ADIOS2 uses `operator bool()` to test
  // if the engine is open
  if (*_engine)
  {
    // Write field associations to file
    if (adios2::Attribute<std::string> assc
        = _io->InquireAttribute<std::string>("Fides_Variable_Associations");
        !assc)
    {
      std::vector<std::string> u_type;
      std::transform(_function_data.cbegin(), _function_data.cend(),
                     std::back_inserter(u_type), [](auto& f) { return f[1]; });
      _io->DefineAttribute<std::string>("Fides_Variable_Associations",
                                        u_type.data(), u_type.size());
    }

    // Write field pointers to file
    if (adios2::Attribute<std::string> fields
        = _io->InquireAttribute<std::string>("Fides_Variable_List");
        !fields)
    {
      std::vector<std::string> names;
      std::transform(_function_data.cbegin(), _function_data.cend(),
                     std::back_inserter(names), [](auto& f) { return f[0]; });
      _io->DefineAttribute<std::string>("Fides_Variable_List", names.data(),
                                        names.size());
    }

    _engine->Close();
  }
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
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u)
{
  assert(_io);
  assert(_engine);
  std::for_each(u.begin(), u.end(),
                [&](const fem::Function<double>& u)
                {
                  _function_data.push_back({u.name, "points"});
                  _write_function<double>(*_io, *_engine, u);
                });
}
//-----------------------------------------------------------------------------
void ADIOS2File::write_function(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u)
{
  assert(_io);
  assert(_engine);
  std::for_each(u.begin(), u.end(),
                [&](const fem::Function<std::complex<double>>& u)
                {
                  _function_data.push_back({u.name + "_real", "points"});
                  _function_data.push_back({u.name + "_imag", "points"});
                  _write_function<std::complex<double>>(*_io, *_engine, u);
                });
}
//-----------------------------------------------------------------------------

#endif