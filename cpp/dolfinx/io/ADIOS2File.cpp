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
//-----------------------------------------------------------------------------
// Safe definition of an attribute (required for time dependent problems)
template <class T>
adios2::Attribute<T> DefineAttribute(adios2::IO& io, const std::string& name,
                                     const T& value,
                                     const std::string& var_name = "",
                                     const std::string& separator = "/")
{
  adios2::Attribute<T> attr = io.InquireAttribute<T>(name);
  if (attr)
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

  // Add number of nodes (mesh data is written with local indices we need
  // the ghost vertices)
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  // Add cell metadata
  // adios2::Variable<std::uint32_t> cell_variable =
  // DefineVariable<std::uint32_t>(
  //     io, "NumberOfEntities", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> cell_variable = DefineVariable<std::uint32_t>(
      io, "NumberOfCells", {adios2::LocalValueDim});
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
  std::vector<std::uint64_t> topology;
  topology.reserve(num_cells * (num_nodes + 1));
  for (size_t c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    topology.push_back(x_dofs.size());
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      topology.push_back(x_dofs[map[i]]);
  }

  // Put topology (nodes)
  adios2::Variable<std::uint64_t> local_topology
      = DefineVariable<std::uint64_t>(io, "connectivity", {}, {},
                                      {num_cells, num_nodes + 1});
  engine.Put<std::uint64_t>(local_topology, topology.data());

  // Start geometry writer
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put<double>(local_geometry, mesh.geometry().x().data());

  engine.PerformPuts();
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
ADIOS2File::~ADIOS2File() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2File::close()
{
  assert(_engine);

  // This looks a bit odd because ADIOS2 uses `operator bool()` to test
  // of the engine is open
  if (*_engine)
  {
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
#endif