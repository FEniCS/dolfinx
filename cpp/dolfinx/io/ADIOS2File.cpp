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
std::string to_fides(mesh::CellType type)
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
    return std::string();
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

  // NOTE: If we start using mixed element types, we can change data-model to
  // "unstructured"
  DefineAttribute<std::string>(io, "Fides_Data_Model", "unstructured_single");
  // Define FIDES attributes pointing to ADIOS2 Variables for geometry and
  // topology
  DefineAttribute<std::string>(io, "Fides_Coordinates_Variable", "points");
  DefineAttribute<std::string>(io, "Fides_Connecticity_Variable",
                               "connectivity");

  std::string cell_type = to_fides(mesh.topology().cell_type());
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
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      topology.push_back(x_dofs[map[i]]);
  }

  // Put topology (nodes)
  adios2::Variable<std::int64_t> local_topology = DefineVariable<std::int64_t>(
      io, "connectivity", {}, {}, {num_cells * num_nodes});
  engine.Put<std::int64_t>(local_topology, topology.data());

  // Start geometry writer
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(io, "points", {}, {}, {num_vertices, 3});
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
  _io = std::make_unique<adios2::IO>(
      _adios->DeclareIO("ADIOS2-FIDES DOLFINx IO"));
  _io->SetEngine("BPFile");

  //   if (mode == "a")
  //   {
  //     // FIXME: Remove this when is resolved
  //     // https://github.com/ornladios/ADIOS2/issues/2482
  //     _io->SetParameter("AggregatorRatio", "1");
  //   }

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