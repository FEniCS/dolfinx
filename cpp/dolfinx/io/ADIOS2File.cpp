// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2File.h"
#include "dolfinx/io/cells.h"
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

ADIOS2File::~ADIOS2File() { close(); };

void ADIOS2File::close()
{
  if (*_engine)
    _engine->Close();
}

void ADIOS2File::write_function(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double t)
{
  _write_function<double>(u, t);
}

void ADIOS2File::write_function(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
    double t)
{
  _write_function<std::complex<double>>(u, t);
}

template <typename Scalar>
void ADIOS2File::_write_function(
    const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
    double t)
{
  if (_mode == "a")
    throw std::runtime_error(
        "Cannot append functions to previously created file.");
  // Write time step information
  _time_dep = true;
  adios2::Variable<double> time = DefineVariable<double>(_io, "step");
  _engine->Put<double>(time, t);

  // Get some data about mesh
  auto mesh = u[0].get().function_space()->mesh();
  auto top = mesh->topology();
  auto x_map = mesh->geometry().index_map();
  const int tdim = top.dim();

  // As the mesh data is written with local indices we need the ghost vertices
  const std::uint32_t num_elements = top.index_map(tdim)->size_local();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<std::uint32_t> vertices = DefineVariable<std::uint32_t>(
      _io, "NumOfVertices", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = DefineVariable<std::uint32_t>(
      _io, "NumOfElements", {adios2::LocalValueDim});

  // Extract geometry for all local cells
  std::vector<int32_t> cells(num_elements);
  std::iota(cells.begin(), cells.end(), 0);
  adios2::Variable<double> local_geometry
      = DefineVariable<double>(_io, "vertices", {}, {}, {num_vertices, 3});

  // Get DOLFINx to VTK permuation
  // FIXME: Use better way to get number of nods
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector<std::uint8_t> map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh->topology().cell_type() == dolfinx::mesh::CellType::hexahedron
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
  int connectivity_offset = 0;
  std::stringstream cc;

  for (size_t c = 0; c < num_elements; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    vtk_topology[connectivity_offset++] = x_dofs.size();
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      vtk_topology[connectivity_offset++] = x_dofs[map[i]];
    }
  }

  // Add element cell types
  adios2::Variable<std::uint32_t> cell_type
      = DefineVariable<std::uint32_t>(_io, "types");
  // Start writer for given function
  _engine->Put<std::uint32_t>(vertices, num_vertices);
  _engine->Put<std::uint32_t>(elements, num_elements);

  _engine->BeginStep();

  _engine->Put<std::uint32_t>(
      cell_type, dolfinx::io::cells::get_vtk_cell_type(*mesh, tdim));
  _engine->Put<double>(local_geometry, mesh->geometry().x().data());
  _engine->Put<std::uint64_t>(local_topology, vtk_topology.data());
  // Extract and write function data
  std::set<std::string> point_data;
  for (auto u_ : u)
  {
    assert(mesh == u_.get().function_space()->mesh());

    // NOTE: Currently CG-1 interpolation of data.
    auto function_data = u_.get().compute_point_values();
    std::uint32_t local_size = function_data.shape[0];
    std::uint32_t block_size = function_data.shape[1];
    // Extract real and imaginary components
    std::vector<std::string> components = {""};
    if constexpr (!std::is_scalar<Scalar>::value)
      components = {"real", "imag"};

    // Write each component
    std::vector<double> out_data(local_size);

    for (const auto& component : components)
    {
      std::string function_name = u_.get().name;
      if (component != "")
        function_name += "_" + component;
      adios2::Variable<double> local_output
          = DefineVariable<double>(_io, function_name, {}, {}, {local_size});
      for (size_t i = 0; i < local_size; ++i)
      {
        if (component == "imag")
          out_data[i] = std::imag(function_data.row(i)[0]);
        else
          out_data[i] = std::real(function_data.row(i)[0]);
      }
      point_data.insert(function_name);
      // To reuse out_data, we use sync mode here
      _engine->Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
    }
  }
  // Check if VTKScheme exists, and if so, check that we are only adding values
  // already existing
  std::string vtk_scheme = VTKSchema(point_data);
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

std::string ADIOS2File::VTKSchema(std::set<std::string> point_data)
{
  std::string schema = R"(
            <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
              <UnstructuredGrid>
                <Piece NumberOfPoints="NumOfVertices" NumberOfCells="NumOfElements">
                  <Points>
                   <DataArray Name="vertices" />
                  </Points>
                  <Cells>
                     <DataArray Name="connectivity" />
                     <DataArray Name="types" />
                  </Cells>)";

  if (point_data.empty())
    schema += "\n";
  else
  {
    schema += R"(
                  <PointData>)";
    for (auto name : point_data)
    {
      schema += R"(
                     <DataArray Name=")"
                + name + R"(" />)";
    }
    if (_time_dep)
    {
      schema += R"(
                     <DataArray Name="TIME">
                       step 
                     </DataArray>)";
    }
    schema += R"(
                   </PointData>)";
  }
  schema += R"(
                </Piece>
              </UnstructuredGrid>
            </VTKFile> )";
  return schema;
}

#endif