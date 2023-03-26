// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "vtk_utils.h"
#include <adios2.h>
#include <cassert>
#include <complex>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <pugixml.hpp>
#include <string>
#include <variant>
#include <vector>

namespace dolfinx::fem
{
template <typename T, typename U>
class Function;
}

namespace dolfinx::io
{

/// Base class for ADIOS2-based writers
/// @tparam T Geometry type for common mesh::Mesh.
template <std::floating_point T>
class ADIOS2Writer
{
public:
  /// @privatesection
  using Fd32 = fem::Function<float, T>;
  using Fd64 = fem::Function<double, T>;
  using Fc64 = fem::Function<std::complex<float>, T>;
  using Fc128 = fem::Function<std::complex<double>, T>;
  using U = std::vector<
      std::variant<std::shared_ptr<const Fd32>, std::shared_ptr<const Fd64>,
                   std::shared_ptr<const Fc64>, std::shared_ptr<const Fc128>>>;

protected:
  /// @brief Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh
  /// @param[in] u
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               std::string tag, std::shared_ptr<const mesh::Mesh<T>> mesh,
               const U& u)
      : _adios(std::make_unique<adios2::ADIOS>(comm)),
        _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
        _engine(std::make_unique<adios2::Engine>(
            _io->Open(filename, adios2::Mode::Write))),
        _mesh(mesh), _u(u)
  {
    _io->SetEngine("BPFile");
  }

  /// @brief Create an ADIOS2-based writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh The mesh
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               std::string tag, std::shared_ptr<const mesh::Mesh<T>> mesh)
      : ADIOS2Writer(comm, filename, tag, mesh, {})
  {
    // Do nothing
  }

  /// @brief Move constructor
  ADIOS2Writer(ADIOS2Writer&& writer) = default;

  /// @brief Copy constructor
  ADIOS2Writer(const ADIOS2Writer&) = delete;

  /// @brief Destructor
  ~ADIOS2Writer() { close(); }

  /// @brief Move assignment
  ADIOS2Writer& operator=(ADIOS2Writer&& writer) = default;

  // Copy assignment
  ADIOS2Writer& operator=(const ADIOS2Writer&) = delete;

public:
  /// @brief  Close the file
  void close()
  {
    assert(_engine);
    // The reason this looks odd is that ADIOS2 uses `operator bool()`
    // to test if the engine is open
    if (*_engine)
      _engine->Close();
  }

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;
  std::shared_ptr<const mesh::Mesh<T>> _mesh;
  U _u;
};

/// @brief Output of meshes and functions compatible with the Fides
/// Paraview reader, see
/// https://fides.readthedocs.io/en/latest/paraview/paraview.html.
class FidesWriter : public ADIOS2Writer<double>
{
public:
  /// Mesh reuse policy
  enum class MeshPolicy
  {
    update, ///< Re-write the mesh to file upon every write of a fem::Function
    reuse   ///< Write the mesh to file only the first time a fem::Function is
            ///< written to file
  };

  /// @brief  Create Fides writer for a mesh
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh. The mesh must a degree 1 mesh.
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps.
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              std::shared_ptr<const mesh::Mesh<double>> mesh);

  /// @brief Create Fides writer for list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh (degree 1) and (2) be degree 1 Lagrange. @note All
  /// functions in `u` must share the same Mesh
  /// @param[in] mesh_policy Controls if the mesh is written to file at
  /// the first time step only or is re-written (updated) at each time
  /// step.
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              const ADIOS2Writer::U& u,
              const MeshPolicy mesh_policy = MeshPolicy::update);

  // Copy constructor
  FidesWriter(const FidesWriter&) = delete;

  /// @brief  Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// @brief  Destructor
  ~FidesWriter() = default;

  /// @brief Move assignment
  FidesWriter& operator=(FidesWriter&&) = default;

  // Copy assignment
  FidesWriter& operator=(const FidesWriter&) = delete;

  /// @brief Write data with a given time
  /// @param[in] t The time step
  void write(double t);

private:
  // Control whether the mesh is written to file once or at every time step
  MeshPolicy _mesh_reuse_policy;
};

/// @privatesection
namespace impl_vtx
{
/// String suffix for real and complex components of a vector-valued
/// field
constexpr std::array field_ext = {"_real", "_imag"};

template <class... Ts>
struct overload : Ts...
{
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>; // line not needed in C++20...

template <typename T>
std::shared_ptr<const mesh::Mesh<T>>
extract_common_mesh(const typename ADIOS2Writer<T>::U& u)
{
  // Extract mesh from first function
  assert(!u.empty());
  auto mesh = std::visit(
      [](const auto& u) { return u->function_space()->mesh(); }, u.front());
  assert(mesh);

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    std::visit(
        [&mesh](const auto& u)
        {
          if (mesh != u->function_space()->mesh())
          {
            throw std::runtime_error(
                "ADIOS2Writer only supports functions sharing the same mesh");
          }
        },
        v);
  }

  return mesh;
}

/// Safe definition of an attribute. First check if it has already been
/// defined and return it. If not defined create new attribute.
template <class T>
adios2::Attribute<T> define_attribute(adios2::IO& io, std::string name,
                                      const T& value, std::string var_name = "",
                                      std::string separator = "/")
{
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
    return attr;
  else
    return io.DefineAttribute<T>(name, value, var_name, separator);
}

/// Safe definition of a variable. First check if it has already been
/// defined and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> define_variable(adios2::IO& io, std::string name,
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

/// Create VTK xml scheme to be interpreted by the VTX reader
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
std::stringstream create_vtk_schema(const std::vector<std::string>& point_data,
                                    const std::vector<std::string>& cell_data);

/// Extract name of functions and split into real and imaginary component
template <typename T>
std::vector<std::string>
extract_function_names(const typename ADIOS2Writer<T>::U& u)
{
  std::vector<std::string> names;
  using X = decltype(names);
  for (auto& v : u)
  {
    auto n = std::visit(
        overload{
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd32>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fd64>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc64>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            },
            [](const std::shared_ptr<const typename ADIOS2Writer<T>::Fc128>& u)
                -> X {
              return {u->name + field_ext[0], u->name + field_ext[1]};
            }},
        v);
    names.insert(names.end(), n.begin(), n.end());
  };

  return names;
}

/// Given a Function, write the coefficient to file using ADIOS2
/// @note Only supports (discontinuous) Lagrange functions.
/// @note For a complex function, the coefficient is split into a real
/// and imaginary function
/// @note Data is padded to be three dimensional if vector and 9
/// dimensional if tensor
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function
template <typename T, typename X>
void vtx_write_data(adios2::IO& io, adios2::Engine& engine,
                    const fem::Function<T, X>& u)
{
  // Get function data array and information about layout
  assert(u.x());
  std::span<const T> u_vector = u.x()->array();
  const int rank = u.function_space()->element()->value_shape().size();
  const std::uint32_t num_comp = std::pow(3, rank);
  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  std::shared_ptr<const common::IndexMap> index_map = dofmap->index_map;
  assert(index_map);
  const int index_map_bs = dofmap->index_map_bs();
  const int dofmap_bs = dofmap->bs();
  const std::uint32_t num_dofs
      = index_map_bs * (index_map->size_local() + index_map->num_ghosts())
        / dofmap_bs;

  if constexpr (std::is_scalar_v<T>)
  {
    // ---- Real
    std::vector<T> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = u_vector[i * index_map_bs + j];

    adios2::Variable<T> output
        = define_variable<T>(io, u.name, {}, {}, {num_dofs, num_comp});
    engine.Put<T>(output, data.data(), adios2::Mode::Sync);
  }
  else
  {
    // ---- Complex
    using U = typename T::value_type;

    std::vector<U> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::real(u_vector[i * index_map_bs + j]);

    adios2::Variable<U> output_real = define_variable<U>(
        io, u.name + field_ext[0], {}, {}, {num_dofs, num_comp});
    engine.Put<U>(output_real, data.data(), adios2::Mode::Sync);

    std::fill(data.begin(), data.end(), 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::imag(u_vector[i * index_map_bs + j]);
    adios2::Variable<U> output_imag = define_variable<U>(
        io, u.name + field_ext[1], {}, {}, {num_dofs, num_comp});
    engine.Put<U>(output_imag, data.data(), adios2::Mode::Sync);
  }
}

/// Write mesh to file using VTX format
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] mesh The mesh
template <typename T>
void vtx_write_mesh(adios2::IO& io, adios2::Engine& engine,
                    const mesh::Mesh<T>& mesh)
{
  const mesh::Geometry<T>& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();

  // "Put" geometry
  std::shared_ptr<const common::IndexMap> x_map = geometry.index_map();
  const std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable<T> local_geometry
      = define_variable<T>(io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put<T>(local_geometry, geometry.x().data());

  // Put number of nodes. The mesh data is written with local indices,
  // therefore we need the ghost vertices.
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  const auto [vtkcells, shape] = io::extract_vtk_connectivity(
      mesh.geometry().dofmap(), mesh.topology().cell_type());

  // Add cell metadata
  const int tdim = topology.dim();
  adios2::Variable<std::uint32_t> cell_variable
      = define_variable<std::uint32_t>(io, "NumberOfCells",
                                       {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_variable, shape[0]);
  adios2::Variable<std::uint32_t> celltype_variable
      = define_variable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(
      celltype_variable, cells::get_vtk_cell_type(topology.cell_type(), tdim));

  // Pack mesh 'nodes'. Output is written as [N0, v0_0,...., v0_N0, N1,
  // v1_0,...., v1_N1,....], where N is the number of cell nodes and v0,
  // etc, is the node index
  std::vector<std::int64_t> cells(shape[0] * (shape[1] + 1), shape[1]);
  for (std::size_t c = 0; c < shape[0]; ++c)
  {
    std::span vtkcell(vtkcells.data() + c * shape[1], shape[1]);
    std::span cell(cells.data() + c * (shape[1] + 1), shape[1] + 1);
    std::copy(vtkcell.begin(), vtkcell.end(), std::next(cell.begin()));
  }

  // Put topology (nodes)
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {shape[0], shape[1] + 1});
  engine.Put<std::int64_t>(local_topology, cells.data());

  // Vertex global ids and ghost markers
  adios2::Variable<std::int64_t> orig_id = define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {num_vertices});
  engine.Put<std::int64_t>(orig_id, geometry.input_global_indices().data());

  std::vector<std::uint8_t> x_ghost(num_vertices, 0);
  std::fill(std::next(x_ghost.begin(), x_map->size_local()), x_ghost.end(), 1);
  adios2::Variable<std::uint8_t> ghost = define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put<std::uint8_t>(ghost, x_ghost.data());

  engine.PerformPuts();
}

/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates. Writes the topology and geometry using ADIOS2 in
/// VTX format.
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] V The function space
template <typename T>
void vtx_write_mesh_from_space(adios2::IO& io, adios2::Engine& engine,
                               const fem::FunctionSpace<T>& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Get a VTK mesh with points at the 'nodes'
  const auto [x, xshape, x_id, x_ghost, vtk, vtkshape]
      = io::vtk_mesh_from_space(V);

  std::uint32_t num_dofs = xshape[0];

  // -- Pack mesh 'nodes'. Output is written as [N0, v0_0,...., v0_N0, N1,
  // v1_0,...., v1_N1,....], where N is the number of cell nodes and v0,
  // etc, is the node index.

  // Create vector, setting all entries to nodes per cell (vtk.shape(1))
  std::vector<std::int64_t> cells(vtkshape[0] * (vtkshape[1] + 1), vtkshape[1]);

  // Set the [v0_0,...., v0_N0, v1_0,...., v1_N1,....] data
  for (std::size_t c = 0; c < vtkshape[0]; ++c)
  {
    std::span vtkcell(vtk.data() + c * vtkshape[1], vtkshape[1]);
    std::span cell(cells.data() + c * (vtkshape[1] + 1), vtkshape[1] + 1);
    std::copy(vtkcell.begin(), vtkcell.end(), std::next(cell.begin()));
  }

  // Define ADIOS2 variables for geometry, topology, celltypes and
  // corresponding VTK data
  adios2::Variable<T> local_geometry
      = define_variable<T>(io, "geometry", {}, {}, {num_dofs, 3});
  adios2::Variable<std::int64_t> local_topology = define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {vtkshape[0], vtkshape[1] + 1});
  adios2::Variable<std::uint32_t> cell_type
      = define_variable<std::uint32_t>(io, "types");
  adios2::Variable<std::uint32_t> vertices = define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable<std::uint32_t> elements = define_variable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});

  // Write mesh information to file
  engine.Put<std::uint32_t>(vertices, num_dofs);
  engine.Put<std::uint32_t>(elements, vtkshape[0]);
  engine.Put<std::uint32_t>(
      cell_type, cells::get_vtk_cell_type(mesh->topology().cell_type(), tdim));
  engine.Put<T>(local_geometry, x.data());
  engine.Put<std::int64_t>(local_topology, cells.data());

  // Node global ids
  adios2::Variable<std::int64_t> orig_id = define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {x_id.size()});
  engine.Put<std::int64_t>(orig_id, x_id.data());
  adios2::Variable<std::uint8_t> ghost = define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put<std::uint8_t>(ghost, x_ghost.data());

  engine.PerformPuts();
}
} // namespace impl_vtx

/// @brief Writer for meshes and functions using the ADIOS2 VTX format,
/// see
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#using-vtk-and-paraview.
///
/// The output files can be visualized using ParaView.
template <std::floating_point T>
class VTXWriter : public ADIOS2Writer<T>
{
public:
  /// @brief Create a VTX writer for a mesh.
  ///
  /// This format supports arbitrary degree meshes.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh to write
  /// @note This format support arbitrary degree meshes
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            std::shared_ptr<const mesh::Mesh<T>> mesh)
      : ADIOS2Writer<T>(comm, filename, "VTX mesh writer", mesh)
  {
    // Define VTK scheme attribute for mesh
    std::string vtk_scheme = impl_vtx::create_vtk_schema({}, {}).str();
    impl_vtx::define_attribute<std::string>(*this->_io, "vtk.xml", vtk_scheme);
  }

  /// @brief Create a VTX writer for list of functions
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh and (2) be (discontinuous) Lagrange functions. The
  /// element family and degree must be the same for all functions.
  /// @note This format supports arbitrary degree meshes
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            const typename ADIOS2Writer<T>::U& u)
      : ADIOS2Writer<T>(comm, filename, "VTX function writer",
                        impl_vtx::extract_common_mesh<T>(u), u)
  {
    if (u.empty())
      throw std::runtime_error("VTXWriter fem::Function list is empty");

    // Extract element from first function
    const fem::FiniteElement* element0 = std::visit(
        [](const auto& u) { return u->function_space()->element().get(); },
        u.front());
    assert(element0);

    // Check if function is mixed
    if (element0->is_mixed())
      throw std::runtime_error(
          "Mixed functions are not supported by VTXWriter");

    // Check if function is DG 0
    if (element0->space_dimension() / element0->block_size() == 1)
    {
      throw std::runtime_error(
          "VTK does not support cell-wise fields. See "
          "https://gitlab.kitware.com/vtk/vtk/-/issues/18458.");
    }

    // FIXME: is the below check adequate for detecting a Lagrange
    // element?
    // Check that element is Lagrange
    if (!element0->interpolation_ident())
    {
      throw std::runtime_error(
          "Only (discontinuous) Lagrange functions are "
          "supported. Interpolate Functions before output.");
    }

    // Check that all functions come from same element type
    for (auto& v : this->_u)
    {
      std::visit(
          [element0](const auto& u)
          {
            auto element = u->function_space()->element();
            assert(element);
            if (*element != *element0)
            {
              throw std::runtime_error(
                  "All functions in VTXWriter must have the same element type");
            }
          },
          v);
    }

    // Define VTK scheme attribute for set of functions
    std::vector<std::string> names = impl_vtx::extract_function_names<T>(u);
    std::string vtk_scheme = impl_vtx::create_vtk_schema(names, {}).str();
    impl_vtx::define_attribute<std::string>(*this->_io, "vtk.xml", vtk_scheme);
  }

  // Copy constructor
  VTXWriter(const VTXWriter&) = delete;

  /// @brief Move constructor
  VTXWriter(VTXWriter&& file) = default;

  /// @brief Destructor
  ~VTXWriter() = default;

  /// @brief Move assignment
  VTXWriter& operator=(VTXWriter&&) = default;

  // Copy assignment
  VTXWriter& operator=(const VTXWriter&) = delete;

  /// @brief  Write data with a given time
  /// @param[in] t The time step
  void write(double t)
  {
    assert(this->_io);
    assert(this->_engine);
    adios2::Variable<double> var_step
        = impl_vtx::define_variable<double>(*this->_io, "step");

    this->_engine->BeginStep();
    this->_engine->template Put<double>(var_step, t);

    // If we have no functions write the mesh to file
    if (this->_u.empty())
      impl_vtx::vtx_write_mesh(*this->_io, *this->_engine, *this->_mesh);
    else
    {
      // Write a single mesh for functions as they share finite element
      std::visit(
          [&](const auto& u)
          {
            impl_vtx::vtx_write_mesh_from_space<T>(*this->_io, *this->_engine,
                                                   *u->function_space());
          },
          this->_u[0]);

      // Write function data for each function to file
      for (auto& v : this->_u)
        std::visit(
            [&](const auto& u)
            { impl_vtx::vtx_write_data(*this->_io, *this->_engine, *u); },
            v);
    }

    this->_engine->EndStep();
  }
};

/// Type deduction
template <typename U, typename T>
VTXWriter(MPI_Comm comm, U filename, T mesh)
    -> VTXWriter<typename std::remove_cvref<
        typename T::element_type>::type::geometry_type::value_type>;

} // namespace dolfinx::io

#endif
