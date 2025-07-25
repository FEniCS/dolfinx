// Copyright (C) 2021-2023 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "vtk_utils.h"
#include <adios2.h>
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cassert>
#include <complex>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/// @file ADIOS2Writers.h
/// @brief ADIOS2-based output writers

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class Function;
}

namespace dolfinx::io
{
namespace adios2_writer
{
/// @privatesection
template <std::floating_point T>
using U = std::vector<std::variant<
    std::shared_ptr<const fem::Function<float, T>>,
    std::shared_ptr<const fem::Function<double, T>>,
    std::shared_ptr<const fem::Function<std::complex<float>, T>>,
    std::shared_ptr<const fem::Function<std::complex<double>, T>>>>;
} // namespace adios2_writer

/// Base class for ADIOS2-based writers
class ADIOS2Writer
{
protected:
  /// @brief Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] engine ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               std::string tag, std::string engine);

  /// @brief Move constructor
  ADIOS2Writer(ADIOS2Writer&& writer) = default;

  /// @brief Copy constructor
  ADIOS2Writer(const ADIOS2Writer&) = delete;

  /// @brief Destructor
  ~ADIOS2Writer();

  /// @brief Move assignment
  ADIOS2Writer& operator=(ADIOS2Writer&& writer) = default;

  // Copy assignment
  ADIOS2Writer& operator=(const ADIOS2Writer&) = delete;

public:
  /// @brief  Close the file
  void close();

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;
};

/// @privatesection
namespace impl_adios2
{
/// String suffix for real and complex components of a vector-valued
/// field
constexpr std::array field_ext = {"_real", "_imag"};

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
  if (adios2::Variable v = io.InquireVariable<T>(name); v)
  {
    if (v.Count() != count and v.ShapeID() == adios2::ShapeID::LocalArray)
      v.SetSelection({start, count});
    return v;
  }
  else
    return io.DefineVariable<T>(name, shape, start, count);
}

/// Extract common mesh from list of Functions
template <std::floating_point T>
std::shared_ptr<const mesh::Mesh<T>>
extract_common_mesh(const typename adios2_writer::U<T>& u)
{
  // Extract mesh from first function
  assert(!u.empty());
  auto mesh = std::visit([](auto&& u) { return u->function_space()->mesh(); },
                         u.front());
  assert(mesh);

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    std::visit(
        [&mesh](auto&& u)
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

} // namespace impl_adios2

/// @privatesection
namespace impl_vtx
{
/// Create VTK xml scheme to be interpreted by the VTX reader
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
std::stringstream create_vtk_schema(const std::vector<std::string>& point_data,
                                    const std::vector<std::string>& cell_data);

/// Extract name of functions and split into real and imaginary component
template <std::floating_point T>
std::vector<std::string>
extract_function_names(const typename adios2_writer::U<T>& u)
{
  std::vector<std::string> names;
  for (auto& v : u)
  {
    std::visit(
        [&names](auto&& u)
        {
          using U = std::decay_t<decltype(u)>;
          using X = typename U::element_type;
          if constexpr (std::is_floating_point_v<typename X::value_type>)
            names.push_back(u->name);
          else
          {
            names.push_back(u->name + impl_adios2::field_ext[0]);
            names.push_back(u->name + impl_adios2::field_ext[1]);
          }
        },
        v);
  }

  return names;
}

/// Given a Function, write the coefficient to file using ADIOS2.
/// @note Only supports (discontinuous) Lagrange functions.
/// @note For a complex function, the coefficient is split into a real
/// and imaginary function.
/// @note Data is padded to be three dimensional if vector and 9
/// dimensional if tensor.
/// @param[in] io ADIOS2 io object.
/// @param[in] engine ADIOS2 engine object.
/// @param[in] u Function to write.
template <typename T, std::floating_point X>
void vtx_write_data(adios2::IO& io, adios2::Engine& engine,
                    const fem::Function<T, X>& u)
{
  // Get function data array and information about layout
  assert(u.x());
  std::span<const T> u_vector = u.x()->array();

  // Pad to 3D if vector/tensor is product of dimensions is smaller than
  // 3**rank to ensure that we can visualize them correctly in Paraview
  std::span<const std::size_t> value_shape
      = u.function_space()->element()->value_shape();
  std::size_t rank = value_shape.size();
  std::size_t num_comp = std::reduce(value_shape.begin(), value_shape.end(), 1,
                                     std::multiplies{});
  if (num_comp < std::pow(3, rank))
    num_comp = std::pow(3, rank);

  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  std::shared_ptr<const common::IndexMap> index_map = dofmap->index_map;
  assert(index_map);
  int index_map_bs = dofmap->index_map_bs();
  int dofmap_bs = dofmap->bs();
  std::uint32_t num_dofs = index_map_bs
                           * (index_map->size_local() + index_map->num_ghosts())
                           / dofmap_bs;
  if constexpr (std::is_scalar_v<T>)
  {
    // ---- Real
    std::vector<T> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = u_vector[i * index_map_bs + j];

    adios2::Variable output = impl_adios2::define_variable<T>(
        io, u.name, {}, {}, {num_dofs, num_comp});
    engine.Put(output, data.data(), adios2::Mode::Sync);
  }
  else
  {
    // ---- Complex
    using U = typename T::value_type;

    std::vector<U> data(num_dofs * num_comp, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::real(u_vector[i * index_map_bs + j]);

    adios2::Variable output_real = impl_adios2::define_variable<U>(
        io, u.name + impl_adios2::field_ext[0], {}, {}, {num_dofs, num_comp});
    engine.Put(output_real, data.data(), adios2::Mode::Sync);

    std::ranges::fill(data, 0);
    for (std::size_t i = 0; i < num_dofs; ++i)
      for (int j = 0; j < index_map_bs; ++j)
        data[i * num_comp + j] = std::imag(u_vector[i * index_map_bs + j]);
    adios2::Variable output_imag = impl_adios2::define_variable<U>(
        io, u.name + impl_adios2::field_ext[1], {}, {}, {num_dofs, num_comp});
    engine.Put(output_imag, data.data(), adios2::Mode::Sync);
  }
}

/// Write mesh to file using VTX format
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] mesh The mesh
template <std::floating_point T>
void vtx_write_mesh(adios2::IO& io, adios2::Engine& engine,
                    const mesh::Mesh<T>& mesh)
{
  const mesh::Geometry<T>& geometry = mesh.geometry();
  auto topology = mesh.topology();
  assert(topology);

  // "Put" geometry
  std::shared_ptr<const common::IndexMap> x_map = geometry.index_map();
  std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable local_geometry = impl_adios2::define_variable<T>(
      io, "geometry", {}, {}, {num_vertices, 3});
  engine.Put(local_geometry, geometry.x().data());

  // Put number of nodes. The mesh data is written with local indices,
  // therefore we need the ghost vertices.
  adios2::Variable vertices = impl_adios2::define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(vertices, num_vertices);

  auto [vtkcells, shape]
      = io::extract_vtk_connectivity(geometry.dofmap(), topology->cell_type());

  // Add cell metadata
  int tdim = topology->dim();
  adios2::Variable cell_var = impl_adios2::define_variable<std::uint32_t>(
      io, "NumberOfCells", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_var, shape[0]);
  adios2::Variable celltype_var
      = impl_adios2::define_variable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(
      celltype_var, cells::get_vtk_cell_type(topology->cell_type(), tdim));

  // Pack mesh 'nodes'. Output is written as [N0, v0_0,...., v0_N0, N1,
  // v1_0,...., v1_N1,....], where N is the number of cell nodes and v0,
  // etc, is the node index
  std::vector<std::int64_t> cells(shape[0] * (shape[1] + 1), shape[1]);
  for (std::size_t c = 0; c < shape[0]; ++c)
  {
    std::span vtkcell(vtkcells.data() + c * shape[1], shape[1]);
    std::span cell(cells.data() + c * (shape[1] + 1), shape[1] + 1);
    std::ranges::copy(vtkcell, std::next(cell.begin()));
  }

  // Put topology (nodes)
  adios2::Variable local_topology = impl_adios2::define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {shape[0], shape[1] + 1});
  engine.Put(local_topology, cells.data());

  // Vertex global ids and ghost markers
  adios2::Variable orig_id = impl_adios2::define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {num_vertices});
  engine.Put(orig_id, geometry.input_global_indices().data());

  std::vector<std::uint8_t> x_ghost(num_vertices, 0);
  std::fill(std::next(x_ghost.begin(), x_map->size_local()), x_ghost.end(), 1);
  adios2::Variable ghost = impl_adios2::define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put(ghost, x_ghost.data());
  engine.PerformPuts();
}

/// @brief Given a FunctionSpace, create a topology and geometry based
/// on the function space dof coordinates. Writes the topology and
/// geometry using ADIOS2 in VTX format.
/// @note Only supports (discontinuous) Lagrange functions.
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] V The function space
template <std::floating_point T>
std::pair<std::vector<std::int64_t>, std::vector<std::uint8_t>>
vtx_write_mesh_from_space(adios2::IO& io, adios2::Engine& engine,
                          const fem::FunctionSpace<T>& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  auto topology = mesh->topology();
  assert(topology);
  int tdim = topology->dim();

  // Get a VTK mesh with points at the 'nodes'
  auto [x, xshape, x_id, x_ghost, vtk, vtkshape] = io::vtk_mesh_from_space(V);

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
    std::ranges::copy(vtkcell, std::next(cell.begin()));
  }

  // Define ADIOS2 variables for geometry, topology, celltypes and
  // corresponding VTK data
  adios2::Variable local_geometry
      = impl_adios2::define_variable<T>(io, "geometry", {}, {}, {num_dofs, 3});
  adios2::Variable local_topology = impl_adios2::define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {vtkshape[0], vtkshape[1] + 1});
  adios2::Variable cell_type
      = impl_adios2::define_variable<std::uint32_t>(io, "types");
  adios2::Variable vertices = impl_adios2::define_variable<std::uint32_t>(
      io, "NumberOfNodes", {adios2::LocalValueDim});
  adios2::Variable elements = impl_adios2::define_variable<std::uint32_t>(
      io, "NumberOfEntities", {adios2::LocalValueDim});

  // Write mesh information to file
  engine.Put<std::uint32_t>(vertices, num_dofs);
  engine.Put<std::uint32_t>(elements, vtkshape[0]);
  engine.Put<std::uint32_t>(
      cell_type, cells::get_vtk_cell_type(topology->cell_type(), tdim));
  engine.Put(local_geometry, x.data());
  engine.Put(local_topology, cells.data());

  // Node global ids
  adios2::Variable orig_id = impl_adios2::define_variable<std::int64_t>(
      io, "vtkOriginalPointIds", {}, {}, {x_id.size()});
  engine.Put(orig_id, x_id.data());
  adios2::Variable ghost = impl_adios2::define_variable<std::uint8_t>(
      io, "vtkGhostType", {}, {}, {x_ghost.size()});
  engine.Put(ghost, x_ghost.data());

  engine.PerformPuts();
  return {std::move(x_id), std::move(x_ghost)};
}
} // namespace impl_vtx

/// Mesh reuse policy
enum class VTXMeshPolicy : std::int8_t
{
  update, ///< Re-write the mesh to file upon every write of a fem::Function
  reuse   ///< Write the mesh to file only the first time a fem::Function is
          ///< written to file
};

/// @brief Writer for meshes and functions using the ADIOS2 VTX format,
/// see
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#using-vtk-and-paraview.
///
/// The output files can be visualized using ParaView.
template <std::floating_point T>
class VTXWriter : public ADIOS2Writer
{
public:
  /// @brief Create a VTX writer for a mesh.
  ///
  /// This format supports arbitrary degree meshes.
  ///
  /// @param[in] comm MPI communicator to open the file on.
  /// @param[in] filename Name of output file.
  /// @param[in] mesh Mesh to write.
  /// @param[in] engine ADIOS2 engine type.
  /// @note This format supports arbitrary degree meshes.
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps.
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            std::shared_ptr<const mesh::Mesh<T>> mesh,
            std::string engine = "BPFile")
      : ADIOS2Writer(comm, filename, "VTX mesh writer", engine), _mesh(mesh),
        _mesh_reuse_policy(VTXMeshPolicy::update), _is_piecewise_constant(false)
  {
    // Define VTK scheme attribute for mesh
    std::string vtk_scheme = impl_vtx::create_vtk_schema({}, {}).str();
    impl_adios2::define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
  }

  /// @brief Create a VTX writer for a list of fem::Functions.
  ///
  /// This format supports arbitrary degree meshes.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh and (2) be (discontinuous) Lagrange functions. The
  /// element family and degree, and degree-of-freedom map (up to the
  /// blocksize) must be the same for all functions.
  /// @param[in] engine ADIOS2 engine type.
  /// @param[in] mesh_policy Controls if the mesh is written to file at
  /// the first time step only or is re-written (updated) at each time
  /// step.
  /// @note This format supports arbitrary degree meshes.
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            const typename adios2_writer::U<T>& u, std::string engine,
            VTXMeshPolicy mesh_policy = VTXMeshPolicy::update)
      : ADIOS2Writer(comm, filename, "VTX function writer", engine),
        _mesh(impl_adios2::extract_common_mesh<T>(u)), _u(u),
        _mesh_reuse_policy(mesh_policy), _is_piecewise_constant(false)
  {
    if (u.empty())
      throw std::runtime_error("VTXWriter fem::Function list is empty.");

    // Extract space from first function
    auto V0 = std::visit([](auto& u) { return u->function_space().get(); },
                         u.front());
    assert(V0);
    auto element0 = V0->element().get();
    assert(element0);

    // Check if function is mixed
    if (element0->is_mixed())
    {
      throw std::runtime_error(
          "Mixed functions are not supported by VTXWriter.");
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

    // Check if function is DG 0
    if (element0->space_dimension() / element0->block_size() == 1)
      _is_piecewise_constant = true;

    // Check that all functions come from same element type
    for (auto& v : _u)
    {
      std::visit(
          [V0](auto& u)
          {
            auto element = u->function_space()->element();
            assert(element);
            if (*element != *V0->element().get())
            {
              throw std::runtime_error("All functions in VTXWriter must have "
                                       "the same element type.");
            }
#ifndef NDEBUG
            auto dmap0 = V0->dofmap()->map();
            auto dmap = u->function_space()->dofmap()->map();
            if (dmap0.size() != dmap.size()
                or !std::equal(dmap0.data_handle(),
                               dmap0.data_handle() + dmap0.size(),
                               dmap.data_handle()))
            {
              throw std::runtime_error(
                  "All functions must have the same dofmap for VTXWriter.");
            }
#endif
          },
          v);
    }

    // Define VTK scheme attribute for set of functions
    std::vector<std::string> names = impl_vtx::extract_function_names<T>(u);
    std::string vtk_scheme;
    if (_is_piecewise_constant)
      vtk_scheme = impl_vtx::create_vtk_schema({}, names).str();
    else
      vtk_scheme = impl_vtx::create_vtk_schema(names, {}).str();

    impl_adios2::define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
  }

  /// @brief Create a VTX writer for a list of fem::Functions using
  /// the default ADIOS2 engine.
  ///
  /// This format supports arbitrary degree meshes.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh and (2) be (discontinuous) Lagrange functions. The
  /// element family and degree must be the same for all functions.
  /// @param[in] mesh_policy Controls if the mesh is written to file at
  /// the first time step only or is re-written (updated) at each time
  /// step.
  /// @note This format supports arbitrary degree meshes.
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            const typename adios2_writer::U<T>& u,
            VTXMeshPolicy mesh_policy = VTXMeshPolicy::update)
      : VTXWriter(comm, filename, u, "BPFile", mesh_policy)
  {
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

  /// @brief Write data with a given time stamp.
  /// @param[in] t Time stamp to associate with output.
  void write(double t)
  {
    assert(_io);
    adios2::Variable var_step
        = impl_adios2::define_variable<double>(*_io, "step");

    assert(_engine);
    _engine->BeginStep();
    _engine->template Put<double>(var_step, t);

    // If we have no functions or DG functions write the mesh to file
    if (_is_piecewise_constant or _u.empty())
    {
      impl_vtx::vtx_write_mesh(*_io, *_engine, *_mesh);
      if (_is_piecewise_constant)
      {
        for (auto& v : _u)
        {
          std::visit([&](auto& u)
                     { impl_vtx::vtx_write_data(*_io, *_engine, *u); }, v);
        }
      }
    }
    else
    {
      if (_mesh_reuse_policy == VTXMeshPolicy::update
          or !(_io->template InquireVariable<std::int64_t>("connectivity")))
      {
        // Write a single mesh for functions as they share finite
        // element
        std::tie(_x_id, _x_ghost) = std::visit(
            [&](auto& u)
            {
              return impl_vtx::vtx_write_mesh_from_space(*_io, *_engine,
                                                         *u->function_space());
            },
            _u[0]);
      }
      else
      {
        // Node global ids
        adios2::Variable orig_id = impl_adios2::define_variable<std::int64_t>(
            *_io, "vtkOriginalPointIds", {}, {}, {_x_id.size()});
        _engine->Put(orig_id, _x_id.data());
        adios2::Variable ghost = impl_adios2::define_variable<std::uint8_t>(
            *_io, "vtkGhostType", {}, {}, {_x_ghost.size()});
        _engine->Put(ghost, _x_ghost.data());
        _engine->PerformPuts();
      }

      // Write function data for each function to file
      for (auto& v : _u)
      {
        std::visit([&](auto& u)
                   { impl_vtx::vtx_write_data(*_io, *_engine, *u); }, v);
      }
    }

    _engine->EndStep();
  }

private:
  std::shared_ptr<const mesh::Mesh<T>> _mesh;
  adios2_writer::U<T> _u;

  // Control whether the mesh is written to file once or at every time
  // step
  VTXMeshPolicy _mesh_reuse_policy;
  std::vector<std::int64_t> _x_id;
  std::vector<std::uint8_t> _x_ghost;

  // Special handling of piecewise constant functions
  bool _is_piecewise_constant;
};

/// Type deduction
template <typename U, typename T>
VTXWriter(MPI_Comm comm, U filename, T mesh)
    -> VTXWriter<typename std::remove_cvref<
        typename T::element_type>::type::geometry_type::value_type>;

} // namespace dolfinx::io

#endif
