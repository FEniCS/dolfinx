// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "vtk_utils.h"
#include <adios2.h>
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
#include <variant>
#include <vector>

/// @file ADIOS2Writers.h
/// @brief ADIOS2-based output writers

namespace dolfinx::fem
{
template <typename T, std::floating_point U>
class Function;
}

namespace dolfinx::io
{
namespace adios2_writer
{
/// @privatesection
template <std::floating_point T>
using Fd32 = fem::Function<float, T>;
template <std::floating_point T>
using Fd64 = fem::Function<double, T>;
template <std::floating_point T>
using Fc64 = fem::Function<std::complex<float>, T>;
template <std::floating_point T>
using Fc128 = fem::Function<std::complex<double>, T>;
template <std::floating_point T>
using U = std::vector<std::variant<
    std::shared_ptr<const Fd32<T>>, std::shared_ptr<const Fd64<T>>,
    std::shared_ptr<const Fc64<T>>, std::shared_ptr<const Fc128<T>>>>;
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

/// @private
template <class... Ts>
struct overload : Ts...
{
  using Ts::operator()...;
};
/// @private
template <class... Ts>
overload(Ts...) -> overload<Ts...>; // line not needed in C++20...

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
  auto mesh = std::visit([](auto& u) { return u->function_space()->mesh(); },
                         u.front());
  assert(mesh);

  // Check that all functions share the same mesh
  for (auto& v : u)
  {
    std::visit(
        [&mesh](auto& u)
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
namespace impl_fides
{
/// Initialize mesh related attributes for the ADIOS2 file used in
/// Fides.
/// @param[in] io ADIOS2 IO object
/// @param[in] type Cell type
void initialize_mesh_attributes(adios2::IO& io, mesh::CellType type);

/// Initialize function related attributes for the ADIOS2 file used in
/// Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] u List of functions
template <std::floating_point T>
void initialize_function_attributes(adios2::IO& io,
                                    const typename adios2_writer::U<T>& u)
{
  // Array of function (name, cell association types) for each function
  // added to the file
  std::vector<std::array<std::string, 2>> u_data;
  using X = decltype(u_data);
  for (auto& v : u)
  {
    auto n = std::visit(
        impl_adios2::overload{
            [](const std::shared_ptr<const typename adios2_writer::Fd32<T>>& u)
                -> X {
              return {{u->name, "points"}};
            },
            [](const std::shared_ptr<const typename adios2_writer::Fd64<T>>& u)
                -> X {
              return {{u->name, "points"}};
            },
            [](const std::shared_ptr<const typename adios2_writer::Fc64<T>>& u)
                -> X
            {
              return {{u->name + impl_adios2::field_ext[0], "points"},
                      {u->name + impl_adios2::field_ext[1], "points"}};
            },
            [](const std::shared_ptr<const typename adios2_writer::Fc128<T>>& u)
                -> X
            {
              return {{u->name + impl_adios2::field_ext[0], "points"},
                      {u->name + impl_adios2::field_ext[1], "points"}};
            }},
        v);
    u_data.insert(u_data.end(), n.begin(), n.end());
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

/// Pack Function data at vertices. The mesh and the function must both
/// be 'P1'.
template <typename T, std::floating_point U>
std::vector<T> pack_function_data(const fem::Function<T, U>& u)
{
  namespace stdex = std::experimental;

  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  auto mesh = V->mesh();
  assert(mesh);
  const mesh::Geometry<U>& geometry = mesh->geometry();
  auto topology = mesh->topology();
  assert(topology);

  // The Function and the mesh must have identical element_dof_layouts
  // (up to the block size)
  assert(dofmap->element_dof_layout()
         == geometry.cmaps()[0].create_dof_layout());

  int tdim = topology->dim();
  auto cell_map = topology->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();

  auto vertex_map = topology->index_map(0);
  assert(vertex_map);
  std::uint32_t num_vertices
      = vertex_map->size_local() + vertex_map->num_ghosts();

  int rank = V->element()->value_shape().size();
  std::uint32_t num_components = std::pow(3, rank);

  // Get dof array and pack into array (padded where appropriate)
  auto dofmap_x = geometry.dofmap();
  const int bs = dofmap->bs();
  std::span<const T> u_data = u.x()->array();
  std::vector<T> data(num_vertices * num_components, 0);
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    auto dofs_x = stdex::submdspan(dofmap_x, c, stdex::full_extent);
    assert(dofs.size() == dofs_x.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int j = 0; j < bs; ++j)
        data[num_components * dofs_x[i] + j] = u_data[bs * dofs[i] + j];
  }
  return data;
}

/// Write a first order Lagrange function (real or complex) using ADIOS2
/// in Fides format. Data is padded to be three dimensional if vector
/// and 9 dimensional if tensor.
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] u The function to write
template <typename T, std::floating_point U>
void write_data(adios2::IO& io, adios2::Engine& engine,
                const fem::Function<T, U>& u)
{
  // FIXME: There is an implicit assumptions that u and the mesh have
  // the same ElementDoflayout
  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  auto mesh = V->mesh();
  assert(mesh);
  const int gdim = mesh->geometry().dim();

  // Vectors and tensor need padding in gdim < 3
  int rank = V->element()->value_shape().size();
  bool need_padding = rank > 0 and gdim != 3 ? true : false;

  // Get vertex data. If the mesh and function dofmaps are the same we
  // can work directly with the dof array.
  std::span<const T> data;
  std::vector<T> _data;
  auto eq_check = [](auto x, auto y) -> bool
  {
    return x.extents() == y.extents()
           and std::equal(x.data_handle(), x.data_handle() + x.size(),
                          y.data_handle());
  };

  if (!need_padding and eq_check(mesh->geometry().dofmap(), dofmap->map()))
    data = u.x()->array();
  else
  {
    _data = impl_fides::pack_function_data(u);
    data = std::span<const T>(_data);
  }

  auto vertex_map = mesh->topology()->index_map(0);
  assert(vertex_map);
  std::uint32_t num_vertices
      = vertex_map->size_local() + vertex_map->num_ghosts();

  // Write each real and imaginary part of the function
  std::uint32_t num_components = std::pow(3, rank);
  assert(data.size() % num_components == 0);
  if constexpr (std::is_scalar_v<T>)
  {
    // ---- Real
    adios2::Variable local_output = impl_adios2::define_variable<T>(
        io, u.name, {}, {}, {num_vertices, num_components});

    // To reuse out_data, we use sync mode here
    engine.Put(local_output, data.data());
    engine.PerformPuts();
  }
  else
  {
    // ---- Complex
    using X = typename T::value_type;

    std::vector<X> data_real(data.size()), data_imag(data.size());

    adios2::Variable local_output_r = impl_adios2::define_variable<X>(
        io, u.name + impl_adios2::field_ext[0], {}, {},
        {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_real.begin(),
                   [](auto x) -> X { return std::real(x); });
    engine.Put(local_output_r, data_real.data());

    adios2::Variable local_output_c = impl_adios2::define_variable<X>(
        io, u.name + impl_adios2::field_ext[1], {}, {},
        {num_vertices, num_components});
    std::transform(data.begin(), data.end(), data_imag.begin(),
                   [](auto x) -> X { return std::imag(x); });
    engine.Put(local_output_c, data_imag.data());
    engine.PerformPuts();
  }
}

/// Write mesh geometry and connectivity (topology) for Fides
/// @param[in] io The ADIOS2 IO
/// @param[in] engine The ADIOS2 engine
/// @param[in] mesh The mesh
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                const mesh::Mesh<T>& mesh)
{
  const mesh::Geometry<T>& geometry = mesh.geometry();
  auto topology = mesh.topology();
  assert(topology);

  // "Put" geometry data
  auto x_map = geometry.index_map();
  std::uint32_t num_vertices = x_map->size_local() + x_map->num_ghosts();
  adios2::Variable local_geometry = impl_adios2::define_variable<T>(
      io, "points", {}, {}, {num_vertices, 3});
  engine.Put(local_geometry, geometry.x().data());

  // TODO: The DOLFINx and VTK topology are the same for some cell types
  // - no need to repack via extract_vtk_connectivity in these cases

  // Get topological dimenson, number of cells and number of 'nodes' per
  // cell, and compute 'VTK' connectivity
  int tdim = topology->dim();
  std::int32_t num_cells = topology->index_map(tdim)->size_local();
  int num_nodes = geometry.cmaps()[0].dim();
  auto [cells, shape] = io::extract_vtk_connectivity(mesh.geometry().dofmap(),
                                                     topology->cell_types()[0]);

  // "Put" topology data in the result in the ADIOS2 file
  adios2::Variable local_topology = impl_adios2::define_variable<std::int64_t>(
      io, "connectivity", {}, {}, {std::size_t(num_cells * num_nodes)});
  engine.Put(local_topology, cells.data());
  engine.PerformPuts();
}

} // namespace impl_fides

/// Mesh reuse policy
enum class FidesMeshPolicy
{
  update, ///< Re-write the mesh to file upon every write of a fem::Function
  reuse   ///< Write the mesh to file only the first time a fem::Function is
          ///< written to file
};

/// @brief Output of meshes and functions compatible with the Fides
/// Paraview reader, see
/// https://fides.readthedocs.io/en/latest/paraview/paraview.html.
template <std::floating_point T>
class FidesWriter : public ADIOS2Writer
{
public:
  /// @brief  Create Fides writer for a mesh
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh. The mesh must a degree 1 mesh.
  /// @param[in] engine ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps.
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              std::shared_ptr<const mesh::Mesh<T>> mesh,
              std::string engine = "BPFile")
      : ADIOS2Writer(comm, filename, "Fides mesh writer", engine),
        _mesh_reuse_policy(FidesMeshPolicy::update), _mesh(mesh)
  {
    assert(_io);
    assert(mesh);
    auto topology = mesh->topology();
    assert(topology);
    mesh::CellType type = topology->cell_types()[0];
    if (mesh->geometry().cmaps()[0].dim() != mesh::cell_num_entities(type, 0))
      throw std::runtime_error("Fides only supports lowest-order meshes.");
    impl_fides::initialize_mesh_attributes(*_io, type);
  }

  /// @brief Create Fides writer for list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh (degree 1) and (2) be degree 1 Lagrange. @note All
  /// functions in `u` must share the same Mesh
  /// @param[in] engine ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mesh_policy Controls if the mesh is written to file at
  /// the first time step only or is re-written (updated) at each time
  /// step.
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              const typename adios2_writer::U<T>& u, std::string engine,
              const FidesMeshPolicy mesh_policy = FidesMeshPolicy::update)
      : ADIOS2Writer(comm, filename, "Fides function writer", engine),
        _mesh_reuse_policy(mesh_policy),
        _mesh(impl_adios2::extract_common_mesh<T>(u)), _u(u)
  {
    if (u.empty())
      throw std::runtime_error("FidesWriter fem::Function list is empty");

    // Extract Mesh from first function
    auto mesh = std::visit([](auto& u) { return u->function_space()->mesh(); },
                           u.front());
    assert(mesh);

    // Extract element from first function
    auto element0 = std::visit([](auto& e)
                               { return e->function_space()->element().get(); },
                               u.front());
    assert(element0);

    // Check if function is mixed
    if (element0->is_mixed())
    {
      throw std::runtime_error(
          "Mixed functions are not supported by VTXWriter");
    }

    // Check if function is DG 0
    if (element0->space_dimension() / element0->block_size() == 1)
    {
      throw std::runtime_error(
          "Piecewise constants are not (yet) supported by VTXWriter");
    }

    // FIXME: is the below check adequate for detecting a
    // Lagrange element? Check that element is Lagrange
    if (!element0->interpolation_ident())
    {
      throw std::runtime_error("Only Lagrange functions are supported. "
                               "Interpolate Functions before output.");
    }

    // Check that all functions are first order Lagrange
    auto cell_types = mesh->topology()->cell_types();
    if (cell_types.size() > 1)
      throw std::runtime_error("Multiple cell types in IO.");
    int num_vertices_per_cell = mesh::cell_num_entities(cell_types.back(), 0);
    for (auto& v : _u)
    {
      std::visit(
          [&](auto& u)
          {
            auto element = u->function_space()->element();
            assert(element);
            if (*element != *element0)
            {
              throw std::runtime_error("All functions in FidesWriter must have "
                                       "the same element type");
            }
            auto dof_layout
                = u->function_space()->dofmap()->element_dof_layout();
            int num_vertex_dofs = dof_layout.num_entity_dofs(0);
            int num_dofs = element->space_dimension() / element->block_size();
            if (num_dofs != num_vertices_per_cell or num_vertex_dofs != 1)
            {
              throw std::runtime_error("Only first order Lagrange spaces are "
                                       "supported by FidesWriter");
            }
          },
          v);
    }

    auto topology = mesh->topology();
    assert(topology);
    mesh::CellType type = topology->cell_types()[0];
    if (mesh->geometry().cmaps()[0].dim() != mesh::cell_num_entities(type, 0))
      throw std::runtime_error("Fides only supports lowest-order meshes.");
    impl_fides::initialize_mesh_attributes(*_io, type);
    impl_fides::initialize_function_attributes<T>(*_io, u);
  }

  /// @brief Create Fides writer for list of functions using default
  /// ADIOS2 engine,
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh (degree 1) and (2) be degree 1 Lagrange. @note All
  /// functions in `u` must share the same Mesh
  /// @param[in] mesh_policy Controls if the mesh is written to file at
  /// the first time step only or is re-written (updated) at each time
  /// step.
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              const typename adios2_writer::U<T>& u,
              const FidesMeshPolicy mesh_policy = FidesMeshPolicy::update)
      : ADIOS2Writer(comm, filename, "Fides function writer", "BPFile"),
        _mesh_reuse_policy(mesh_policy),
        _mesh(impl_adios2::extract_common_mesh<T>(u)), _u(u)
  {
  }

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
  void write(double t)
  {
    assert(_io);
    assert(_engine);
    _engine->BeginStep();
    adios2::Variable var_step
        = impl_adios2::define_variable<double>(*_io, "step");
    _engine->template Put<double>(var_step, t);

    if (auto v = _io->template InquireVariable<std::int64_t>("connectivity");
        !v or _mesh_reuse_policy == FidesMeshPolicy::update)
    {
      impl_fides::write_mesh(*_io, *_engine, *_mesh);
    }

    for (auto& v : _u)
    {
      std::visit([&](auto& u) { impl_fides::write_data(*_io, *_engine, *u); },
                 v);
    }

    _engine->EndStep();
  }

private:
  // Control whether the mesh is written to file once or at every time
  // step
  FidesMeshPolicy _mesh_reuse_policy;

  std::shared_ptr<const mesh::Mesh<T>> _mesh;
  adios2_writer::U<T> _u;
};

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
  using X = decltype(names);
  for (auto& v : u)
  {
    auto n = std::visit(
        impl_adios2::overload{
            [](const std::shared_ptr<const typename adios2_writer::Fd32<T>>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename adios2_writer::Fd64<T>>& u)
                -> X { return {u->name}; },
            [](const std::shared_ptr<const typename adios2_writer::Fc64<T>>& u)
                -> X
            {
              return {u->name + impl_adios2::field_ext[0],
                      u->name + impl_adios2::field_ext[1]};
            },
            [](const std::shared_ptr<const typename adios2_writer::Fc128<T>>& u)
                -> X
            {
              return {u->name + impl_adios2::field_ext[0],
                      u->name + impl_adios2::field_ext[1]};
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
template <typename T, std::floating_point X>
void vtx_write_data(adios2::IO& io, adios2::Engine& engine,
                    const fem::Function<T, X>& u)
{
  // Get function data array and information about layout
  assert(u.x());
  std::span<const T> u_vector = u.x()->array();
  int rank = u.function_space()->element()->value_shape().size();
  std::uint32_t num_comp = std::pow(3, rank);
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

    std::fill(data.begin(), data.end(), 0);
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

  auto [vtkcells, shape] = io::extract_vtk_connectivity(
      geometry.dofmap(), topology->cell_types()[0]);

  // Add cell metadata
  int tdim = topology->dim();
  adios2::Variable cell_var = impl_adios2::define_variable<std::uint32_t>(
      io, "NumberOfCells", {adios2::LocalValueDim});
  engine.Put<std::uint32_t>(cell_var, shape[0]);
  adios2::Variable celltype_var
      = impl_adios2::define_variable<std::uint32_t>(io, "types");
  engine.Put<std::uint32_t>(
      celltype_var, cells::get_vtk_cell_type(topology->cell_types()[0], tdim));

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

/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates. Writes the topology and geometry using ADIOS2 in
/// VTX format.
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] io The ADIOS2 io object
/// @param[in] engine The ADIOS2 engine object
/// @param[in] V The function space
template <std::floating_point T>
void vtx_write_mesh_from_space(adios2::IO& io, adios2::Engine& engine,
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
    std::copy(vtkcell.begin(), vtkcell.end(), std::next(cell.begin()));
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
      cell_type, cells::get_vtk_cell_type(topology->cell_types()[0], tdim));
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
}
} // namespace impl_vtx

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
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh to write
  /// @param[in] engine ADIOS2 engine type
  /// @note This format support arbitrary degree meshes
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            std::shared_ptr<const mesh::Mesh<T>> mesh,
            std::string engine = "BPFile")
      : ADIOS2Writer(comm, filename, "VTX mesh writer", engine), _mesh(mesh)
  {
    // Define VTK scheme attribute for mesh
    std::string vtk_scheme = impl_vtx::create_vtk_schema({}, {}).str();
    impl_adios2::define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
  }

  /// @brief Create a VTX writer for list of functions
  ///
  /// This format supports arbitrary degree meshes.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh and (2) be (discontinuous) Lagrange functions. The
  /// element family and degree must be the same for all functions.
  /// @param[in] engine ADIOS2 engine type.
  /// @note This format supports arbitrary degree meshes
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            const typename adios2_writer::U<T>& u,
            std::string engine = "BPFile")
      : ADIOS2Writer(comm, filename, "VTX function writer", engine),
        _mesh(impl_adios2::extract_common_mesh<T>(u)), _u(u)
  {
    if (u.empty())
      throw std::runtime_error("VTXWriter fem::Function list is empty");

    // Extract element from first function
    auto element0 = std::visit([](auto& u)
                               { return u->function_space()->element().get(); },
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
    for (auto& v : _u)
    {
      std::visit(
          [element0](auto& u)
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
    impl_adios2::define_attribute<std::string>(*_io, "vtk.xml", vtk_scheme);
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
    assert(_io);
    assert(_engine);
    adios2::Variable var_step
        = impl_adios2::define_variable<double>(*_io, "step");

    _engine->BeginStep();
    _engine->template Put<double>(var_step, t);

    // If we have no functions write the mesh to file
    if (_u.empty())
      impl_vtx::vtx_write_mesh(*_io, *_engine, *_mesh);
    else
    {
      // Write a single mesh for functions as they share finite element
      std::visit(
          [&](auto& u) {
            impl_vtx::vtx_write_mesh_from_space(*_io, *_engine,
                                                *u->function_space());
          },
          _u[0]);

      // Write function data for each function to file
      for (auto& v : _u)
        std::visit(
            [&](auto& u) { impl_vtx::vtx_write_data(*_io, *_engine, *u); }, v);
    }

    _engine->EndStep();
  }

private:
  std::shared_ptr<const mesh::Mesh<T>> _mesh;
  adios2_writer::U<T> _u;
};

/// Type deduction
template <typename U, typename T>
VTXWriter(MPI_Comm comm, U filename, T mesh)
    -> VTXWriter<typename std::remove_cvref<
        typename T::element_type>::type::geometry_type::value_type>;

/// Type deduction
template <typename U, typename T>
FidesWriter(MPI_Comm comm, U filename, T mesh)
    -> FidesWriter<typename std::remove_cvref<
        typename T::element_type>::type::geometry_type::value_type>;

} // namespace dolfinx::io

#endif
