// Copyright (C) 2017-2025 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include "array.h"
#include <basix/mdspan.hpp>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <span>
#include <string>
#include <vector>

#ifdef HAS_ADIOS2
#include <dolfinx/io/ADIOS2Writers.h>
#endif

namespace dolfinx_wrappers
{

namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

template <typename T, std::size_t ndim>
using mdspan_t = md::mdspan<const T, md::dextents<std::size_t, ndim>>;

/// Add real-valued mesh functions to XDMFFile
template <typename T>
void xdmf_real_fn(auto&& m)
{
  namespace nb = nanobind;

  m.def(
      "write_mesh",
      [](dolfinx::io::XDMFFile& self, const dolfinx::mesh::Mesh<T>& mesh,
         const std::string& xpath) { self.write_mesh(mesh, xpath); },
      nb::arg("mesh"), nb::arg("xpath") = "/Xdmf/Domain");
  m.def(
      "write_meshtags",
      [](dolfinx::io::XDMFFile& self,
         const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
         const dolfinx::mesh::Geometry<T>& x, const std::string& geometry_xpath,
         const std::string& xpath)
      { self.write_meshtags(meshtags, x, geometry_xpath, xpath); },
      nb::arg("meshtags"), nb::arg("x"), nb::arg("geometry_xpath"),
      nb::arg("xpath") = "/Xdmf/Domain");
}

/// Add scalar function write methods to XDMFFile
template <typename T, typename U>
void xdmf_scalar_fn(auto&& m)
{
  namespace nb = nanobind;

  m.def(
      "write_function",
      [](dolfinx::io::XDMFFile& self, const dolfinx::fem::Function<T, U>& u,
         double t, const std::string& mesh_xpath)
      { self.write_function(u, t, mesh_xpath); },
      nb::arg("u"), nb::arg("t"),
      nb::arg("mesh_xpath") = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");
}

/// Add real-valued mesh write methods to VTKFile
template <typename T>
void vtk_real_fn(auto&& m)
{
  namespace nb = nanobind;

  m.def(
      "write",
      [](dolfinx::io::VTKFile& self, const dolfinx::mesh::Mesh<T>& mesh,
         double t) { self.write(mesh, t); },
      nb::arg("mesh"), nb::arg("t") = 0);
}

/// Add scalar function write methods to VTKFile
template <typename T, typename U>
void vtk_scalar_fn(auto&& m)
{
  namespace nb = nanobind;

  m.def(
      "write",
      [](dolfinx::io::VTKFile& self,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             u_ptr,
         double t)
      {
        std::vector<std::reference_wrapper<const dolfinx::fem::Function<T, U>>>
            u;
        u.reserve(u_ptr.size());
        for (auto& q : u_ptr)
          u.push_back(*q);

        self.write(u, t);
      },
      nb::arg("u"), nb::arg("t") = 0);
}

#ifdef HAS_ADIOS2
/// Declare VTXWriter for a given scalar type
/// @param m The nanobind module
/// @param type String representation of the scalar type (e.g., "float64", "float32")
template <typename T>
void declare_vtx_writer(nanobind::module_& m, const std::string& type)
{
  namespace nb = nanobind;

  std::string pyclass_name = "VTXWriter_" + type;
  nb::class_<dolfinx::io::VTXWriter<T>>(m, pyclass_name.c_str())
      .def(
          "__init__",
          [](dolfinx::io::VTXWriter<T>* self, MPICommWrapper comm,
             std::filesystem::path filename,
             std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
             std::string engine)
          {
            new (self)
                dolfinx::io::VTXWriter<T>(comm.get(), filename, mesh, engine);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mesh"),
          nb::arg("engine"))
      .def(
          "__init__",
          [](dolfinx::io::VTXWriter<T>* self, MPICommWrapper comm,
             std::filesystem::path filename,
             const std::vector<std::variant<
                 std::shared_ptr<const dolfinx::fem::Function<float, T>>,
                 std::shared_ptr<const dolfinx::fem::Function<double, T>>,
                 std::shared_ptr<
                     const dolfinx::fem::Function<std::complex<float>, T>>,
                 std::shared_ptr<const dolfinx::fem::Function<
                     std::complex<double>, T>>>>& u,
             const std::string& engine, dolfinx::io::VTXMeshPolicy policy)
          {
            new (self) dolfinx::io::VTXWriter<T>(comm.get(), filename, u,
                                                 engine, policy);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("u"),
          nb::arg("engine") = "BPFile",
          nb::arg("policy") = dolfinx::io::VTXMeshPolicy::update)
      .def("close", [](dolfinx::io::VTXWriter<T>& self) { self.close(); })
      .def(
          "write", [](dolfinx::io::VTXWriter<T>& self, double t)
          { self.write(t); }, nb::arg("t"));
}
#endif

/// Declare data distribution utility functions for a given scalar type
/// @param m The nanobind module
template <typename T>
void declare_data_types(nanobind::module_& m)
{
  namespace nb = nanobind;

  m.def(
      "distribute_entity_data",
      [](const dolfinx::mesh::Topology& topology,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>
             input_global_indices,
         std::int64_t num_nodes_g,
         const dolfinx::fem::ElementDofLayout& cmap_dof_layout,
         nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> xdofmap,
         int entity_dim,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> entities,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> values)
      {
        assert(entities.shape(0) == values.size());
        mdspan_t<const std::int64_t, 2> entities_span(
            entities.data(), entities.shape(0), entities.shape(1));
        mdspan_t<const std::int32_t, 2> xdofmap_span(
            xdofmap.data(), xdofmap.shape(0), xdofmap.shape(1));

        std::span<const std::int64_t> input_global_indices_span(
            input_global_indices.data(), input_global_indices.size());
        std::pair<std::vector<std::int32_t>, std::vector<T>> entities_values
            = dolfinx::io::distribute_entity_data<T>(
                topology, input_global_indices_span, num_nodes_g,
                cmap_dof_layout, xdofmap_span, entity_dim, entities_span,
                std::span(values.data(), values.size()));

        std::size_t num_vert_per_entity = dolfinx::mesh::cell_num_entities(
            dolfinx::mesh::cell_entity_type(topology.cell_type(), entity_dim,
                                            0),
            0);
        return std::pair(
            as_nbarray(std::move(entities_values.first),
                       {entities_values.first.size() / num_vert_per_entity,
                        num_vert_per_entity}),
            as_nbarray(std::move(entities_values.second)));
      },
      nb::arg("topology"), nb::arg("input_global_indices"),
      nb::arg("num_nodes_g"), nb::arg("cmap_dof_layout"), nb::arg("xdofmap"),
      nb::arg("entity_dim"), nb::arg("entities"),
      nb::arg("values").noconvert());
}

} // namespace dolfinx_wrappers
