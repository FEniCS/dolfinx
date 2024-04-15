// Copyright (C) 2017-2021 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <basix/mdspan.hpp>
#include <dolfinx/common/defines.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/vtk_utils.h>
#include <dolfinx/io/xdmf_utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <filesystem>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
namespace
{
template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

template <typename T>
void xdmf_real_fn(auto&& m)
{
  m.def(
      "write_mesh",
      [](dolfinx::io::XDMFFile& self, const dolfinx::mesh::Mesh<T>& mesh,
         std::string xpath) { self.write_mesh(mesh, xpath); },
      nb::arg("mesh"), nb::arg("xpath") = "/Xdmf/Domain");
  m.def(
      "write_meshtags",
      [](dolfinx::io::XDMFFile& self,
         const dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
         const dolfinx::mesh::Geometry<T>& x, std::string geometry_xpath,
         std::string xpath)
      { self.write_meshtags(meshtags, x, geometry_xpath, xpath); },
      nb::arg("meshtags"), nb::arg("x"), nb::arg("geometry_xpath"),
      nb::arg("xpath") = "/Xdmf/Domain");
}

template <typename T, typename U>
void xdmf_scalar_fn(auto&& m)
{
  m.def(
      "write_function",
      [](dolfinx::io::XDMFFile& self, const dolfinx::fem::Function<T, U>& u,
         double t, std::string mesh_xpath)
      { self.write_function(u, t, mesh_xpath); },
      nb::arg("u"), nb::arg("t"),
      nb::arg("mesh_xpath") = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");
}

template <typename T>
void vtk_real_fn(auto&& m)
{
  m.def(
      "write",
      [](dolfinx::io::VTKFile& self, const dolfinx::mesh::Mesh<T>& mesh,
         double t) { self.write(mesh, t); },
      nb::arg("mesh"), nb::arg("t") = 0.0);
}

template <typename T, typename U>
void vtk_scalar_fn(auto&& m)
{
  m.def(
      "write",
      [](dolfinx::io::VTKFile& self,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>
             u_ptr,
         double t)
      {
        std::vector<std::reference_wrapper<const dolfinx::fem::Function<T, U>>>
            u;
        for (auto q : u_ptr)
          u.push_back(*q);

        self.write(u, t);
      },
      nb::arg("u"), nb::arg("t") = 0.0);
}

template <typename T>
void declare_vtx_writer(nb::module_& m, std::string type)
{
#ifdef HAS_ADIOS2
  {
    std::string pyclass_name = "VTXWriter_" + type;
    nb::class_<dolfinx::io::VTXWriter<T>>(m, pyclass_name.c_str())
        .def(
            "__init__",
            [](dolfinx::io::VTXWriter<T>* self, MPICommWrapper comm,
               std::filesystem::path filename,
               std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::string engine) {
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
               std::string engine, dolfinx::io::VTXMeshPolicy policy)
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

  {
    std::string pyclass_name = "FidesWriter_" + type;
    nb::class_<dolfinx::io::FidesWriter<T>>(m, pyclass_name.c_str(),
                                            "FidesWriter object")
        .def(
            "__init__",
            [](dolfinx::io::FidesWriter<T>* self, MPICommWrapper comm,
               std::filesystem::path filename,
               std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::string engine) {
              new (self) dolfinx::io::FidesWriter<T>(comm.get(), filename, mesh,
                                                     engine);
            },
            nb::arg("comm"), nb::arg("filename"), nb::arg("mesh"),
            nb::arg("engine") = "BPFile")
        .def(
            "__init__",
            [](dolfinx::io::FidesWriter<T>* self, MPICommWrapper comm,
               std::filesystem::path filename,
               const std::vector<std::variant<
                   std::shared_ptr<const dolfinx::fem::Function<float, T>>,
                   std::shared_ptr<const dolfinx::fem::Function<double, T>>,
                   std::shared_ptr<
                       const dolfinx::fem::Function<std::complex<float>, T>>,
                   std::shared_ptr<const dolfinx::fem::Function<
                       std::complex<double>, T>>>>& u,
               std::string engine, dolfinx::io::FidesMeshPolicy policy)
            {
              new (self) dolfinx::io::FidesWriter<T>(comm.get(), filename, u,
                                                     engine, policy);
            },
            nb::arg("comm"), nb::arg("filename"), nb::arg("u"),
            nb::arg("engine") = "BPFile",
            nb::arg("policy") = dolfinx::io::FidesMeshPolicy::update)
        .def("close", [](dolfinx::io::FidesWriter<T>& self) { self.close(); })
        .def(
            "write", [](dolfinx::io::FidesWriter<T>& self, double t)
            { self.write(t); }, nb::arg("t"));
  }
#endif
}

template <typename T>
void declare_real_types(nb::module_& m)
{
  // TODO: Remove this template, pass lower level mesh data
  m.def(
      "distribute_entity_data",
      [](const dolfinx::mesh::Mesh<T>& mesh, int entity_dim,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> entities,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> values)
      {
        assert(entities.shape(0) == values.size());
        mdspan_t<const std::int64_t, 2> entities_span(
            entities.data(), entities.shape(0), entities.shape(1));
        std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
            entities_values = dolfinx::io::xdmf_utils::distribute_entity_data(
                *mesh.topology(), mesh.geometry().input_global_indices(),
                mesh.geometry().index_map()->size_global(),
                mesh.geometry().cmap().create_dof_layout(),
                mesh.geometry().dofmap(), entity_dim, entities_span,
                std::span(values.data(), values.size()));

        std::size_t num_vert_per_entity = dolfinx::mesh::cell_num_entities(
            dolfinx::mesh::cell_entity_type(mesh.topology()->cell_type(),
                                            entity_dim, 0),
            0);
        return std::pair(
            as_nbarray(std::move(entities_values.first),
                       {entities_values.first.size() / num_vert_per_entity,
                        num_vert_per_entity}),
            as_nbarray(std::move(entities_values.second)));
      },
      nb::arg("mesh"), nb::arg("entity_dim"), nb::arg("entities"),
      nb::arg("values"));
}

} // namespace

void io(nb::module_& m)
{
  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        nb::arg("cell"), nb::arg("dim"), "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
         dolfinx::mesh::CellType cell)
      {
        mdspan_t<const std::int32_t, 2> _dofmap(dofmap.data(), dofmap.shape(0),
                                                dofmap.shape(1));
        auto [cells, shape]
            = dolfinx::io::extract_vtk_connectivity(_dofmap, cell);
        return as_nbarray(std::move(cells), shape.size(), shape.data());
      },
      nb::arg("dofmap"), nb::arg("celltype"),
      "Extract the mesh topology with VTK ordering using "
      "geometry indices");

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from VTK to DOLFINx node ordering");
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from Gmsh to DOLFINx node ordering");

  // dolfinx::io::XDMFFile
  nb::class_<dolfinx::io::XDMFFile> xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  nb::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5, "HDF5 encoding")
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII,
             "Plain text encoding");

  xdmf_file
      .def(
          "__init__",
          [](dolfinx::io::XDMFFile* x, MPICommWrapper comm,
             std::filesystem::path filename, std::string file_mode,
             dolfinx::io::XDMFFile::Encoding encoding) {
            new (x) dolfinx::io::XDMFFile(comm.get(), filename, file_mode,
                                          encoding);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("file_mode"),
          nb::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           nb::arg("geometry"), nb::arg("name") = "geometry",
           nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [cells, shape] = self.read_topology_data(name, xpath);
            return as_nbarray(std::move(cells), shape.size(), shape.data());
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [x, shape] = self.read_geometry_data(name, xpath);
            std::vector<double>& _x = std::get<std::vector<double>>(x);
            return as_nbarray(std::move(_x), shape.size(), shape.data());
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_meshtags", &dolfinx::io::XDMFFile::read_meshtags,
           nb::arg("mesh"), nb::arg("name"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("write_information", &dolfinx::io::XDMFFile::write_information,
           nb::arg("name"), nb::arg("value"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_information", &dolfinx::io::XDMFFile::read_information,
           nb::arg("name"), nb::arg("xpath") = "/Xdmf/Domain")
      .def_prop_ro(
          "comm", [](dolfinx::io::XDMFFile& self)
          { return MPICommWrapper(self.comm()); }, nb::keep_alive<0, 1>());

  xdmf_real_fn<float>(xdmf_file);
  xdmf_real_fn<double>(xdmf_file);
  xdmf_scalar_fn<float, float>(xdmf_file);
  xdmf_scalar_fn<double, double>(xdmf_file);
  xdmf_scalar_fn<std::complex<float>, float>(xdmf_file);
  xdmf_scalar_fn<std::complex<double>, double>(xdmf_file);

  // dolfinx::io::VTKFile
  nb::class_<dolfinx::io::VTKFile> vtk_file(m, "VTKFile");
  vtk_file
      .def(
          "__init__",
          [](dolfinx::io::VTKFile* v, MPICommWrapper comm,
             std::filesystem::path filename, std::string mode)
          { new (v) dolfinx::io::VTKFile(comm.get(), filename, mode); },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mode"))
      .def("close", &dolfinx::io::VTKFile::close);

  vtk_real_fn<float>(vtk_file);
  vtk_real_fn<double>(vtk_file);
  vtk_scalar_fn<float, float>(vtk_file);
  vtk_scalar_fn<double, double>(vtk_file);
  vtk_scalar_fn<std::complex<float>, float>(vtk_file);
  vtk_scalar_fn<std::complex<double>, double>(vtk_file);

#ifdef HAS_ADIOS2
  nb::enum_<dolfinx::io::FidesMeshPolicy>(m, "FidesMeshPolicy")
      .value("update", dolfinx::io::FidesMeshPolicy::update)
      .value("reuse", dolfinx::io::FidesMeshPolicy::reuse);

  nb::enum_<dolfinx::io::VTXMeshPolicy>(m, "VTXMeshPolicy")
      .value("update", dolfinx::io::VTXMeshPolicy::update)
      .value("reuse", dolfinx::io::VTXMeshPolicy::reuse);
#endif

  declare_vtx_writer<float>(m, "float32");
  declare_vtx_writer<double>(m, "float64");

  declare_real_types<float>(m);
  declare_real_types<double>(m);
}
} // namespace dolfinx_wrappers
