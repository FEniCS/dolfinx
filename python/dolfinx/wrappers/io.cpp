// Copyright (C) 2017-2021 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
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
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{

void io(nb::module_& m)
{
  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        nb::arg("cell"), nb::arg("dim"), "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](const dolfinx::mesh::Geometry<double>& x, dolfinx::mesh::CellType cell)
      {
        auto [cells, shape]
            = dolfinx::io::extract_vtk_connectivity(x.dofmap(), cell);
        return as_nbarray(std::move(cells), shape);
      },
      nb::arg("x"), nb::arg("celltype"),
      "Extract the mesh topology with VTK ordering using "
      "geometry indices");

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from VTK to DOLFINx node ordering");
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from Gmsh to DOLFINx node ordering");

  // TODO: Template for different values dtypes
  m.def(
      "distribute_entity_data",
      [](const dolfinx::mesh::Mesh<double>& mesh, int entity_dim,
         const nb::ndarray<std::int64_t>& entities,
         const nb::ndarray<std::int32_t>& values)
      {
        assert(entities.ndim() == 2);
        assert(values.ndim() == 1);
        assert(entities.shape(0) == values.shape(0));
        std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
            entities_values = dolfinx::io::xdmf_utils::distribute_entity_data(
                mesh, entity_dim,
                std::span(entities.data(),
                          entities.shape(0) * entities.shape(1)),
                std::span(values.data(), values.shape(0)));

        std::size_t num_vert_per_entity = dolfinx::mesh::cell_num_entities(
            dolfinx::mesh::cell_entity_type(
                mesh.topology()->cell_types().back(), entity_dim, 0),
            0);
        std::array shape_e{entities_values.first.size() / num_vert_per_entity,
                           num_vert_per_entity};
        return std::pair(as_nbarray(std::move(entities_values.first), shape_e),
                         as_nbarray(std::move(entities_values.second)));
      },
      nb::arg("mesh"), nb::arg("entity_dim"), nb::arg("entities"),
      nb::arg("values"));

  // dolfinx::io::XDMFFile
  nb::class_<dolfinx::io::XDMFFile> xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  nb::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

  xdmf_file
      .def(
          "__init__",
          [](dolfinx::io::XDMFFile* x, const MPICommWrapper comm,
             std::filesystem::path filename, std::string file_mode,
             dolfinx::io::XDMFFile::Encoding encoding) {
            new (x) dolfinx::io::XDMFFile(comm.get(), filename, file_mode,
                                          encoding);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("file_mode"),
          nb::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_mesh", &dolfinx::io::XDMFFile::write_mesh, nb::arg("mesh"),
           nb::arg("xpath"))
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           nb::arg("geometry"), nb::arg("name") = "geometry",
           nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [cells, shape] = self.read_topology_data(name, xpath);
            return as_nbarray(std::move(cells), shape);
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [x, shape] = self.read_geometry_data(name, xpath);
            return as_nbarray(std::move(x), shape);
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("write_function",
           nb::overload_cast<const dolfinx::fem::Function<double, double>&,
                             double, std::string>(
               &dolfinx::io::XDMFFile::write_function),
           nb::arg("function"), nb::arg("t"), nb::arg("mesh_xpath"))
      .def("write_function",
           nb::overload_cast<
               const dolfinx::fem::Function<std::complex<double>, double>&,
               double, std::string>(&dolfinx::io::XDMFFile::write_function),
           nb::arg("function"), nb::arg("t"), nb::arg("mesh_xpath"))
      .def("write_meshtags", &dolfinx::io::XDMFFile::write_meshtags,
           nb::arg("meshtags"), nb::arg("x"),
           nb::arg("geometry_xpath") = "/Xdmf/Domain/Grid/Geometry",
           nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_meshtags", &dolfinx::io::XDMFFile::read_meshtags,
           nb::arg("mesh"), nb::arg("name"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("write_information", &dolfinx::io::XDMFFile::write_information,
           nb::arg("name"), nb::arg("value"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_information", &dolfinx::io::XDMFFile::read_information,
           nb::arg("name"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("comm", [](dolfinx::io::XDMFFile& self)
           { return MPICommWrapper(self.comm()); });

  // dolfinx::io::VTKFile
  nb::class_<dolfinx::io::VTKFile>(m, "VTKFile")
      .def(
          "__init__",
          [](dolfinx::io::VTKFile* v, MPICommWrapper comm,
             std::filesystem::path filename, std::string mode)
          { new (v) dolfinx::io::VTKFile(comm.get(), filename, mode); },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mode"))
      .def("close", &dolfinx::io::VTKFile::close)
      .def("write", &dolfinx::io::VTKFile::write<double>, nb::arg("u"),
           nb::arg("t") = 0.0)
      .def("write", &dolfinx::io::VTKFile::write<std::complex<double>>,
           nb::arg("u"), nb::arg("t") = 0.0)
      .def("write",
           static_cast<void (dolfinx::io::VTKFile::*)(
               const dolfinx::mesh::Mesh<double>&, double)>(
               &dolfinx::io::VTKFile::write),
           nb::arg("mesh"), nb::arg("t") = 0.0);

#ifdef HAS_ADIOS2
  // dolfinx::io::FidesWriter
  nb::class_<dolfinx::io::FidesWriter<double>,
             std::shared_ptr<dolfinx::io::FidesWriter<double>>>
      fides_writer(m, "FidesWriter", "FidesWriter object");

  nb::enum_<dolfinx::io::FidesMeshPolicy>(fides_writer, "FidesMeshPolicy")
      .value("update", dolfinx::io::FidesMeshPolicy::update)
      .value("reuse", dolfinx::io::FidesMeshPolicy::reuse);

  fides_writer
      .def(
          "__init__",
          [](dolfinx::io::FidesWriter<double>* fw, MPICommWrapper comm,
             std::filesystem::path filename,
             std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
             std::string engine)
          {
            new (fw) dolfinx::io::FidesWriter<double>(comm.get(), filename,
                                                      mesh, engine);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mesh"),
          nb::arg("engine") = "BPFile")
      .def(
          "__init__",
          [](dolfinx::io::FidesWriter<double>* fw, MPICommWrapper comm,
             std::filesystem::path filename,
             const std::vector<std::variant<
                 std::shared_ptr<const dolfinx::fem::Function<float, double>>,
                 std::shared_ptr<const dolfinx::fem::Function<double, double>>,
                 std::shared_ptr<
                     const dolfinx::fem::Function<std::complex<float>, double>>,
                 std::shared_ptr<const dolfinx::fem::Function<
                     std::complex<double>, double>>>>& u,
             std::string engine, dolfinx::io::FidesMeshPolicy policy) {
            new (fw) dolfinx::io::FidesWriter<double>(comm.get(), filename, u,
                                                      policy);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("u"),
          nb::arg("engine") = "BPFile",
          nb::arg("policy") = dolfinx::io::FidesMeshPolicy::update)
      .def("close",
           [](dolfinx::io::FidesWriter<double>& self) { self.close(); })
      .def(
          "write",
          [](dolfinx::io::FidesWriter<double>& self, double t)
          { self.write(t); },
          nb::arg("t"));

  // dolfinx::io::VTXWriter
  nb::class_<dolfinx::io::VTXWriter<double>>(m, "VTXWriter", "VTXWriter object")
      .def(
          "__init__",
          [](dolfinx::io::VTXWriter<double>* v, MPICommWrapper comm,
             std::filesystem::path filename,
             std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
             std::string engine) {
            new (v) dolfinx::io::VTXWriter<double>(comm.get(), filename, mesh,
                                                   engine);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mesh"),
          nb::arg("engine") = "BPFile")
      .def(
          "__init__",
          [](dolfinx::io::VTXWriter<double>* v, MPICommWrapper comm,
             std::filesystem::path filename,
             const std::vector<std::variant<
                 std::shared_ptr<const dolfinx::fem::Function<float, double>>,
                 std::shared_ptr<const dolfinx::fem::Function<double, double>>,
                 std::shared_ptr<
                     const dolfinx::fem::Function<std::complex<float>, double>>,
                 std::shared_ptr<const dolfinx::fem::Function<
                     std::complex<double>, double>>>>& u,
             std::string engine) {
            new (v)
                dolfinx::io::VTXWriter<double>(comm.get(), filename, u, engine);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("u"),
          nb::arg("engine") = "BPFile")
      .def("close", [](dolfinx::io::VTXWriter<double>& self) { self.close(); })
      .def(
          "write",
          [](dolfinx::io::VTXWriter<double>& self, double t) { self.write(t); },
          nb::arg("t"));

#endif
}
} // namespace dolfinx_wrappers
