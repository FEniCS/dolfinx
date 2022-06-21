// Copyright (C) 2017-2021 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void io(py::module& m)
{
  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](const dolfinx::mesh::Mesh& mesh)
      { return xt_as_pyarray(dolfinx::io::extract_vtk_connectivity(mesh)); },
      "Extract the mesh topology with VTK ordering using geometry indices");

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk,
        "Permutation array to map from VTK to DOLFINx node ordering");
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh,
        "Permutation array to map from Gmsh to DOLFINx node ordering");

  // TODO: Template for different values dtypes
  m.def("distribute_entity_data",
        [](const dolfinx::mesh::Mesh& mesh, int entity_dim,
           const py::array_t<std::int64_t, py::array::c_style>& entities,
           const py::array_t<std::int32_t, py::array::c_style>& values)
        {
          assert(entities.ndim() == 2);
          assert(values.ndim() == 1);
          assert(entities.shape(0) == values.shape(0));
          std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
              entities_values = dolfinx::io::xdmf_utils::distribute_entity_data(
                  mesh, entity_dim, xtl::span(entities.data(), entities.size()),
                  xtl::span(values.data(), values.size()));

          std::size_t num_vert_per_entity = dolfinx::mesh::cell_num_entities(
              dolfinx::mesh::cell_entity_type(mesh.topology().cell_type(),
                                              entity_dim, 0),
              0);
          std::array shape_e
              = {entities_values.first.size() / num_vert_per_entity,
                 num_vert_per_entity};
          return std::pair(
              as_pyarray(std::move(entities_values.first), shape_e),
              as_pyarray(std::move(entities_values.second)));
        });

  // dolfinx::io::XDMFFile
  py::class_<dolfinx::io::XDMFFile, std::shared_ptr<dolfinx::io::XDMFFile>>
      xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  py::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

  xdmf_file
      .def(py::init(
               [](const MPICommWrapper comm,
                  const std::filesystem::path& filename,
                  const std::string& file_mode,
                  dolfinx::io::XDMFFile::Encoding encoding)
               {
                 return std::make_unique<dolfinx::io::XDMFFile>(
                     comm.get(), filename, file_mode, encoding);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"),
           py::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_mesh", &dolfinx::io::XDMFFile::write_mesh, py::arg("mesh"),
           py::arg("xpath") = "/Xdmf/Domain")
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           py::arg("geometry"), py::arg("name") = "geometry",
           py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath)
          { return xt_as_pyarray(self.read_topology_data(name, xpath)); },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath)
          { return xt_as_pyarray(self.read_geometry_data(name, xpath)); },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("write_function",
           py::overload_cast<const dolfinx::fem::Function<double>&, double,
                             const std::string&>(
               &dolfinx::io::XDMFFile::write_function),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def(
          "write_function",
          py::overload_cast<const dolfinx::fem::Function<std::complex<double>>&,
                            double, const std::string&>(
              &dolfinx::io::XDMFFile::write_function),
          py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_meshtags", &dolfinx::io::XDMFFile::write_meshtags,
           py::arg("meshtags"),
           py::arg("geometry_xpath") = "/Xdmf/Domain/Grid/Geometry",
           py::arg("xpath") = "/Xdmf/Domain")
      .def("read_meshtags", &dolfinx::io::XDMFFile::read_meshtags,
           py::arg("mesh"), py::arg("name"), py::arg("xpath") = "/Xdmf/Domain")
      .def("write_information", &dolfinx::io::XDMFFile::write_information,
           py::arg("name"), py::arg("value"), py::arg("xpath") = "/Xdmf/Domain")
      .def("read_information", &dolfinx::io::XDMFFile::read_information,
           py::arg("name"), py::arg("xpath") = "/Xdmf/Domain")
      .def("comm", [](dolfinx::io::XDMFFile& self)
           { return MPICommWrapper(self.comm()); });

  // dolfinx::io::VTKFile
  py::class_<dolfinx::io::VTKFile, std::shared_ptr<dolfinx::io::VTKFile>>(
      m, "VTKFile")
      .def(py::init(
               [](const MPICommWrapper comm,
                  const std::filesystem::path& filename,
                  const std::string& mode) {
                 return std::make_unique<dolfinx::io::VTKFile>(comm.get(),
                                                               filename, mode);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("mode"))
      .def("close", &dolfinx::io::VTKFile::close)
      .def("write", &dolfinx::io::VTKFile::write<double>, py::arg("u"),
           py::arg("t") = 0.0)
      .def("write", &dolfinx::io::VTKFile::write<std::complex<double>>,
           py::arg("u"), py::arg("t") = 0.0)
      .def("write",
           static_cast<void (dolfinx::io::VTKFile::*)(
               const dolfinx::mesh::Mesh&, double)>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"), py::arg("t") = 0.0);

#ifdef HAS_ADIOS2
  // dolfinx::io::FidesWriter
  std::string pyclass_name = std::string("FidesWriter");
  py::class_<dolfinx::io::FidesWriter,
             std::shared_ptr<dolfinx::io::FidesWriter>>(m, pyclass_name.c_str(),
                                                        "FidesWriter object")
      .def(py::init(
          [](const MPICommWrapper comm, const std::filesystem::path& filename,
             std::shared_ptr<const dolfinx::mesh::Mesh> mesh)
          {
            return std::make_unique<dolfinx::io::FidesWriter>(comm.get(),
                                                              filename, mesh);
          }))
      .def(py::init(
          [](const MPICommWrapper comm, const std::filesystem::path& filename,
             const std::vector<std::variant<
                 std::shared_ptr<const dolfinx::fem::Function<double>>,
                 std::shared_ptr<
                     const dolfinx::fem::Function<std::complex<double>>>>>& u)
          {
            return std::make_unique<dolfinx::io::FidesWriter>(comm.get(),
                                                              filename, u);
          }))
      .def("close", [](dolfinx::io::FidesWriter& self) { self.close(); })
      .def("write",
           [](dolfinx::io::FidesWriter& self, double t) { self.write(t); });

  // dolfinx::io::VTXWriter
  pyclass_name = std::string("VTXWriter");
  py::class_<dolfinx::io::VTXWriter, std::shared_ptr<dolfinx::io::VTXWriter>>(
      m, pyclass_name.c_str(), "VTXWriter object")
      .def(py::init(
          [](const MPICommWrapper comm, const std::filesystem::path& filename,
             std::shared_ptr<const dolfinx::mesh::Mesh> mesh)
          {
            return std::make_unique<dolfinx::io::VTXWriter>(comm.get(),
                                                            filename, mesh);
          }))
      .def(py::init(
          [](const MPICommWrapper comm, const std::filesystem::path& filename,
             const std::vector<std::variant<
                 std::shared_ptr<const dolfinx::fem::Function<double>>,
                 std::shared_ptr<
                     const dolfinx::fem::Function<std::complex<double>>>>>& u) {
            return std::make_unique<dolfinx::io::VTXWriter>(comm.get(),
                                                            filename, u);
          }))
      .def("close", [](dolfinx::io::VTXWriter& self) { self.close(); })
      .def("write",
           [](dolfinx::io::VTXWriter& self, double t) { self.write(t); });

#endif
}
} // namespace dolfinx_wrappers
