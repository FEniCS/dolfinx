// Copyright (C) 2017-2019 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/xdmf_utils.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void io(py::module& m)
{

  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type);

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk);
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh);

  // TODO: Template for different values dtypes
  m.def("extract_local_entities",
        [](const dolfinx::mesh::Mesh& mesh, int entity_dim,
           const py::array_t<std::int64_t, py::array::c_style>& entities,
           const py::array_t<std::int32_t, py::array::c_style>& values) {
          assert(entities.ndim() == 2);
          std::array<std::size_t, 2> shape
              = {static_cast<std::size_t>(entities.shape(0)),
                 static_cast<std::size_t>(entities.shape(1))};
          auto _entities = xt::adapt(entities.data(), entities.size(),
                                     xt::no_ownership(), shape);
          std::pair<xt::xtensor<std::int32_t, 2>, std::vector<std::int32_t>> e
              = dolfinx::io::xdmf_utils::extract_local_entities(
                  mesh, entity_dim, _entities,
                  xtl::span(values.data(), values.size()));
          return std::pair(xt_as_pyarray(std::move(e.first)),
                           as_pyarray(std::move(e.second)));
        });

  // dolfinx::io::XDMFFile
  py::class_<dolfinx::io::XDMFFile, std::shared_ptr<dolfinx::io::XDMFFile>>
      xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  py::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

  xdmf_file
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode,
                       dolfinx::io::XDMFFile::Encoding encoding) {
             return std::make_unique<dolfinx::io::XDMFFile>(
                 comm.get(), filename, file_mode, encoding);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"),
           py::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("__enter__",
           [](std::shared_ptr<dolfinx::io::XDMFFile>& self) { return self; })
      .def("__exit__",
           [](dolfinx::io::XDMFFile& self, py::object exc_type,
              py::object exc_value, py::object traceback) { self.close(); })
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_mesh", &dolfinx::io::XDMFFile::write_mesh, py::arg("mesh"),
           py::arg("xpath") = "/Xdmf/Domain")
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           py::arg("geometry"), py::arg("name") = "geometry",
           py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath) {
            return xt_as_pyarray(self.read_topology_data(name, xpath));
          },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath) {
            return xt_as_pyarray(self.read_geometry_data(name, xpath));
          },
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
      .def("comm", [](dolfinx::io::XDMFFile& self) {
        return MPICommWrapper(self.comm());
      });

  // dolfinx::io::VTKFile
  py::class_<dolfinx::io::VTKFile, std::shared_ptr<dolfinx::io::VTKFile>>(
      m, "VTKFile")
      .def(py::init([](const MPICommWrapper comm, const std::string& filename,
                       const std::string& mode) {
             return std::make_unique<dolfinx::io::VTKFile>(comm.get(),
                                                              filename, mode);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("mode"))
      .def("__enter__",
           [](std::shared_ptr<dolfinx::io::VTKFile>& self) { return self; })
      .def("__exit__",
           [](dolfinx::io::VTKFile& self, py::object exc_type,
              py::object exc_value, py::object traceback) { self.close(); })
      .def("close", &dolfinx::io::VTKFile::close)
      .def("write",
           py::overload_cast<const std::vector<std::reference_wrapper<
                                 const dolfinx::fem::Function<double>>>&,
                             double>(&dolfinx::io::VTKFile::write),
           py::arg("u"), py::arg("t") = 0.0)
      .def("write",
           py::overload_cast<
               const std::vector<std::reference_wrapper<
                   const dolfinx::fem::Function<std::complex<double>>>>&,
               double>(&dolfinx::io::VTKFile::write),
           py::arg("u"), py::arg("t") = 0.0)

      .def("write",
           py::overload_cast<const dolfinx::mesh::Mesh&, double>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"), py::arg("t") = 0.0);
}
} // namespace dolfinx_wrappers
