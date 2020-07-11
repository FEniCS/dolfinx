// Copyright (C) 2017-2019 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/xdmf_utils.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void io(py::module& m)
{

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk);
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh);

  // TODO: Template for different values dtypes
  m.def("extract_local_entities",
        [](const dolfinx::mesh::Mesh& mesh, const int entity_dim,
           const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& entities,
           const py::array_t<std::int32_t>& values) {
          py::buffer_info buf = values.request();
          std::vector<std::int32_t> vals((std::int32_t*)buf.ptr,
                                         (std::int32_t*)buf.ptr + buf.size);
          return dolfinx::io::xdmf_utils::extract_local_entities(
              mesh, entity_dim, entities, vals);
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
      .def("read_topology_data", &dolfinx::io::XDMFFile::read_topology_data,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("write_function", &dolfinx::io::XDMFFile::write_function,
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
  py::class_<dolfinx::io::VTKFile, std::shared_ptr<dolfinx::io::VTKFile>>
      vtk_file(m, "VTKFile");

  vtk_file
      .def(py::init([](std::string filename) {
             return std::make_unique<dolfinx::io::VTKFile>(filename);
           }),
           py::arg("filename"))
      .def("write",
           py::overload_cast<const dolfinx::function::Function<PetscScalar>&>(
               &dolfinx::io::VTKFile::write),
           py::arg("u"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::Mesh&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"));
}
} // namespace dolfinx_wrappers
