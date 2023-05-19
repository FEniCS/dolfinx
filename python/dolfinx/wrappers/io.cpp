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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <string>
#include <vector>

#include <petscsystypes.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void io(py::module& m)
{
  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        py::arg("cell"), py::arg("dim"), "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](const dolfinx::mesh::Geometry<double>& x, dolfinx::mesh::CellType cell)
      {
        auto [cells, shape]
            = dolfinx::io::extract_vtk_connectivity(x.dofmap(), cell);
        return as_pyarray(std::move(cells), shape);
      },
      py::arg("x"), py::arg("celltype"),
      "Extract the mesh topology with VTK ordering using "
      "geometry indices");

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk, py::arg("type"),
        py::arg("num_nodes"),
        "Permutation array to map from VTK to DOLFINx node ordering");
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh, py::arg("type"),
        py::arg("num_nodes"),
        "Permutation array to map from Gmsh to DOLFINx node ordering");

  // TODO: Template for different values dtypes
  m.def(
      "distribute_entity_data",
      [](const dolfinx::mesh::Mesh<double>& mesh, int entity_dim,
         const py::array_t<std::int64_t, py::array::c_style>& entities,
         const py::array_t<std::int32_t, py::array::c_style>& values)
      {
        assert(entities.ndim() == 2);
        assert(values.ndim() == 1);
        assert(entities.shape(0) == values.shape(0));
        std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
            entities_values = dolfinx::io::xdmf_utils::distribute_entity_data(
                mesh, entity_dim, std::span(entities.data(), entities.size()),
                std::span(values.data(), values.size()));

        std::size_t num_vert_per_entity = dolfinx::mesh::cell_num_entities(
            dolfinx::mesh::cell_entity_type(
                mesh.topology()->cell_types().back(), entity_dim, 0),
            0);
        std::array shape_e{entities_values.first.size() / num_vert_per_entity,
                           num_vert_per_entity};
        return std::pair(as_pyarray(std::move(entities_values.first), shape_e),
                         as_pyarray(std::move(entities_values.second)));
      },
      py::arg("mesh"), py::arg("entity_dim"), py::arg("entities"),
      py::arg("values"));

  // dolfinx::io::XDMFFile
  py::class_<dolfinx::io::XDMFFile, std::shared_ptr<dolfinx::io::XDMFFile>>
      xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  py::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

  xdmf_file
      .def(py::init(
               [](const MPICommWrapper comm, std::filesystem::path filename,
                  std::string file_mode,
                  dolfinx::io::XDMFFile::Encoding encoding)
               {
                 return std::make_unique<dolfinx::io::XDMFFile>(
                     comm.get(), filename, file_mode, encoding);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"),
           py::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_mesh", &dolfinx::io::XDMFFile::write_mesh, py::arg("mesh"),
           py::arg("xpath"))
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           py::arg("geometry"), py::arg("name") = "geometry",
           py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [cells, shape] = self.read_topology_data(name, xpath);
            return as_pyarray(std::move(cells), shape);
          },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, std::string name, std::string xpath)
          {
            auto [x, shape] = self.read_geometry_data(name, xpath);
            return as_pyarray(std::move(x), shape);
          },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("write_function",
           py::overload_cast<const dolfinx::fem::Function<double, double>&,
                             double, std::string>(
               &dolfinx::io::XDMFFile::write_function),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_function",
           py::overload_cast<
               const dolfinx::fem::Function<std::complex<double>, double>&,
               double, std::string>(&dolfinx::io::XDMFFile::write_function),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_meshtags", &dolfinx::io::XDMFFile::write_meshtags,
           py::arg("meshtags"), py::arg("x"),
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
               [](MPICommWrapper comm, std::filesystem::path filename,
                  std::string mode) {
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
               const dolfinx::mesh::Mesh<double>&, double)>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"), py::arg("t") = 0.0);

#ifdef HAS_ADIOS2
  // dolfinx::io::FidesWriter
  py::class_<dolfinx::io::FidesWriter<PetscReal>,
             std::shared_ptr<dolfinx::io::FidesWriter<PetscReal>>>
      fides_writer(m, "FidesWriter", "FidesWriter object");

  py::enum_<dolfinx::io::FidesMeshPolicy>(fides_writer, "FidesMeshPolicy")
      .value("update", dolfinx::io::FidesMeshPolicy::update)
      .value("reuse", dolfinx::io::FidesMeshPolicy::reuse);

  fides_writer
      .def(py::init(
               [](MPICommWrapper comm, std::filesystem::path filename,
                  std::shared_ptr<const dolfinx::mesh::Mesh<PetscReal>> mesh,
                  std::string engine)
               {
                 return std::make_unique<dolfinx::io::FidesWriter<PetscReal>>(
                     comm.get(), filename, mesh, engine);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("mesh"),
           py::arg("engine") = "BPFile")
      .def(py::init(
               [](MPICommWrapper comm, std::filesystem::path filename,
                  const std::vector<std::variant<
                      std::shared_ptr<
                          const dolfinx::fem::Function<float, PetscReal>>,
                      std::shared_ptr<
                          const dolfinx::fem::Function<double, PetscReal>>,
                      std::shared_ptr<const dolfinx::fem::Function<
                          std::complex<float>, PetscReal>>,
                      std::shared_ptr<const dolfinx::fem::Function<
                          std::complex<double>, PetscReal>>>>& u,
                  std::string engine, dolfinx::io::FidesMeshPolicy policy)
               {
                 return std::make_unique<dolfinx::io::FidesWriter<PetscReal>>(
                     comm.get(), filename, u, policy);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("u"),
           py::arg("engine") = "BPFile",
           py::arg("policy") = dolfinx::io::FidesMeshPolicy::update)
      .def("close",
           [](dolfinx::io::FidesWriter<PetscReal>& self) { self.close(); })
      .def(
          "write",
          [](dolfinx::io::FidesWriter<PetscReal>& self, double t)
          { self.write(t); },
          py::arg("t"));

  // dolfinx::io::VTXWriter
  py::class_<dolfinx::io::VTXWriter<PetscReal>,
             std::shared_ptr<dolfinx::io::VTXWriter<PetscReal>>>(
      m, "VTXWriter", "VTXWriter object")
      .def(py::init(
               [](MPICommWrapper comm, std::filesystem::path filename,
                  std::shared_ptr<const dolfinx::mesh::Mesh<PetscReal>> mesh,
                  std::string engine)
               {
                 return std::make_unique<dolfinx::io::VTXWriter<PetscReal>>(
                     comm.get(), filename, mesh, engine);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("mesh"),
           py::arg("engine") = "BPFile")
      .def(py::init(
               [](MPICommWrapper comm, std::filesystem::path filename,
                  const std::vector<std::variant<
                      std::shared_ptr<
                          const dolfinx::fem::Function<float, PetscReal>>,
                      std::shared_ptr<
                          const dolfinx::fem::Function<double, PetscReal>>,
                      std::shared_ptr<const dolfinx::fem::Function<
                          std::complex<float>, PetscReal>>,
                      std::shared_ptr<const dolfinx::fem::Function<
                          std::complex<double>, PetscReal>>>>& u,
                  std::string engine)
               {
                 return std::make_unique<dolfinx::io::VTXWriter<PetscReal>>(
                     comm.get(), filename, u, engine);
               }),
           py::arg("comm"), py::arg("filename"), py::arg("u"),
           py::arg("engine") = "BPFile")
      .def("close",
           [](dolfinx::io::VTXWriter<PetscReal>& self) { self.close(); })
      .def(
          "write",
          [](dolfinx::io::VTXWriter<PetscReal>& self, double t)
          { self.write(t); },
          py::arg("t"));

#endif
}
} // namespace dolfinx_wrappers
