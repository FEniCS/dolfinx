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

namespace py = pybind11;

namespace dolfinx_wrappers
{

namespace
{
template <typename T>
void declare_vtx_writer(py::module& m, std::string type)
{
#ifdef HAS_ADIOS2
  {
    std::string pyclass_name = "VTXWriter_" + type;
    py::class_<dolfinx::io::VTXWriter<T>,
               std::shared_ptr<dolfinx::io::VTXWriter<T>>>(
        m, pyclass_name.c_str(), "VTXWriter object")
        .def(py::init(
                 [](MPICommWrapper comm, std::filesystem::path filename,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                    std::string engine)
                 {
                   return std::make_unique<dolfinx::io::VTXWriter<T>>(
                       comm.get(), filename, mesh, engine);
                 }),
             py::arg("comm"), py::arg("filename"), py::arg("mesh"),
             py::arg("engine") = "BPFile")
        .def(
            py::init(
                [](MPICommWrapper comm, std::filesystem::path filename,
                   const std::vector<std::variant<
                       std::shared_ptr<const dolfinx::fem::Function<float, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<double, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<
                           std::complex<float>, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<
                           std::complex<double>, T>>>>& u,
                   std::string engine)
                {
                  return std::make_unique<dolfinx::io::VTXWriter<T>>(
                      comm.get(), filename, u, engine);
                }),
            py::arg("comm"), py::arg("filename"), py::arg("u"),
            py::arg("engine") = "BPFile")
        .def("close", [](dolfinx::io::VTXWriter<T>& self) { self.close(); })
        .def(
            "write",
            [](dolfinx::io::VTXWriter<T>& self, double t) { self.write(t); },
            py::arg("t"));
  }

  {
    std::string pyclass_name = "FidesWriter_" + type;
    py::class_<dolfinx::io::FidesWriter<T>,
               std::shared_ptr<dolfinx::io::FidesWriter<T>>>(
        m, pyclass_name.c_str(), "FidesWriter object")
        .def(py::init(
                 [](MPICommWrapper comm, std::filesystem::path filename,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                    std::string engine)
                 {
                   return std::make_unique<dolfinx::io::FidesWriter<T>>(
                       comm.get(), filename, mesh, engine);
                 }),
             py::arg("comm"), py::arg("filename"), py::arg("mesh"),
             py::arg("engine") = "BPFile")
        .def(
            py::init(
                [](MPICommWrapper comm, std::filesystem::path filename,
                   const std::vector<std::variant<
                       std::shared_ptr<const dolfinx::fem::Function<float, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<double, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<
                           std::complex<float>, T>>,
                       std::shared_ptr<const dolfinx::fem::Function<
                           std::complex<double>, T>>>>& u,
                   std::string engine, dolfinx::io::FidesMeshPolicy policy)
                {
                  return std::make_unique<dolfinx::io::FidesWriter<T>>(
                      comm.get(), filename, u, policy);
                }),
            py::arg("comm"), py::arg("filename"), py::arg("u"),
            py::arg("engine") = "BPFile",
            py::arg("policy") = dolfinx::io::FidesMeshPolicy::update)
        .def("close", [](dolfinx::io::FidesWriter<T>& self) { self.close(); })
        .def(
            "write",
            [](dolfinx::io::FidesWriter<T>& self, double t) { self.write(t); },
            py::arg("t"));
  }
#endif
}
} // namespace

void io(py::module& m)
{
  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        py::arg("cell"), py::arg("dim"), "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](py::array_t<std::int32_t, py::array::c_style> dofmap,
         dolfinx::mesh::CellType cell)
      {
        if (dofmap.ndim() != 2)
          throw std::runtime_error("Geometry dofmap must be rank 2.");
        std::experimental::mdspan<const std::int32_t,
                                  std::experimental::dextents<std::size_t, 2>>
            _dofmap(dofmap.data(), dofmap.shape(0), dofmap.shape(1));
        auto [cells, shape]
            = dolfinx::io::extract_vtk_connectivity(_dofmap, cell);
        return as_pyarray(std::move(cells), shape);
      },
      py::arg("dofmap"), py::arg("celltype"),
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
      .def("write_mesh",
           static_cast<void (dolfinx::io::XDMFFile::*)(
               const dolfinx::mesh::Mesh<double>&, std::string)>(
               &dolfinx::io::XDMFFile::write_mesh),
           py::arg("mesh"), py::arg("xpath"))
      .def("write_mesh",
           static_cast<void (dolfinx::io::XDMFFile::*)(
               const dolfinx::mesh::Mesh<float>&, std::string)>(
               &dolfinx::io::XDMFFile::write_mesh),
           py::arg("mesh"), py::arg("xpath"))
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
            std::vector<double>& _x = std::get<std::vector<double>>(x);
            return as_pyarray(std::move(_x), shape);
          },
          py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           py::arg("name") = "mesh", py::arg("xpath") = "/Xdmf/Domain")
      .def("write_function",
           py::overload_cast<const dolfinx::fem::Function<float, float>&,
                             double, std::string>(
               &dolfinx::io::XDMFFile::write_function<float, float>),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_function",
           py::overload_cast<const dolfinx::fem::Function<double, double>&,
                             double, std::string>(
               &dolfinx::io::XDMFFile::write_function<double, double>),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_function",
           py::overload_cast<
               const dolfinx::fem::Function<std::complex<float>, float>&,
               double, std::string>(
               &dolfinx::io::XDMFFile::write_function<std::complex<float>,
                                                      float>),
           py::arg("function"), py::arg("t"), py::arg("mesh_xpath"))
      .def("write_function",
           py::overload_cast<
               const dolfinx::fem::Function<std::complex<double>, double>&,
               double, std::string>(
               &dolfinx::io::XDMFFile::write_function<std::complex<double>,
                                                      double>),
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
      .def("write", &dolfinx::io::VTKFile::write<float, float>, py::arg("u"),
           py::arg("t") = 0.0)
      .def("write", &dolfinx::io::VTKFile::write<double, double>, py::arg("u"),
           py::arg("t") = 0.0)
      .def("write", &dolfinx::io::VTKFile::write<std::complex<double>, double>,
           py::arg("u"), py::arg("t") = 0.0)
      .def("write",
           static_cast<void (dolfinx::io::VTKFile::*)(
               const dolfinx::mesh::Mesh<float>&, double)>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"), py::arg("t") = 0.0)
      .def("write",
           static_cast<void (dolfinx::io::VTKFile::*)(
               const dolfinx::mesh::Mesh<double>&, double)>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"), py::arg("t") = 0.0);

#ifdef HAS_ADIOS2
  py::enum_<dolfinx::io::FidesMeshPolicy>(m, "FidesMeshPolicy")
      .value("update", dolfinx::io::FidesMeshPolicy::update)
      .value("reuse", dolfinx::io::FidesMeshPolicy::reuse);
#endif

  declare_vtx_writer<float>(m, "float32");
  declare_vtx_writer<double>(m, "float64");
}
} // namespace dolfinx_wrappers
