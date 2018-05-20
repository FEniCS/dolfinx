// Copyright (C) 2017 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/io/HDF5File.h>
#include <dolfin/io/VTKFile.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{

void io(py::module& m)
{
  // dolfin::io::VTKFile
  py::class_<dolfin::io::VTKFile, std::shared_ptr<dolfin::io::VTKFile>>(
      m, "VTKFile")
      .def(py::init<std::string>())
      .def("write",
           [](dolfin::io::VTKFile& instance, const dolfin::mesh::Mesh& mesh) {
             instance.write(mesh);
           });

  // dolfin::io::HDF5File
  py::class_<dolfin::io::HDF5File, std::shared_ptr<dolfin::io::HDF5File>,
             dolfin::common::Variable>(m, "HDF5File", py::dynamic_attr())
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode) {
             return std::make_unique<dolfin::io::HDF5File>(comm.get(), filename,
                                                           file_mode);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"))
      .def("__enter__", [](dolfin::io::HDF5File& self) { return &self; })
      .def("__exit__", [](dolfin::io::HDF5File& self, py::args args,
                          py::kwargs kwargs) { self.close(); })
      .def("close", &dolfin::io::HDF5File::close)
      .def("flush", &dolfin::io::HDF5File::flush)
      // read
      .def("read_mesh",
           [](dolfin::io::HDF5File& self, const MPICommWrapper comm,
              const std::string data_path, bool use_partition_from_file,
              const dolfin::mesh::GhostMode ghost_mode) {
             return self.read_mesh(comm.get(), data_path,
                                   use_partition_from_file, ghost_mode);
           })
      .def("read_vector",
           [](dolfin::io::HDF5File& self, const MPICommWrapper comm,
              const std::string data_path, bool use_partition_from_file) {
             return self.read_vector(comm.get(), data_path,
                                     use_partition_from_file);
           })

      //   .def("read",
      //        (void (dolfin::io::HDF5File::*)(
      //            dolfin::mesh::MeshValueCollection<bool>&, std::string)
      //            const) & dolfin::io::HDF5File::read,
      //        py::arg("mvc"), py::arg("name"))
      //   .def("read",
      //        (void (dolfin::io::HDF5File::*)(
      //            dolfin::mesh::MeshValueCollection<std::size_t>&,
      //            std::string)
      //             const)
      //            & dolfin::io::HDF5File::read,
      //        py::arg("mvc"), py::arg("name"))
      //   .def("read",
      //        (void (dolfin::io::HDF5File::*)(
      //            dolfin::mesh::MeshValueCollection<double>&, std::string)
      //            const) & dolfin::io::HDF5File::read,
      //        py::arg("mvc"), py::arg("name"))

      .def("read_mf_bool", &dolfin::io::HDF5File::read_mf_bool, py::arg("mesh"),
           py::arg("name"))
      .def("read_mf_int", &dolfin::io::HDF5File::read_mf_int, py::arg("mesh"),
           py::arg("name"))
      .def("read_mf_size_t", &dolfin::io::HDF5File::read_mf_size_t,
           py::arg("mesh"), py::arg("name"))
      .def("read_mf_double", &dolfin::io::HDF5File::read_mf_double,
           py::arg("mesh"), py::arg("name"))
      //
      .def("read_mvc_bool", &dolfin::io::HDF5File::read_mvc_bool,
           py::arg("mesh"), py::arg("name"))
      .def("read_mvc_size_t", &dolfin::io::HDF5File::read_mvc_size_t,
           py::arg("mesh"), py::arg("name"))
      .def("read_mvc_double", &dolfin::io::HDF5File::read_mvc_double,
           py::arg("mesh"), py::arg("name"))

      //   .def("read",
      //        (void
      //        (dolfin::io::HDF5File::*)(dolfin::mesh::MeshFunction<bool>&,
      //                                        std::string) const)
      //            & dolfin::io::HDF5File::read,
      //        py::arg("meshfunction"), py::arg("name"))
      //   .def("read",
      //        (void (dolfin::io::HDF5File::*)(
      //            dolfin::mesh::MeshFunction<std::size_t>&, std::string)
      //            const) & dolfin::io::HDF5File::read,
      //        py::arg("meshfunction"), py::arg("name"))
      //   .def("read",
      //        (void
      //        (dolfin::io::HDF5File::*)(dolfin::mesh::MeshFunction<int>&,
      //                                        std::string) const)
      //            & dolfin::io::HDF5File::read,
      //        py::arg("meshfunction"), py::arg("name"))
      //   .def("read",
      //        (void
      //        (dolfin::io::HDF5File::*)(dolfin::mesh::MeshFunction<double>&,
      //                                        std::string) const)
      //            & dolfin::io::HDF5File::read,
      //        py::arg("meshfunction"), py::arg("name"))

      .def("read",
           py::overload_cast<
               std::shared_ptr<const dolfin::function::FunctionSpace>,
               const std::string>(&dolfin::io::HDF5File::read, py::const_),
           py::arg("V"), py::arg("name"))
      // write
      .def("write", (void (dolfin::io::HDF5File::*)(const dolfin::mesh::Mesh&,
                                                    std::string))
                        & dolfin::io::HDF5File::write)
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshValueCollection<bool>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshValueCollection<std::size_t>&,
               std::string))
               & dolfin::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshValueCollection<double>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshFunction<bool>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshFunction<std::size_t>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshFunction<int>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(
               const dolfin::mesh::MeshFunction<double>&, std::string))
               & dolfin::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(const dolfin::la::PETScVector&,
                                           std::string))
               & dolfin::io::HDF5File::write,
           py::arg("vector"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(const dolfin::function::Function&,
                                           std::string))
               & dolfin::io::HDF5File::write,
           py::arg("u"), py::arg("name"))
      .def("write",
           (void (dolfin::io::HDF5File::*)(const dolfin::function::Function&,
                                           std::string, double))
               & dolfin::io::HDF5File::write,
           py::arg("u"), py::arg("name"), py::arg("t"))
      .def("write",
           [](dolfin::io::HDF5File& self, py::object u, std::string name) {
             auto _u
                 = u.attr("_cpp_object").cast<dolfin::function::Function*>();
             self.write(*_u, name);
           },
           py::arg("u"), py::arg("name"))
      .def("write",
           [](dolfin::io::HDF5File& self, py::object u, std::string name,
              double t) {
             auto _u
                 = u.attr("_cpp_object").cast<dolfin::function::Function*>();
             self.write(*_u, name, t);
           },
           py::arg("u"), py::arg("name"), py::arg("t"))
      .def("set_mpi_atomicity", &dolfin::io::HDF5File::set_mpi_atomicity)
      .def("get_mpi_atomicity", &dolfin::io::HDF5File::get_mpi_atomicity)
      // others
      .def("has_dataset", &dolfin::io::HDF5File::has_dataset);

  // dolfin::io::XDMFFile
  py::class_<dolfin::io::XDMFFile, std::shared_ptr<dolfin::io::XDMFFile>,
             dolfin::common::Variable>
      xdmf_file(m, "XDMFFile");

  xdmf_file
      .def(py::init([](const MPICommWrapper comm, std::string filename) {
             return std::make_unique<dolfin::io::XDMFFile>(comm.get(),
                                                           filename);
           }),
           py::arg("comm"), py::arg("filename"))
      .def(py::init<std::string>())
      .def("close", &dolfin::io::XDMFFile::close)
      .def("__enter__", [](dolfin::io::XDMFFile& self) { return &self; })
      .def("__exit__", [](dolfin::io::XDMFFile& self, py::args args,
                          py::kwargs kwargs) { self.close(); });

  // dolfin::io::XDMFFile::Encoding enums
  py::enum_<dolfin::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfin::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfin::io::XDMFFile::Encoding::ASCII);

  // dolfin::io::XDMFFile::write
  xdmf_file
      // Function
      .def("write",
           (void (dolfin::io::XDMFFile::*)(const dolfin::function::Function&,
                                           dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("u"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(const dolfin::function::Function&,
                                           double,
                                           dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("u"), py::arg("t"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // Mesh
      .def("write",
           (void (dolfin::io::XDMFFile::*)(const dolfin::mesh::Mesh&,
                                           dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mesh"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // MeshFunction
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshFunction<bool>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshFunction<std::size_t>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshFunction<int>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshFunction<double>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // MeshValueCollection
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshValueCollection<bool>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshValueCollection<std::size_t>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshValueCollection<int>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::io::XDMFFile::*)(
               const dolfin::mesh::MeshValueCollection<double>&,
               dolfin::io::XDMFFile::Encoding))
               & dolfin::io::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // Points
      .def("write",
           [](dolfin::io::XDMFFile& instance, py::list points,
              dolfin::io::XDMFFile::Encoding encoding) {
             auto _points = points.cast<std::vector<dolfin::geometry::Point>>();
             instance.write(_points, encoding);
           },
           py::arg("points"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // Points with values
      .def("write",
           [](dolfin::io::XDMFFile& instance, py::list points,
              std::vector<double>& values,
              dolfin::io::XDMFFile::Encoding encoding) {
             auto _points = points.cast<std::vector<dolfin::geometry::Point>>();
             instance.write(_points, values, encoding);
           },
           py::arg("points"), py::arg("values"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // py:object / dolfin.function.function.Function
      .def("write",
           [](dolfin::io::XDMFFile& instance, const py::object u,
              dolfin::io::XDMFFile::Encoding encoding) {
             auto _u
                 = u.attr("_cpp_object").cast<dolfin::function::Function*>();
             instance.write(*_u, encoding);
           },
           py::arg("u"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write",
           [](dolfin::io::XDMFFile& instance, const py::object u, double t,
              dolfin::io::XDMFFile::Encoding encoding) {
             auto _u
                 = u.attr("_cpp_object").cast<dolfin::function::Function*>();
             instance.write(*_u, t, encoding);
           },
           py::arg("u"), py::arg("t"),
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      // Check points
      .def("write_checkpoint",
           [](dolfin::io::XDMFFile& instance,
              const dolfin::function::Function& u, std::string function_name,
              double time_step, dolfin::io::XDMFFile::Encoding encoding) {
             instance.write_checkpoint(u, function_name, time_step, encoding);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0,
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5)
      .def("write_checkpoint",
           [](dolfin::io::XDMFFile& instance, const py::object u,
              std::string function_name, double time_step,
              dolfin::io::XDMFFile::Encoding encoding) {
             auto _u
                 = u.attr("_cpp_object").cast<dolfin::function::Function*>();
             instance.write_checkpoint(*_u, function_name, time_step, encoding);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0,
           py::arg("encoding") = dolfin::io::XDMFFile::Encoding::HDF5);

  // XDFMFile::read
  xdmf_file
      // Mesh
      .def("read_mesh",
           [](dolfin::io::XDMFFile& self, const MPICommWrapper comm,
              const dolfin::mesh::GhostMode ghost_mode) {
             return self.read_mesh(comm.get(), ghost_mode);
           })
      // MeshFunction
      .def("read_mf_bool", &dolfin::io::XDMFFile::read_mf_bool, py::arg("mesh"),
           py::arg("name") = "")
      .def("read_mf_int", &dolfin::io::XDMFFile::read_mf_int, py::arg("mesh"),
           py::arg("name") = "")
      .def("read_mf_size_t", &dolfin::io::XDMFFile::read_mf_size_t,
           py::arg("mesh"), py::arg("name") = "")
      .def("read_mf_double", &dolfin::io::XDMFFile::read_mf_double,
           py::arg("mesh"), py::arg("name") = "")
      // MeshValueCollection
      .def("read_mvc_bool", &dolfin::io::XDMFFile::read_mvc_bool,
           py::arg("mesh"), py::arg("name") = "")
      .def("read_mvc_int", &dolfin::io::XDMFFile::read_mvc_int, py::arg("mesh"),
           py::arg("name") = "")
      .def("read_mvc_size_t", &dolfin::io::XDMFFile::read_mvc_size_t,
           py::arg("mesh"), py::arg("name") = "")
      .def("read_mvc_double", &dolfin::io::XDMFFile::read_mvc_double,
           py::arg("mesh"), py::arg("name") = "")
      // Checkpointing
      .def("_read_checkpoint", &dolfin::io::XDMFFile::read_checkpoint,
           py::arg("V"), py::arg("name"), py::arg("counter") = -1);
}
} // namespace dolfin_wrappers
