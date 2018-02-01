// Copyright (C) 2017 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
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
  // dolfin::VTKFile
  py::class_<dolfin::VTKFile, std::shared_ptr<dolfin::VTKFile>>(m, "VTKFile")
      .def(py::init<std::string, std::string>())
      .def("write", [](dolfin::VTKFile& instance, const dolfin::Mesh& mesh) {
        instance.write(mesh);
      });

#ifdef HAS_HDF5
  // dolfin::HDF5File
  py::class_<dolfin::HDF5File, std::shared_ptr<dolfin::HDF5File>,
             dolfin::Variable>(m, "HDF5File")
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode) {
             return std::unique_ptr<dolfin::HDF5File>(
                 new dolfin::HDF5File(comm.get(), filename, file_mode));
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"))
      .def("__enter__", [](dolfin::HDF5File& self) { return &self; })
      .def("__exit__", [](dolfin::HDF5File& self, py::args args,
                          py::kwargs kwargs) { self.close(); })
      .def("close", &dolfin::HDF5File::close)
      .def("flush", &dolfin::HDF5File::flush)
      // read
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::Mesh&, std::string, bool) const)
               & dolfin::HDF5File::read)
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshValueCollection<bool>&,
                                       std::string) const)
               & dolfin::HDF5File::read,
           py::arg("mvc"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(
               dolfin::MeshValueCollection<std::size_t>&, std::string) const)
               & dolfin::HDF5File::read,
           py::arg("mvc"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshValueCollection<double>&,
                                       std::string) const)
               & dolfin::HDF5File::read,
           py::arg("mvc"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshFunction<bool>&, std::string)
                const)
               & dolfin::HDF5File::read,
           py::arg("meshfunction"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshFunction<std::size_t>&,
                                       std::string) const)
               & dolfin::HDF5File::read,
           py::arg("meshfunction"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshFunction<int>&, std::string)
                const)
               & dolfin::HDF5File::read,
           py::arg("meshfunction"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::MeshFunction<double>&,
                                       std::string) const)
               & dolfin::HDF5File::read,
           py::arg("meshfunction"), py::arg("name"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::PETScVector&, std::string, bool)
                const)
               & dolfin::HDF5File::read,
           py::arg("vector"), py::arg("name"), py::arg("use_partitioning"))
      .def("read",
           (void (dolfin::HDF5File::*)(dolfin::Function&, const std::string))
               & dolfin::HDF5File::read,
           py::arg("u"), py::arg("name"))
      .def("read",
           [](dolfin::HDF5File& self, py::object u, std::string name) {
             try
             {
               auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
               self.read(*_u, name);
             }
             catch (const std::exception& e)
             {
               // Do nothing, pybind11 will try next function
             }
           },
           py::arg("u"), py::arg("name"))
      // write
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::Mesh&, std::string))
               & dolfin::HDF5File::write)
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::MeshValueCollection<bool>&,
                                       std::string))
               & dolfin::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(
               const dolfin::MeshValueCollection<std::size_t>&, std::string))
               & dolfin::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(
               const dolfin::MeshValueCollection<double>&, std::string))
               & dolfin::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::MeshFunction<bool>&,
                                       std::string))
               & dolfin::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::MeshFunction<std::size_t>&,
                                       std::string))
               & dolfin::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::MeshFunction<int>&,
                                       std::string))
               & dolfin::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::MeshFunction<double>&,
                                       std::string))
               & dolfin::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::PETScVector&, std::string))
               & dolfin::HDF5File::write,
           py::arg("vector"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::Function&, std::string))
               & dolfin::HDF5File::write,
           py::arg("u"), py::arg("name"))
      .def("write",
           (void (dolfin::HDF5File::*)(const dolfin::Function&, std::string,
                                       double))
               & dolfin::HDF5File::write,
           py::arg("u"), py::arg("name"), py::arg("t"))
      .def("write",
           [](dolfin::HDF5File& self, py::object u, std::string name) {
             try
             {
               auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
               self.write(*_u, name);
             }
             catch (const std::exception& e)
             {
               // Do nothing, pybind11 will try next function
             }
           },
           py::arg("u"), py::arg("name"))
      .def(
          "write",
          [](dolfin::HDF5File& self, py::object u, std::string name, double t) {
            try
            {
              auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
              self.write(*_u, name, t);
            }
            catch (const std::exception& e)
            {
              // Do nothing, pybind11 will try next function
            }
          },
          py::arg("u"), py::arg("name"), py::arg("t"))
      .def("set_mpi_atomicity", &dolfin::HDF5File::set_mpi_atomicity)
      .def("get_mpi_atomicity", &dolfin::HDF5File::get_mpi_atomicity)
      // others
    .def("has_dataset", &dolfin::HDF5File::has_dataset);

#endif

  // dolfin::XDMFFile
  py::class_<dolfin::XDMFFile, std::shared_ptr<dolfin::XDMFFile>,
             dolfin::Variable>
      xdmf_file(m, "XDMFFile");

  xdmf_file
      .def(py::init([](const MPICommWrapper comm, std::string filename) {
             return std::unique_ptr<dolfin::XDMFFile>(
                 new dolfin::XDMFFile(comm.get(), filename));
           }),
           py::arg("comm"), py::arg("filename"))
      .def(py::init<std::string>())
      .def("__enter__", [](dolfin::XDMFFile& self) { return &self; })
      .def("__exit__", [](dolfin::XDMFFile& self, py::args args,
                          py::kwargs kwargs) { self.close(); });

  // dolfin::XDMFFile::Encoding enums
  py::enum_<dolfin::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfin::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfin::XDMFFile::Encoding::ASCII);

  // dolfin::XDMFFile::write
  xdmf_file
      // Function
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::Function&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("u"), py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::Function&, double,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("u"), py::arg("t"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // Mesh
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::Mesh&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mesh"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // MeshFunction
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshFunction<bool>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshFunction<std::size_t>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshFunction<int>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshFunction<double>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // MeshValueCollection
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshValueCollection<bool>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(
               const dolfin::MeshValueCollection<std::size_t>&,
               dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const dolfin::MeshValueCollection<int>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(
               const dolfin::MeshValueCollection<double>&,
               dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("mvc"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // py:object / dolfin.function.Function
      .def("write",
           [](dolfin::XDMFFile& instance, const py::object u,
              dolfin::XDMFFile::Encoding encoding) {
             try
             {
               auto _u = u.attr("_cpp_object").cast<dolfin::Function&>();
               instance.write(_u, encoding);
             }
             catch (const std::exception& e)
             {
               // Do nothing, pybind11 will try next function
             }
           },
           py::arg("u"), py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           [](dolfin::XDMFFile& instance, const py::object u, double t,
              dolfin::XDMFFile::Encoding encoding) {
             try
             {
               auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
               instance.write(*_u, t, encoding);
             }
             catch (const std::exception& e)
             {
               // Do nothing, pybind11 will try next function
             }
           },
           py::arg("u"), py::arg("t"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // Points
      .def("write",
           (void (dolfin::XDMFFile::*)(const std::vector<dolfin::Point>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("points"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write",
           (void (dolfin::XDMFFile::*)(const std::vector<dolfin::Point>&,
                                       const std::vector<double>&,
                                       dolfin::XDMFFile::Encoding))
               & dolfin::XDMFFile::write,
           py::arg("points"), py::arg("values"),
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      // Check points
      .def("write_checkpoint",
           [](dolfin::XDMFFile& instance, const dolfin::Function& u,
              std::string function_name, double time_step,
              dolfin::XDMFFile::Encoding encoding) {
             instance.write_checkpoint(u, function_name, time_step, encoding);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0,
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5)
      .def("write_checkpoint",
           [](dolfin::XDMFFile& instance, const py::object u,
              std::string function_name, double time_step,
              dolfin::XDMFFile::Encoding encoding) {
             auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
             instance.write_checkpoint(*_u, function_name, time_step, encoding);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0,
           py::arg("encoding") = dolfin::XDMFFile::Encoding::HDF5);

  // XDFMFile::read
  xdmf_file
      // Mesh
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::Mesh&) const)
               & dolfin::XDMFFile::read)
      // MeshFunction
      .def(
          "read",
          (void (dolfin::XDMFFile::*)(dolfin::MeshFunction<bool>&, std::string))
              & dolfin::XDMFFile::read,
          py::arg("mf"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshFunction<std::size_t>&,
                                       std::string))
               & dolfin::XDMFFile::read,
           py::arg("mf"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshFunction<int>&, std::string))
               & dolfin::XDMFFile::read,
           py::arg("mf"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshFunction<double>&,
                                       std::string))
               & dolfin::XDMFFile::read,
           py::arg("mf"), py::arg("name") = "")
      // MeshValueCollection
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshValueCollection<bool>&,
                                       std::string))
               & dolfin::XDMFFile::read,
           py::arg("mvc"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(
               dolfin::MeshValueCollection<std::size_t>&, std::string))
               & dolfin::XDMFFile::read,
           py::arg("mvc"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshValueCollection<int>&,
                                       std::string))
               & dolfin::XDMFFile::read,
           py::arg("mvc"), py::arg("name") = "")
      .def("read",
           (void (dolfin::XDMFFile::*)(dolfin::MeshValueCollection<double>&,
                                       std::string))
               & dolfin::XDMFFile::read,
           py::arg("mvc"), py::arg("name") = "")
      //
      .def("read_checkpoint", &dolfin::XDMFFile::read_checkpoint, py::arg("u"),
           py::arg("name"), py::arg("counter") = -1)
      .def("read_checkpoint",
           [](dolfin::XDMFFile& instance, py::object u, std::string name,
              std::int64_t counter) {
             auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
             instance.read_checkpoint(*_u, name, counter);
           },
           py::arg("u"), py::arg("name"), py::arg("counter") = -1);
}
}
