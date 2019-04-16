// Copyright (C) 2017-2019 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/io/HDF5File.h>
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

namespace py = pybind11;

namespace dolfin_wrappers
{

void io(py::module& m)
{
  // dolfin::io::HDF5File
  py::class_<dolfin::io::HDF5File, std::shared_ptr<dolfin::io::HDF5File>>(
      m, "HDF5File", py::dynamic_attr())
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode) {
             return std::make_unique<dolfin::io::HDF5File>(comm.get(), filename,
                                                           file_mode);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"))
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
             auto x = self.read_vector(comm.get(), data_path,
                                       use_partition_from_file);
             Vec _x = x.vec();
             PetscObjectReference((PetscObject)_x);
             return _x;
           },
           py::return_value_policy::take_ownership)
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
           [](dolfin::io::HDF5File& self, Vec x, std::string s) {
             dolfin::la::PETScVector _x(x);
             self.write(_x, s);
           },
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
      .def("set_mpi_atomicity", &dolfin::io::HDF5File::set_mpi_atomicity)
      .def("get_mpi_atomicity", &dolfin::io::HDF5File::get_mpi_atomicity)
      .def_readwrite("chunking", &dolfin::io::HDF5File::chunking)
      // others
      .def("has_dataset", &dolfin::io::HDF5File::has_dataset);

  // dolfin::io::XDMFFile
  py::class_<dolfin::io::XDMFFile, std::shared_ptr<dolfin::io::XDMFFile>>
      xdmf_file(m, "XDMFFile");

  xdmf_file
      .def(py::init([](const MPICommWrapper comm, std::string filename,
                       dolfin::io::XDMFFile::Encoding encoding) {
             return std::make_unique<dolfin::io::XDMFFile>(comm.get(), filename,
                                                           encoding);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("encoding"))
      .def("close", &dolfin::io::XDMFFile::close)
      .def_readwrite("functions_share_mesh",
                     &dolfin::io::XDMFFile::functions_share_mesh)
      .def_readwrite("flush_output", &dolfin::io::XDMFFile::flush_output)
      .def_readwrite("rewrite_function_mesh",
                     &dolfin::io::XDMFFile::rewrite_function_mesh);

  // dolfin::io::XDMFFile::Encoding enums
  py::enum_<dolfin::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfin::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfin::io::XDMFFile::Encoding::ASCII);

  // dolfin::io::XDMFFile::write
  xdmf_file
      // Function
      .def("write",
           py::overload_cast<const dolfin::function::Function&>(
               &dolfin::io::XDMFFile::write),
           py::arg("u"))
      .def("write",
           py::overload_cast<const dolfin::function::Function&, double>(
               &dolfin::io::XDMFFile::write),
           py::arg("u"), py::arg("t"))
      // Mesh
      .def("write",
           py::overload_cast<const dolfin::mesh::Mesh&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mesh"))
      // MeshFunction
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshFunction<bool>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshFunction<std::size_t>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshFunction<int>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshFunction<double>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mf"))
      // MeshValueCollection
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshValueCollection<bool>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mvc"))
      .def("write",
           py::overload_cast<
               const dolfin::mesh::MeshValueCollection<std::size_t>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mvc"))
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshValueCollection<int>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mvc"))
      .def("write",
           py::overload_cast<const dolfin::mesh::MeshValueCollection<double>&>(
               &dolfin::io::XDMFFile::write),
           py::arg("mvc"))
      // Points
      .def("write",
           [](dolfin::io::XDMFFile& instance, py::list points) {
             auto _points = points.cast<std::vector<dolfin::geometry::Point>>();
             instance.write(_points);
           },
           py::arg("points"))
      // Points with values
      .def("write",
           [](dolfin::io::XDMFFile& instance, py::list points,
              std::vector<double>& values) {
             auto _points = points.cast<std::vector<dolfin::geometry::Point>>();
             instance.write(_points, values);
           },
           py::arg("points"), py::arg("values"))
      // Check points
      .def("write_checkpoint",
           [](dolfin::io::XDMFFile& instance,
              const dolfin::function::Function& u, std::string function_name,
              double time_step) {
             instance.write_checkpoint(u, function_name, time_step);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0);

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
      .def("read_checkpoint", &dolfin::io::XDMFFile::read_checkpoint,
           py::arg("V"), py::arg("name"), py::arg("counter") = -1);
}
} // namespace dolfin_wrappers
