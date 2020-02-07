// Copyright (C) 2017-2019 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/io/HDF5File.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/MeshValueCollection.h>
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
  m.def("permutation_vtk_to_dolfin", &dolfinx::io::cells::vtk_to_dolfin);
  m.def("permute_cell_ordering", &dolfinx::io::cells::permute_ordering);

  // dolfinx::io::HDF5File
  py::class_<dolfinx::io::HDF5File, std::shared_ptr<dolfinx::io::HDF5File>>(
      m, "HDF5File", py::dynamic_attr())
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode) {
             return std::make_unique<dolfinx::io::HDF5File>(comm.get(), filename,
                                                           file_mode);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"))
      .def("close", &dolfinx::io::HDF5File::close)
      .def("flush", &dolfinx::io::HDF5File::flush)
      // read
      .def("read_mesh",
           [](dolfinx::io::HDF5File& self, const std::string data_path,
              bool use_partition_from_file,
              const dolfinx::mesh::GhostMode ghost_mode) {
             return self.read_mesh(data_path, use_partition_from_file,
                                   ghost_mode);
           })
      .def("read_vector",
           [](dolfinx::io::HDF5File& self, const MPICommWrapper comm,
              const std::string data_path, bool use_partition_from_file) {
             auto x = self.read_vector(comm.get(), data_path,
                                       use_partition_from_file);
             Vec _x = x.vec();
             PetscObjectReference((PetscObject)_x);
             return _x;
           },
           py::return_value_policy::take_ownership)
      .def("read_mf_int", &dolfinx::io::HDF5File::read_mf_int, py::arg("mesh"),
           py::arg("name"))
      .def("read_mf_size_t", &dolfinx::io::HDF5File::read_mf_size_t,
           py::arg("mesh"), py::arg("name"))
      .def("read_mf_double", &dolfinx::io::HDF5File::read_mf_double,
           py::arg("mesh"), py::arg("name"))
      //
      .def("read_mvc_bool", &dolfinx::io::HDF5File::read_mvc_bool,
           py::arg("mesh"), py::arg("name"))
      .def("read_mvc_size_t", &dolfinx::io::HDF5File::read_mvc_size_t,
           py::arg("mesh"), py::arg("name"))
      .def("read_mvc_double", &dolfinx::io::HDF5File::read_mvc_double,
           py::arg("mesh"), py::arg("name"))
      .def("read",
           py::overload_cast<
               std::shared_ptr<const dolfinx::function::FunctionSpace>,
               const std::string>(&dolfinx::io::HDF5File::read, py::const_),
           py::arg("V"), py::arg("name"))
      // write
      .def("write", (void (dolfinx::io::HDF5File::*)(const dolfinx::mesh::Mesh&,
                                                    std::string))
                        & dolfinx::io::HDF5File::write)
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshValueCollection<bool>&, std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshValueCollection<std::size_t>&,
               std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshValueCollection<double>&, std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("mvc"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshFunction<std::size_t>&, std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshFunction<int>&, std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(
               const dolfinx::mesh::MeshFunction<double>&, std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("meshfunction"), py::arg("name"))
      .def("write",
           [](dolfinx::io::HDF5File& self, Vec x, std::string s) {
             dolfinx::la::PETScVector _x(x, true);
             self.write(_x, s);
           },
           py::arg("vector"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(const dolfinx::function::Function&,
                                           std::string))
               & dolfinx::io::HDF5File::write,
           py::arg("u"), py::arg("name"))
      .def("write",
           (void (dolfinx::io::HDF5File::*)(const dolfinx::function::Function&,
                                           std::string, double))
               & dolfinx::io::HDF5File::write,
           py::arg("u"), py::arg("name"), py::arg("t"))
      .def("set_mpi_atomicity", &dolfinx::io::HDF5File::set_mpi_atomicity)
      .def("get_mpi_atomicity", &dolfinx::io::HDF5File::get_mpi_atomicity)
      .def_readwrite("chunking", &dolfinx::io::HDF5File::chunking)
      // others
      .def("has_dataset", &dolfinx::io::HDF5File::has_dataset);

  // dolfinx::io::XDMFFile
  py::class_<dolfinx::io::XDMFFile, std::shared_ptr<dolfinx::io::XDMFFile>>
      xdmf_file(m, "XDMFFile");

  xdmf_file
      .def(py::init([](const MPICommWrapper comm, std::string filename,
                       dolfinx::io::XDMFFile::Encoding encoding) {
             return std::make_unique<dolfinx::io::XDMFFile>(comm.get(), filename,
                                                           encoding);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("encoding"))
      .def("close", &dolfinx::io::XDMFFile::close)
      .def_readwrite("functions_share_mesh",
                     &dolfinx::io::XDMFFile::functions_share_mesh)
      .def_readwrite("flush_output", &dolfinx::io::XDMFFile::flush_output)
      .def_readwrite("rewrite_function_mesh",
                     &dolfinx::io::XDMFFile::rewrite_function_mesh);

  // dolfinx::io::XDMFFile::Encoding enums
  py::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

  // dolfinx::io::XDMFFile::write
  xdmf_file
      // Function
      .def("write",
           py::overload_cast<const dolfinx::function::Function&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("u"))
      .def("write",
           py::overload_cast<const dolfinx::function::Function&, double>(
               &dolfinx::io::XDMFFile::write),
           py::arg("u"), py::arg("t"))
      // Mesh
      .def("write",
           py::overload_cast<const dolfinx::mesh::Mesh&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mesh"))
      // MeshFunction
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<std::size_t>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<int>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<double>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mf"))
      // MeshValueCollection
      .def("write",
           py::overload_cast<
               const dolfinx::mesh::MeshValueCollection<std::size_t>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mvc"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshValueCollection<int>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mvc"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshValueCollection<double>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mvc"))
      // Points
      .def(
          "write",
          py::overload_cast<const Eigen::Ref<
              const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>&>(
              &dolfinx::io::XDMFFile::write),
          py::arg("points"))
      .def("write",
           py::overload_cast<const Eigen::Ref<const Eigen::Array<
                                 double, Eigen::Dynamic, 3, Eigen::RowMajor>>&,
                             const std::vector<double>&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("points"), py::arg("values"))
      // Checkpoints
      .def("write_checkpoint",
           [](dolfinx::io::XDMFFile& instance,
              const dolfinx::function::Function& u, std::string function_name,
              double time_step) {
             instance.write_checkpoint(u, function_name, time_step);
           },
           py::arg("u"), py::arg("function_name"), py::arg("time_step") = 0.0);

  // dolfinx::io::VTKFile
  py::class_<dolfinx::io::VTKFile, std::shared_ptr<dolfinx::io::VTKFile>>
      vtk_file(m, "VTKFile");

  vtk_file
      .def(py::init([](std::string filename) {
             return std::make_unique<dolfinx::io::VTKFile>(filename);
           }),
           py::arg("filename"))
      .def("write",
           py::overload_cast<const dolfinx::function::Function&>(
               &dolfinx::io::VTKFile::write),
           py::arg("u"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::Mesh&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mesh"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<std::size_t>&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<std::size_t>&,
                             double>(&dolfinx::io::VTKFile::write),
           py::arg("mf"), py::arg("t"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<bool>&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<bool>&, double>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"), py::arg("t"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<double>&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<double>&, double>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"), py::arg("t"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<int>&>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"))
      .def("write",
           py::overload_cast<const dolfinx::mesh::MeshFunction<int>&, double>(
               &dolfinx::io::VTKFile::write),
           py::arg("mf"), py::arg("t"));

  // XDFMFile::read
  xdmf_file
      // Mesh
      .def("read_mesh",
           [](dolfinx::io::XDMFFile& self,
              const dolfinx::mesh::GhostMode ghost_mode) {
             return self.read_mesh(ghost_mode);
           })
      .def("read_mesh_data",
           [](dolfinx::io::XDMFFile& self, const MPICommWrapper comm) {
             return self.read_mesh_data(comm.get());
           })
      // MeshFunction
      .def("read_mf_int", &dolfinx::io::XDMFFile::read_mf_int, py::arg("mesh"),
           py::arg("name") = "")
      .def("read_mf_size_t", &dolfinx::io::XDMFFile::read_mf_size_t,
           py::arg("mesh"), py::arg("name") = "")
      .def("read_mf_double", &dolfinx::io::XDMFFile::read_mf_double,
           py::arg("mesh"), py::arg("name") = "")
      // MeshValueCollection
      .def("read_mvc_int", &dolfinx::io::XDMFFile::read_mvc_int, py::arg("mesh"),
           py::arg("name") = "")
      .def("read_mvc_size_t", &dolfinx::io::XDMFFile::read_mvc_size_t,
           py::arg("mesh"), py::arg("name") = "")
      .def("read_mvc_double", &dolfinx::io::XDMFFile::read_mvc_double,
           py::arg("mesh"), py::arg("name") = "")
      // Checkpointing
      .def("read_checkpoint", &dolfinx::io::XDMFFile::read_checkpoint,
           py::arg("V"), py::arg("name"), py::arg("counter") = -1);
}
} // namespace dolfinx_wrappers
