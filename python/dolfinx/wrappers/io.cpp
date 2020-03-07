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
  m.def("permutation_dolfin_to_vtk", &dolfinx::io::cells::dolfin_to_vtk);
  m.def("permute_cell_ordering", &dolfinx::io::cells::permute_ordering);

  // dolfinx::io::HDF5File
  py::class_<dolfinx::io::HDF5File, std::shared_ptr<dolfinx::io::HDF5File>>(
      m, "HDF5File", py::dynamic_attr())
      .def(py::init([](const MPICommWrapper comm, const std::string filename,
                       const std::string file_mode) {
             return std::make_unique<dolfinx::io::HDF5File>(
                 comm.get(), filename, file_mode);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("file_mode"))
      .def("close", &dolfinx::io::HDF5File::close)
      .def("flush", &dolfinx::io::HDF5File::flush)
      .def("set_mpi_atomicity", &dolfinx::io::HDF5File::set_mpi_atomicity)
      .def("get_mpi_atomicity", &dolfinx::io::HDF5File::get_mpi_atomicity)
      .def_readwrite("chunking", &dolfinx::io::HDF5File::chunking)
      .def("has_dataset", &dolfinx::io::HDF5File::has_dataset);

  // dolfinx::io::XDMFFile
  py::class_<dolfinx::io::XDMFFile,
             std::shared_ptr<dolfinx::io::XDMFFile>>
      xdmf_file_new(m, "XDMFFile");

  xdmf_file_new
      .def(py::init([](const MPICommWrapper comm, std::string filename,
                       dolfinx::io::XDMFFile::Encoding encoding) {
             return std::make_unique<dolfinx::io::XDMFFile>(
                 comm.get(), filename, encoding);
           }),
           py::arg("comm"), py::arg("filename"), py::arg("encoding"))
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write",
           py::overload_cast<const dolfinx::mesh::Mesh&>(
               &dolfinx::io::XDMFFile::write),
           py::arg("mesh"))
      .def("write",
           py::overload_cast<const dolfinx::function::Function&, double>(
               &dolfinx::io::XDMFFile::write),
           py::arg("u"), py::arg("t") = 0.0)
      .def("read_mesh", &dolfinx::io::XDMFFile::read_mesh)
      .def("read_mesh_data", &dolfinx::io::XDMFFile::read_mesh_data)
      .def("read_mf_int", &dolfinx::io::XDMFFile::read_mf_int);

  // dolfinx::io::XDMFFile::Encoding enums
  py::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file_new, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5)
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII);

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
      .def(
          "write",
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
}
} // namespace dolfinx_wrappers
