// Copyright (C) 2017-2021 Chris N. Richardson Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/io.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <basix/mdspan.hpp>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/VTKHDF.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/utils.h>
#include <dolfinx/io/vtk_utils.h>
#include <dolfinx/io/xdmf_utils.h>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace dolfinx_wrappers
{

void io(nb::module_& m)
{

  declare_data_types<std::int32_t>(m);
  declare_data_types<std::int64_t>(m);
  declare_data_types<float>(m);
  declare_data_types<std::complex<float>>(m);
  declare_data_types<double>(m);
  declare_data_types<std::complex<double>>(m);

  // dolfinx::io::cell vtk cell type converter
  m.def("get_vtk_cell_type", &dolfinx::io::cells::get_vtk_cell_type,
        nb::arg("cell"), nb::arg("dim"), "Get VTK cell identifier");

  m.def(
      "extract_vtk_connectivity",
      [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
         dolfinx::mesh::CellType cell)
      {
        dolfinx_wrappers::mdspan_t<const std::int32_t, 2> _dofmap(
            dofmap.data(), dofmap.shape(0), dofmap.shape(1));
        auto [cells, shape]
            = dolfinx::io::extract_vtk_connectivity(_dofmap, cell);
        return as_nbarray(std::move(cells), shape);
      },
      nb::arg("dofmap"), nb::arg("celltype"),
      "Extract the mesh topology with VTK ordering using "
      "geometry indices");

  m.def("write_vtkhdf_mesh", &dolfinx::io::VTKHDF::write_mesh<double>)
      .def("write_vtkhdf_mesh", &dolfinx::io::VTKHDF::write_mesh<float>);
  m.def("write_vtkhdf_data", &dolfinx::io::VTKHDF::write_data<double>);
  m.def("write_vtkhdf_data", &dolfinx::io::VTKHDF::write_data<float>);
  m.def("write_vtkhdf_function", &dolfinx::io::VTKHDF::write_function<float>);
  m.def("write_vtkhdf_function", &dolfinx::io::VTKHDF::write_function<double>);
  m.def("read_vtkhdf_cg1_function", &dolfinx::io::VTKHDF::read_CG1_function<float>);
  m.def("read_vtkhdf_cg1_function", &dolfinx::io::VTKHDF::read_CG1_function<double>);
  m.def("read_vtkhdf_mesh_float64",
        [](MPICommWrapper comm, const std::string& filename, std::size_t gdim,
           std::optional<std::int32_t> max_facet_to_cell_links)
        {
          return dolfinx::io::VTKHDF::read_mesh<double>(
              comm.get(), filename, gdim, max_facet_to_cell_links);
        });
  m.def("read_vtkhdf_mesh_float32",
        [](MPICommWrapper comm, const std::string& filename, std::size_t gdim,
           std::optional<std::int32_t> max_facet_to_cell_links)
        {
          return dolfinx::io::VTKHDF::read_mesh<float>(
              comm.get(), filename, gdim, max_facet_to_cell_links);
        });

  // dolfinx::io::cell permutation functions
  m.def("perm_vtk", &dolfinx::io::cells::perm_vtk, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from VTK to DOLFINx node ordering");
  m.def("perm_gmsh", &dolfinx::io::cells::perm_gmsh, nb::arg("type"),
        nb::arg("num_nodes"),
        "Permutation array to map from Gmsh to DOLFINx node ordering");

  // dolfinx::io::XDMFFile
  nb::class_<dolfinx::io::XDMFFile> xdmf_file(m, "XDMFFile");

  // dolfinx::io::XDMFFile::Encoding enums
  nb::enum_<dolfinx::io::XDMFFile::Encoding>(xdmf_file, "Encoding")
      .value("HDF5", dolfinx::io::XDMFFile::Encoding::HDF5, "HDF5 encoding")
      .value("ASCII", dolfinx::io::XDMFFile::Encoding::ASCII,
             "Plain text encoding");

  xdmf_file
      .def(
          "__init__",
          [](dolfinx::io::XDMFFile* x, MPICommWrapper comm,
             std::filesystem::path filename, const std::string& file_mode,
             dolfinx::io::XDMFFile::Encoding encoding)
          {
            new (x) dolfinx::io::XDMFFile(comm.get(), filename, file_mode,
                                          encoding);
          },
          nb::arg("comm"), nb::arg("filename"), nb::arg("file_mode"),
          nb::arg("encoding") = dolfinx::io::XDMFFile::Encoding::HDF5)
      .def("close", &dolfinx::io::XDMFFile::close)
      .def("write_geometry", &dolfinx::io::XDMFFile::write_geometry,
           nb::arg("geometry"), nb::arg("name") = "geometry",
           nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_topology_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath)
          {
            auto [cells, shape] = self.read_topology_data(name, xpath);
            return as_nbarray(std::move(cells), shape);
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def(
          "read_geometry_data",
          [](dolfinx::io::XDMFFile& self, const std::string& name,
             const std::string& xpath)
          {
            auto [x, shape] = self.read_geometry_data(name, xpath);
            std::vector<double>& _x = std::get<std::vector<double>>(x);
            return as_nbarray(std::move(_x), shape);
          },
          nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_geometry_data", &dolfinx::io::XDMFFile::read_geometry_data,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_cell_type", &dolfinx::io::XDMFFile::read_cell_type,
           nb::arg("name") = "mesh", nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_meshtags", &dolfinx::io::XDMFFile::read_meshtags,
           nb::arg("mesh"), nb::arg("name"), nb::arg("attribute_name").none(),
           nb::arg("xpath"))
      .def("write_information", &dolfinx::io::XDMFFile::write_information,
           nb::arg("name"), nb::arg("value"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("read_information", &dolfinx::io::XDMFFile::read_information,
           nb::arg("name"), nb::arg("xpath") = "/Xdmf/Domain")
      .def("flush", &dolfinx::io::XDMFFile::flush)
      .def_prop_ro(
          "comm", [](dolfinx::io::XDMFFile& self)
          { return MPICommWrapper(self.comm()); }, nb::keep_alive<0, 1>());

  declare_xdmf_real_fn<float>(xdmf_file);
  declare_xdmf_real_fn<double>(xdmf_file);
  declare_xdmf_scalar_fn<float, float>(xdmf_file);
  declare_xdmf_scalar_fn<double, double>(xdmf_file);
  declare_xdmf_scalar_fn<std::complex<float>, float>(xdmf_file);
  declare_xdmf_scalar_fn<std::complex<double>, double>(xdmf_file);

  // dolfinx::io::VTKFile
  nb::class_<dolfinx::io::VTKFile> vtk_file(m, "VTKFile");
  vtk_file
      .def(
          "__init__",
          [](dolfinx::io::VTKFile* v, MPICommWrapper comm,
             std::filesystem::path filename, const std::string& mode)
          { new (v) dolfinx::io::VTKFile(comm.get(), filename, mode); },
          nb::arg("comm"), nb::arg("filename"), nb::arg("mode"))
      .def("close", &dolfinx::io::VTKFile::close);

  declare_vtk_real_fn<float>(vtk_file);
  declare_vtk_real_fn<double>(vtk_file);
  declare_vtk_scalar_fn<float, float>(vtk_file);
  declare_vtk_scalar_fn<double, double>(vtk_file);
  declare_vtk_scalar_fn<std::complex<float>, float>(vtk_file);
  declare_vtk_scalar_fn<std::complex<double>, double>(vtk_file);

#ifdef HAS_ADIOS2
  nb::enum_<dolfinx::io::VTXMeshPolicy>(m, "VTXMeshPolicy")
      .value("update", dolfinx::io::VTXMeshPolicy::update)
      .value("reuse", dolfinx::io::VTXMeshPolicy::reuse);

  declare_vtx_writer<float>(m, "float32");
  declare_vtx_writer<double>(m, "float64");
#endif
}
} // namespace dolfinx_wrappers
