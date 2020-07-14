// Copyright (C) 2017-2020 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <cfloat>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

template <typename T>
void declare_meshtags(py::module& m, std::string type)
{
  std::string pyclass_name = std::string("MeshTags_") + type;
  py::class_<dolfinx::mesh::MeshTags<T>,
             std::shared_ptr<dolfinx::mesh::MeshTags<T>>>(
      m, pyclass_name.c_str(), "MeshTags object")
      .def(py::init([](const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
                       int dim, const py::array_t<std::int32_t>& indices,
                       const py::array_t<T>& values) {
        std::vector<std::int32_t> indices_vec(indices.data(),
                                              indices.data() + indices.size());
        std::vector<T> values_vec(values.data(), values.data() + values.size());
        return std::make_unique<dolfinx::mesh::MeshTags<T>>(
            mesh, dim, std::move(indices_vec), std::move(values_vec));
      }))
      .def_readwrite("name", &dolfinx::mesh::MeshTags<T>::name)
      .def_property_readonly("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_property_readonly("mesh", &dolfinx::mesh::MeshTags<T>::mesh)
      .def("ufl_id", &dolfinx::mesh::MeshTags<T>::id)
      .def_property_readonly(
          "values",
          [](dolfinx::mesh::MeshTags<T>& self) {
            return py::array_t<T>(self.values().size(), self.values().data(),
                                  py::none());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "indices",
          [](dolfinx::mesh::MeshTags<T>& self) {
            return py::array_t<std::int32_t>(self.indices().size(),
                                             self.indices().data(), py::none());
          },
          py::return_value_policy::reference_internal);

  m.def("create_meshtags",
        [](const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
           const int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           const py::array_t<T>& values) {
          py::buffer_info buf = values.request();
          std::vector<T> vals((T*)buf.ptr, (T*)buf.ptr + buf.size);
          return dolfinx::mesh::create_meshtags(mesh, dim, entities, vals);
        });
}

void mesh(py::module& m)
{

  py::enum_<dolfinx::mesh::CellType>(m, "CellType")
      .value("point", dolfinx::mesh::CellType::point)
      .value("interval", dolfinx::mesh::CellType::interval)
      .value("triangle", dolfinx::mesh::CellType::triangle)
      .value("quadrilateral", dolfinx::mesh::CellType::quadrilateral)
      .value("tetrahedron", dolfinx::mesh::CellType::tetrahedron)
      .value("hexahedron", dolfinx::mesh::CellType::hexahedron);

  m.def("to_string", &dolfinx::mesh::to_string);
  m.def("to_type", &dolfinx::mesh::to_type);
  m.def("is_simplex", &dolfinx::mesh::is_simplex);

  m.def("cell_entity_type", &dolfinx::mesh::cell_entity_type);
  m.def("cell_dim", &dolfinx::mesh::cell_dim);
  m.def("cell_num_entities", &dolfinx::mesh::cell_num_entities);
  m.def("cell_num_vertices", &dolfinx::mesh::num_cell_vertices);
  m.def("get_entity_vertices", &dolfinx::mesh::get_entity_vertices);

  m.def("extract_topology", &dolfinx::mesh::extract_topology);

  m.def("volume_entities", &dolfinx::mesh::volume_entities,
        "Generalised volume of entities of given dimension.");

  m.def("circumradius", &dolfinx::mesh::circumradius);
  m.def("h", &dolfinx::mesh::h,
        "Compute maximum distance between any two vertices.");
  m.def("inradius", &dolfinx::mesh::inradius, "Compute inradius of cells.");
  m.def("radius_ratio", &dolfinx::mesh::radius_ratio);
  m.def("midpoints", &dolfinx::mesh::midpoints);
  m.def("compute_boundary_facets", &dolfinx::mesh::compute_boundary_facets);

  m.def(
      "create_mesh",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         const dolfinx::fem::CoordinateElement& element,
         const Eigen::Ref<const Eigen::Array<
             double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x,
         dolfinx::mesh::GhostMode ghost_mode) {
        return dolfinx::mesh::create_mesh(comm.get(), cells, element, x,
                                          ghost_mode);
      },
      "Helper function for creating meshes.");

  // dolfinx::mesh::GhostMode enums
  py::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfinx::mesh::GhostMode::shared_vertex);

  // dolfinx::mesh::Geometry class
  py::class_<dolfinx::mesh::Geometry, std::shared_ptr<dolfinx::mesh::Geometry>>(
      m, "Geometry", "Geometry object")
      .def(py::init<std::shared_ptr<const dolfinx::common::IndexMap>,
                    const dolfinx::graph::AdjacencyList<std::int32_t>&,
                    const dolfinx::fem::CoordinateElement&,
                    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&,
                    const std::vector<std::int64_t>&>())
      .def_property_readonly("dim", &dolfinx::mesh::Geometry::dim,
                             "Geometric dimension")
      .def_property_readonly("dofmap", &dolfinx::mesh::Geometry::dofmap)
      .def("index_map", &dolfinx::mesh::Geometry::index_map)
      .def_property(
          "x", py::overload_cast<>(&dolfinx::mesh::Geometry::x),
          [](dolfinx::mesh::Geometry& self,
             const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>& values) {
            self.x() = values;
          },
          py::return_value_policy::reference_internal,
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_property_readonly("cmap", &dolfinx::mesh::Geometry::cmap,
                             "The coordinate map")
      .def_property_readonly("input_global_indices",
                             &dolfinx::mesh::Geometry::input_global_indices);

  // dolfinx::mesh::TopologyComputation
  m.def("compute_entities", [](const MPICommWrapper comm,
                               const dolfinx::mesh::Topology& topology,
                               int dim) {
    return dolfinx::mesh::TopologyComputation::compute_entities(comm.get(),
                                                                topology, dim);
  });
  m.def("compute_connectivity",
        &dolfinx::mesh::TopologyComputation::compute_connectivity);

  // dolfinx::mesh::Topology class
  py::class_<dolfinx::mesh::Topology, std::shared_ptr<dolfinx::mesh::Topology>>(
      m, "Topology", "Topology object")
      .def(py::init([](const MPICommWrapper comm,
                       const dolfinx::mesh::CellType cell_type) {
        return std::make_unique<dolfinx::mesh::Topology>(comm.get(), cell_type);
      }))
      .def("set_connectivity", &dolfinx::mesh::Topology::set_connectivity)
      .def("set_index_map", &dolfinx::mesh::Topology::set_index_map)
      .def("create_entities", &dolfinx::mesh::Topology::create_entities)
      .def("create_entity_permutations",
           &dolfinx::mesh::Topology::create_entity_permutations)
      .def("create_connectivity", &dolfinx::mesh::Topology::create_connectivity)
      .def("create_connectivity_all",
           &dolfinx::mesh::Topology::create_connectivity_all)
      .def("get_facet_permutations",
           &dolfinx::mesh::Topology::get_facet_permutations)
      .def("get_cell_permutation_info",
           &dolfinx::mesh::Topology::get_cell_permutation_info)
      .def_property_readonly("dim", &dolfinx::mesh::Topology::dim,
                             "Topological dimension")
      .def("connectivity",
           py::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       py::const_))
      .def("hash", &dolfinx::mesh::Topology::hash)
      .def("index_map", &dolfinx::mesh::Topology::index_map)
      .def_property_readonly("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def("cell_name",
           [](const dolfinx::mesh::Topology& self) {
             return dolfinx::mesh::to_string(self.cell_type());
           })
      .def("mpi_comm", [](dolfinx::mesh::Mesh& self) {
        return MPICommWrapper(self.mpi_comm());
      });

  // dolfinx::mesh::Mesh
  py::class_<dolfinx::mesh::Mesh, std::shared_ptr<dolfinx::mesh::Mesh>>(
      m, "Mesh", py::dynamic_attr(), "Mesh object")
      .def(py::init([](const MPICommWrapper comm,
                       const dolfinx::mesh::Topology& topology,
                       dolfinx::mesh::Geometry& geometry) {
        return std::make_unique<dolfinx::mesh::Mesh>(comm.get(), topology,
                                                     geometry);
      }))
      .def_property_readonly(
          "geometry", py::overload_cast<>(&dolfinx::mesh::Mesh::geometry),
          "Mesh geometry")
      .def("hash", &dolfinx::mesh::Mesh::hash)
      .def("hmax", &dolfinx::mesh::Mesh::hmax)
      .def("hmin", &dolfinx::mesh::Mesh::hmin)
      .def("mpi_comm",
           [](dolfinx::mesh::Mesh& self) {
             return MPICommWrapper(self.mpi_comm());
           })
      .def("rmax", &dolfinx::mesh::Mesh::rmax)
      .def("rmin", &dolfinx::mesh::Mesh::rmin)
      .def_property_readonly(
          "topology", py::overload_cast<>(&dolfinx::mesh::Mesh::topology),
          "Mesh topology", py::return_value_policy::reference_internal)
      .def("ufl_id", &dolfinx::mesh::Mesh::id)
      .def_property_readonly("id", &dolfinx::mesh::Mesh::id)
      .def_readwrite("name", &dolfinx::mesh::Mesh::name);

  // dolfinx::mesh::MeshTags

  declare_meshtags<std::int8_t>(m, "int8");
  declare_meshtags<std::int32_t>(m, "int32");
  declare_meshtags<double>(m, "double");
  declare_meshtags<std::int64_t>(m, "int64");

  // Partitioning interface
  m.def("partition_cells",
        [](const MPICommWrapper comm, int nparts,
           dolfinx::mesh::CellType cell_type,
           const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
           dolfinx::mesh::GhostMode ghost_mode) {
          return dolfinx::mesh::Partitioning::partition_cells(
              comm.get(), nparts, cell_type, cells, ghost_mode);
        });

  m.def("locate_entities", &dolfinx::mesh::locate_entities);
  m.def("locate_entities_boundary", &dolfinx::mesh::locate_entities_boundary);

} // namespace dolfinx_wrappers
} // namespace dolfinx_wrappers
