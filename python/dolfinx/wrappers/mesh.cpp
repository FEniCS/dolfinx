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
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshQuality.h>
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
      "create",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         const dolfinx::fem::CoordinateElement& element,
         const Eigen::Ref<const Eigen::Array<
             double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x,
         dolfinx::mesh::GhostMode ghost_mode) {
        return dolfinx::mesh::create(comm.get(), cells, element, x, ghost_mode);
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
      .def(py::init(
          [](const MPICommWrapper comm, dolfinx::mesh::CellType type,
             const Eigen::Ref<const Eigen::Array<
                 double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&
                 geometry,
             const Eigen::Ref<
                 const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>& topology,
             const dolfinx::fem::CoordinateElement& element,
             const std::vector<std::int64_t>& global_cell_indices,
             const dolfinx::mesh::GhostMode ghost_mode) {
            return std::make_unique<dolfinx::mesh::Mesh>(
                comm.get(), type, geometry, topology, element,
                global_cell_indices, ghost_mode);
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

  // dolfinx::mesh::MeshEntity class
  py::class_<dolfinx::mesh::MeshEntity,
             std::shared_ptr<dolfinx::mesh::MeshEntity>>(m, "MeshEntity",
                                                         "MeshEntity object")
      .def(py::init<const dolfinx::mesh::Mesh&, std::size_t, std::size_t>())
      .def_property_readonly("dim", &dolfinx::mesh::MeshEntity::dim,
                             "Topological dimension")
      .def("mesh", &dolfinx::mesh::MeshEntity::mesh, "Associated mesh")
      .def("index",
           py::overload_cast<>(&dolfinx::mesh::MeshEntity::index, py::const_),
           "Entity index")
      .def("entities", &dolfinx::mesh::MeshEntity::entities,
           py::return_value_policy::reference_internal);

// dolfinx::mesh::MeshTags
#define MESHTAGS_MACRO(SCALAR, SCALAR_NAME)                                    \
  py::class_<dolfinx::mesh::MeshTags<SCALAR>,                                  \
             std::shared_ptr<dolfinx::mesh::MeshTags<SCALAR>>>(                \
      m, "MeshTags_" #SCALAR_NAME, "MeshTags object")                          \
      .def(py::init([](const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh, \
                       int dim, const py::array_t<std::int32_t>& indices,      \
                       const py::array_t<SCALAR>& values) {                    \
        std::vector<std::int32_t> indices_vec(                                 \
            indices.data(), indices.data() + indices.size());                  \
        std::vector<SCALAR> values_vec(values.data(),                          \
                                       values.data() + values.size());         \
        return std::make_unique<dolfinx::mesh::MeshTags<SCALAR>>(              \
            mesh, dim, std::move(indices_vec), std::move(values_vec));         \
      }))                                                                      \
      .def_readwrite("name", &dolfinx::mesh::MeshTags<SCALAR>::name)           \
      .def_property_readonly("dim", &dolfinx::mesh::MeshTags<SCALAR>::dim)     \
      .def_property_readonly("mesh", &dolfinx::mesh::MeshTags<SCALAR>::mesh)   \
      .def("ufl_id", &dolfinx::mesh::MeshTags<SCALAR>::id)                     \
      .def_property_readonly("values",                                         \
                             [](dolfinx::mesh::MeshTags<SCALAR>& self) {       \
                               return py::array_t<SCALAR>(                     \
                                   self.values().size(), self.values().data(), \
                                   py::none());                                \
                             })                                                \
      .def_property_readonly(                                                  \
          "indices", [](dolfinx::mesh::MeshTags<SCALAR>& self) {               \
            return py::array_t<std::int32_t>(                                  \
                self.indices().size(), self.indices().data(), py::none());     \
          });

  MESHTAGS_MACRO(std::int8_t, int8);
  MESHTAGS_MACRO(int, int);
  MESHTAGS_MACRO(double, double);
  MESHTAGS_MACRO(std::int64_t, int64);
#undef MESHTAGS_MACRO

  // dolfinx::mesh::MeshQuality
  py::class_<dolfinx::mesh::MeshQuality>(m, "MeshQuality", "MeshQuality class")
      .def_static("dihedral_angle_histogram_data",
                  &dolfinx::mesh::MeshQuality::dihedral_angle_histogram_data,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("dihedral_angles_min_max",
                  &dolfinx::mesh::MeshQuality::dihedral_angles_min_max);

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

  // TODO Remove
  m.def("compute_vertex_exterior_markers",
        &dolfinx::mesh::Partitioning::compute_vertex_exterior_markers);

} // namespace dolfinx_wrappers
} // namespace dolfinx_wrappers
