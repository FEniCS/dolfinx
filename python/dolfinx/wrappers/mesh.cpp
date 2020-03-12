// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
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
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/MeshQuality.h>
#include <dolfinx/mesh/MeshValueCollection.h>
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

  m.def("compute_interior_facets", &dolfinx::mesh::compute_interior_facets);

  m.def("volume_entities", &dolfinx::mesh::volume_entities,
        "Generalised volume of entities of given dimension.");

  m.def("circumradius", &dolfinx::mesh::circumradius);
  m.def("h", &dolfinx::mesh::h,
        "Compute maximum distance between any two vertices.");
  m.def("inradius", &dolfinx::mesh::inradius, "Compute inradius of cells.");
  m.def("radius_ratio", &dolfinx::mesh::radius_ratio);
  m.def("midpoints", &dolfinx::mesh::midpoints);

  m.def("create", &dolfinx::mesh::create,
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
                    const dolfinx::fem::ElementDofLayout&,
                    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&,
                    const std::vector<std::int64_t>&>())
      .def_property_readonly("dim", &dolfinx::mesh::Geometry::dim,
                             "Geometric dimension")
      .def("dofmap",
           py::overload_cast<>(&dolfinx::mesh::Geometry::dofmap, py::const_))
      .def("dof_layout", &dolfinx::mesh::Geometry::dof_layout)
      .def("index_map", &dolfinx::mesh::Geometry::index_map)
      .def("global_indices", &dolfinx::mesh::Geometry::global_indices)
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
      .def_readwrite("coord_mapping", &dolfinx::mesh::Geometry::coord_mapping);

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
      .def(py::init<dolfinx::mesh::CellType>())
      .def("set_connectivity", &dolfinx::mesh::Topology::set_connectivity)
      .def("set_index_map", &dolfinx::mesh::Topology::set_index_map)
      .def("set_interior_facets", &dolfinx::mesh::Topology::set_interior_facets)
      .def("get_facet_permutations",
           &dolfinx::mesh::Topology::get_facet_permutations)
      .def("get_edge_reflections",
           &dolfinx::mesh::Topology::get_edge_reflections)
      .def("get_face_reflections",
           &dolfinx::mesh::Topology::get_face_reflections)
      .def("get_face_rotations", &dolfinx::mesh::Topology::get_face_rotations)
      .def_property_readonly("dim", &dolfinx::mesh::Topology::dim,
                             "Topological dimension")
      .def("connectivity",
           py::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       py::const_))
      .def("hash", &dolfinx::mesh::Topology::hash)
      .def("on_boundary", &dolfinx::mesh::Topology::on_boundary)
      .def("index_map", &dolfinx::mesh::Topology::index_map)
      .def_property_readonly("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def("cell_name", [](const dolfinx::mesh::Topology& self) {
        return dolfinx::mesh::to_string(self.cell_type());
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
             const std::vector<std::int64_t>& global_cell_indices,
             const dolfinx::mesh::GhostMode ghost_mode) {
            return std::make_unique<dolfinx::mesh::Mesh>(
                comm.get(), type, geometry, topology, global_cell_indices,
                ghost_mode);
          }))
      .def_property_readonly(
          "geometry", py::overload_cast<>(&dolfinx::mesh::Mesh::geometry),
          "Mesh geometry")
      .def("hash", &dolfinx::mesh::Mesh::hash)
      .def("hmax", &dolfinx::mesh::Mesh::hmax)
      .def("hmin", &dolfinx::mesh::Mesh::hmin)
      .def("create_entities", &dolfinx::mesh::Mesh::create_entities)
      .def("create_connectivity", &dolfinx::mesh::Mesh::create_connectivity)
      .def("create_connectivity_all",
           &dolfinx::mesh::Mesh::create_connectivity_all)
      .def("mpi_comm",
           [](dolfinx::mesh::Mesh& self) {
             return MPICommWrapper(self.mpi_comm());
           })
      .def("num_entities", &dolfinx::mesh::Mesh::num_entities,
           "Number of mesh entities")
      .def("rmax", &dolfinx::mesh::Mesh::rmax)
      .def("rmin", &dolfinx::mesh::Mesh::rmin)
      .def("num_entities_global", &dolfinx::mesh::Mesh::num_entities_global)
      .def_property_readonly(
          "topology", py::overload_cast<>(&dolfinx::mesh::Mesh::topology),
          "Mesh topology", py::return_value_policy::reference_internal)
      .def("ufl_id", &dolfinx::mesh::Mesh::id)
      .def_property_readonly("id", &dolfinx::mesh::Mesh::id);

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

// dolfinx::mesh::MeshFunction
#define MESHFUNCTION_MACRO(SCALAR, SCALAR_NAME)                                \
  py::class_<dolfinx::mesh::MeshFunction<SCALAR>,                              \
             std::shared_ptr<dolfinx::mesh::MeshFunction<SCALAR>>>(            \
      m, "MeshFunction" #SCALAR_NAME, "DOLFIN MeshFunction object")            \
      .def(py::init<std::shared_ptr<const dolfinx::mesh::Mesh>, std::size_t,   \
                    SCALAR>())                                                 \
      .def(py::init<std::shared_ptr<const dolfinx::mesh::Mesh>,                \
                    const dolfinx::mesh::MeshValueCollection<SCALAR>&,         \
                    const SCALAR&>())                                          \
      .def_property_readonly("dim", &dolfinx::mesh::MeshFunction<SCALAR>::dim) \
      .def_readwrite("name", &dolfinx::mesh::MeshFunction<SCALAR>::name)       \
      .def("mesh", &dolfinx::mesh::MeshFunction<SCALAR>::mesh)                 \
      .def("ufl_id",                                                           \
           [](const dolfinx::mesh::MeshFunction<SCALAR>& self) {               \
             return self.id;                                                   \
           })                                                                  \
      .def("mark", &dolfinx::mesh::MeshFunction<SCALAR>::mark)                 \
      .def_property_readonly(                                                  \
          "values",                                                            \
          py::overload_cast<>(&dolfinx::mesh::MeshFunction<SCALAR>::values));

  MESHFUNCTION_MACRO(int, Int);
  MESHFUNCTION_MACRO(double, Double);
  MESHFUNCTION_MACRO(std::size_t, Sizet);
#undef MESHFUNCTION_MACRO

// dolfinx::mesh::MeshValueCollection
#define MESHVALUECOLLECTION_MACRO(SCALAR, SCALAR_NAME)                         \
  py::class_<dolfinx::mesh::MeshValueCollection<SCALAR>,                       \
             std::shared_ptr<dolfinx::mesh::MeshValueCollection<SCALAR>>>(     \
      m, "MeshValueCollection_" #SCALAR_NAME,                                  \
      "DOLFIN MeshValueCollection object")                                     \
      .def(                                                                    \
          py::init<std::shared_ptr<const dolfinx::mesh::Mesh>, std::size_t>()) \
      .def(py::init<                                                           \
           std::shared_ptr<const dolfinx::mesh::Mesh>, int,                    \
           const Eigen::Ref<const Eigen::Array<                                \
               SCALAR, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&,     \
           const Eigen::Ref<const Eigen::Array<SCALAR, 1, Eigen::Dynamic,      \
                                               Eigen::RowMajor>>&>())          \
      .def_readwrite("name",                                                   \
                     &dolfinx::mesh::MeshValueCollection<SCALAR>::name)        \
      .def_property_readonly("dim",                                            \
                             &dolfinx::mesh::MeshValueCollection<SCALAR>::dim) \
      .def("size", &dolfinx::mesh::MeshValueCollection<SCALAR>::size)          \
      .def("get_value",                                                        \
           &dolfinx::mesh::MeshValueCollection<SCALAR>::get_value)             \
      .def("set_value",                                                        \
           (bool (dolfinx::mesh::MeshValueCollection<SCALAR>::*)(              \
               std::size_t, const SCALAR&))                                    \
               & dolfinx::mesh::MeshValueCollection<SCALAR>::set_value)        \
      .def("set_value",                                                        \
           (bool (dolfinx::mesh::MeshValueCollection<SCALAR>::*)(              \
               std::size_t, std::size_t, const SCALAR&))                       \
               & dolfinx::mesh::MeshValueCollection<SCALAR>::set_value)        \
      .def("values",                                                           \
           (std::map<                                                          \
                std::pair<std::size_t, std::size_t>,                           \
                SCALAR> & (dolfinx::mesh::MeshValueCollection<SCALAR>::*)())   \
               & dolfinx::mesh::MeshValueCollection<SCALAR>::values,           \
           py::return_value_policy::reference)                                 \
      .def("assign",                                                           \
           [](dolfinx::mesh::MeshValueCollection<SCALAR>& self,                \
              const dolfinx::mesh::MeshFunction<SCALAR>& mf) { self = mf; })   \
      .def("assign",                                                           \
           [](dolfinx::mesh::MeshValueCollection<SCALAR>& self,                \
              const dolfinx::mesh::MeshValueCollection<SCALAR>& other) {       \
             self = other;                                                     \
           })

  MESHVALUECOLLECTION_MACRO(bool, bool);
  MESHVALUECOLLECTION_MACRO(int, int);
  MESHVALUECOLLECTION_MACRO(double, double);
  MESHVALUECOLLECTION_MACRO(std::size_t, sizet);
#undef MESHVALUECOLLECTION_MACRO

  // dolfinx::mesh::MeshQuality
  py::class_<dolfinx::mesh::MeshQuality>(m, "MeshQuality", "MeshQuality class")
      .def_static("dihedral_angle_histogram_data",
                  &dolfinx::mesh::MeshQuality::dihedral_angle_histogram_data,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("dihedral_angles_min_max",
                  &dolfinx::mesh::MeshQuality::dihedral_angles_min_max);

  // New Partition interface

  m.def("create_local_adjacency_list",
        &dolfinx::mesh::Partitioning::create_local_adjacency_list);
  m.def("create_distributed_adjacency_list",
        [](const MPICommWrapper comm,
           const dolfinx::graph::AdjacencyList<std::int32_t>& list_local,
           const std::vector<std::int64_t>& global_links,
           const std::vector<bool>& exterior_links) {
          return dolfinx::mesh::Partitioning::create_distributed_adjacency_list(
              comm.get(), list_local, global_links, exterior_links);
        });
  m.def("distribute",
        [](const MPICommWrapper comm,
           const dolfinx::graph::AdjacencyList<std::int64_t>& list,
           const dolfinx::graph::AdjacencyList<std::int32_t>& destinations) {
          return dolfinx::mesh::Partitioning::distribute(comm.get(), list,
                                                         destinations);
        });

  m.def("exchange",
        [](const MPICommWrapper comm,
           const dolfinx::graph::AdjacencyList<std::int64_t>& list,
           const dolfinx::graph::AdjacencyList<std::int32_t>& destinations,
           const std::set<int>& sources) {
          return dolfinx::mesh::Partitioning::exchange(comm.get(), list,
                                                       destinations, sources);
        });

  m.def("partition_cells",
        [](const MPICommWrapper comm, int nparts,
           dolfinx::mesh::CellType cell_type,
           const dolfinx::graph::AdjacencyList<std::int64_t>& cells) {
          return dolfinx::mesh::Partitioning::partition_cells(
              comm.get(), nparts, cell_type, cells);
        });

  m.def("distribute_data",
        [](const MPICommWrapper comm, const std::vector<std::int64_t>& indices,
           const Eigen::Ref<const Eigen::Array<
               double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x) {
          return dolfinx::mesh::Partitioning::distribute_data(comm.get(),
                                                              indices, x);
        });

  m.def("compute_local_to_global_links",
        &dolfinx::mesh::Partitioning::compute_local_to_global_links);

  m.def("compute_local_to_local",
        &dolfinx::mesh::Partitioning::compute_local_to_local);

  m.def("compute_marked_boundary_entities",
        &dolfinx::mesh::compute_marked_boundary_entities);

  // TODO Remove
  m.def("compute_vertex_exterior_markers",
        &dolfinx::mesh::Partitioning::compute_vertex_exterior_markers);

} // namespace dolfinx_wrappers
} // namespace dolfinx_wrappers
