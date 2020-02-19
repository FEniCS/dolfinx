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
#include <dolfinx/mesh/CoordinateDofs.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/MeshQuality.h>
#include <dolfinx/mesh/MeshValueCollection.h>
#include <dolfinx/mesh/Ordering.h>
#include <dolfinx/mesh/PartitionData.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
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

  m.def("cell_num_entities", &dolfinx::mesh::cell_num_entities);
  m.def("cell_num_vertices", &dolfinx::mesh::num_cell_vertices);

  m.def("volume_entities", &dolfinx::mesh::volume_entities,
        "Generalised volume of entities of given dimension.");

  m.def("circumradius", &dolfinx::mesh::circumradius);
  m.def("h", &dolfinx::mesh::h,
        "Compute maximum distance between any two vertices.");
  m.def("inradius", &dolfinx::mesh::inradius, "Compute inradius of cells.");
  m.def("radius_ratio", &dolfinx::mesh::radius_ratio);
  m.def("midpoints", &dolfinx::mesh::midpoints);

  // dolfinx::mesh::GhostMode enums
  py::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfinx::mesh::GhostMode::shared_vertex);

  // dolfinx::mesh::Partitioner enums
  py::enum_<dolfinx::mesh::Partitioner>(m, "Partitioner")
      .value("scotch", dolfinx::mesh::Partitioner::scotch)
      .value("kahip", dolfinx::mesh::Partitioner::kahip)
      .value("parmetis", dolfinx::mesh::Partitioner::parmetis);

  // dolfinx::mesh::CoordinateDofs class
  py::class_<dolfinx::mesh::CoordinateDofs,
             std::shared_ptr<dolfinx::mesh::CoordinateDofs>>(
      m, "CoordinateDofs", "CoordinateDofs object")
      .def(
          "entity_points",
          [](const dolfinx::mesh::CoordinateDofs& self) {
            const dolfinx::graph::AdjacencyList<std::int32_t>& connectivity
                = self.entity_points();
            Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
                connections = connectivity.array();
            const int num_entities = connectivity.offsets().size() - 1;

            // FIXME: mesh::CoordinateDofs should know its dimension
            // (entity_size) to handle empty case on a process.
            int entity_size = 0;
            if (num_entities > 0)
            {
              assert(connections.size() % num_entities == 0);
              entity_size = connections.size() / num_entities;
            }
            return py::array({num_entities, entity_size}, connections.data(),
                             py::none());
          },
          py::return_value_policy::reference_internal);

  // dolfinx::mesh::Geometry class
  py::class_<dolfinx::mesh::Geometry, std::shared_ptr<dolfinx::mesh::Geometry>>(
      m, "Geometry", "Geometry object")
      .def_property_readonly("dim", &dolfinx::mesh::Geometry::dim,
                             "Geometric dimension")
      .def("num_points", &dolfinx::mesh::Geometry::num_points)
      .def("num_points_global", &dolfinx::mesh::Geometry::num_points_global)
      .def("global_indices", &dolfinx::mesh::Geometry::global_indices)
      .def("x", &dolfinx::mesh::Geometry::x,
           py::return_value_policy::reference_internal,
           "Return coordinates of a point")
      .def_property(
          "points",
          // Get
          py::overload_cast<>(&dolfinx::mesh::Geometry::points),
          // Set
          [](dolfinx::mesh::Geometry& self,
             const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>& values) {
            self.points() = values;
          },
          py::return_value_policy::reference_internal,
          "Return coordinates of all points")
      .def_readwrite("coord_mapping", &dolfinx::mesh::Geometry::coord_mapping);

  // dolfinx::mesh::Topology class
  py::class_<dolfinx::mesh::Topology, std::shared_ptr<dolfinx::mesh::Topology>>(
      m, "Topology", "DOLFIN Topology object")
      .def_property_readonly("dim", &dolfinx::mesh::Topology::dim,
                             "Topological dimension")
      .def("connectivity",
           py::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       py::const_))
      .def("hash", &dolfinx::mesh::Topology::hash)
      .def("on_boundary", &dolfinx::mesh::Topology::on_boundary)
      .def("index_map", &dolfinx::mesh::Topology::index_map)
      .def_property_readonly("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def("cell_name",
           [](const dolfinx::mesh::Topology& self) {
             return dolfinx::mesh::to_string(self.cell_type());
           })
      .def("str", &dolfinx::mesh::Topology::str);

  // dolfinx::mesh::Mesh
  py::class_<dolfinx::mesh::Mesh, std::shared_ptr<dolfinx::mesh::Mesh>>(
      m, "Mesh", py::dynamic_attr(), "Mesh object")
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
      .def("cells",
           [](const dolfinx::mesh::Mesh& self) {
             const int tdim = self.topology().dim();
             auto map = self.topology().index_map(tdim);
             assert(map);
             const std::int32_t size = map->size_local() + map->num_ghosts();
             return py::array(
                 {size, (std::int32_t)dolfinx::mesh::num_cell_vertices(
                            self.topology().cell_type())},
                 self.topology().connectivity(tdim, 0)->array().data(),
                 py::none());
           },
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "geometry", py::overload_cast<>(&dolfinx::mesh::Mesh::geometry),
          "Mesh geometry")
      .def("coordinate_dofs",
           py::overload_cast<>(&dolfinx::mesh::Mesh::coordinate_dofs,
                               py::const_))
      .def("degree", &dolfinx::mesh::Mesh::degree)
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
           py::return_value_policy::reference_internal)
      .def("__str__",
           [](dolfinx::mesh::MeshEntity& self) { return self.str(false); });

  py::class_<dolfinx::mesh::EntityRange,
             std::shared_ptr<dolfinx::mesh::EntityRange>>(
      m, "EntityRange", "Range for iteration over entities of another entity")
      .def(py::init<const dolfinx::mesh::MeshEntity&, int>())
      .def("__iter__", [](const dolfinx::mesh::EntityRange& r) {
        return py::make_iterator(r.begin(), r.end());
      });

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

  py::class_<dolfinx::mesh::Ordering>(m, "Ordering", "Order mesh cell entities")
      .def_static("order_simplex", &dolfinx::mesh::Ordering::order_simplex)
      .def_static("is_ordered_simplex",
                  &dolfinx::mesh::Ordering::is_ordered_simplex);

  // dolfinx::mesh::PartitionData class
  py::class_<dolfinx::mesh::PartitionData,
             std::shared_ptr<dolfinx::mesh::PartitionData>>(
      m, "PartitionData", "PartitionData object")
      .def(py::init(
          [](const std::vector<int>& cell_partition,
             const std::map<std::int64_t, std::vector<int>>& ghost_procs) {
            return dolfinx::mesh::PartitionData(cell_partition, ghost_procs);
          }))
      .def("num_procs", &dolfinx::mesh::PartitionData::num_procs)
      .def("size", &dolfinx::mesh::PartitionData::num_ghosts)
      .def("num_ghosts", &dolfinx::mesh::PartitionData::num_ghosts);

  // dolfinx::mesh::Partitioning::partition_cells
  m.def(
      "partition_cells",
      [](const MPICommWrapper comm, int nparts,
         dolfinx::mesh::CellType cell_type,
         const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>&
             cells,
         dolfinx::mesh::Partitioner partitioner) {
        return dolfinx::mesh::Partitioning::partition_cells(
            comm.get(), nparts, cell_type, cells, partitioner);
      });

  m.def(
      "build_distributed_mesh",
      [](const MPICommWrapper comm, dolfinx::mesh::CellType cell_type,
         const Eigen::Ref<const Eigen::Array<
             double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& points,
         const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>&
             cells,
         const std::vector<std::int64_t>& global_cell_indices,
         const dolfinx::mesh::GhostMode ghost_mode,
         const dolfinx::mesh::Partitioner graph_partitioner) {
        return dolfinx::mesh::Partitioning::build_distributed_mesh(
            comm.get(), cell_type, points, cells, global_cell_indices,
            ghost_mode, graph_partitioner);
      });

  m.def(
      "build_from_partition",
      [](const MPICommWrapper comm, dolfinx::mesh::CellType cell_type,
         const Eigen::Ref<const Eigen::Array<
             double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& points,
         const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>&
             cells,
         const std::vector<std::int64_t>& global_cell_indices,
         const dolfinx::mesh::GhostMode ghost_mode,
         const dolfinx::mesh::PartitionData& cell_partition) {
        return dolfinx::mesh::Partitioning::build_from_partition(
            comm.get(), cell_type, points, cells, global_cell_indices,
            ghost_mode, cell_partition);
      });

  m.def(
      "ghost_cell_mapping",
      [](const MPICommWrapper comm, py::array_t<int> parttition,
         dolfinx::mesh::CellType cell_type,
         const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>&
             cells) {
        std::vector<int> part(parttition.data(),
                              parttition.data() + parttition.size());
        return dolfinx::mesh::Partitioning::compute_halo_cells(
            comm.get(), part, cell_type, cells);
      });

  m.def("compute_marked_boundary_entities",
        &dolfinx::mesh::compute_marked_boundary_entities);
}
} // namespace dolfinx_wrappers
