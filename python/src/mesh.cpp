// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include <cfloat>
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshQuality.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Ordering.h>
#include <dolfin/mesh/Partitioning.h>
#include <dolfin/mesh/Topology.h>
#include <dolfin/mesh/cell_types.h>
#include <dolfin/mesh/utils.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

void mesh(py::module& m)
{

  py::enum_<dolfin::mesh::CellType>(m, "CellType")
      .value("point", dolfin::mesh::CellType::point)
      .value("interval", dolfin::mesh::CellType::interval)
      .value("triangle", dolfin::mesh::CellType::triangle)
      .value("quadrilateral", dolfin::mesh::CellType::quadrilateral)
      .value("tetrahedron", dolfin::mesh::CellType::tetrahedron)
      .value("hexahedron", dolfin::mesh::CellType::hexahedron);

  m.def("to_string", &dolfin::mesh::to_string);
  m.def("to_type", &dolfin::mesh::to_type);
  m.def("is_simplex", &dolfin::mesh::is_simplex);

  m.def("cell_num_entities", &dolfin::mesh::cell_num_entities);
  m.def("cell_num_vertices", &dolfin::mesh::num_cell_vertices);

  m.def("volume_entities", &dolfin::mesh::volume_entities,
        "Generalised volume of entities of given dimension.");

  m.def("circumradius", &dolfin::mesh::circumradius);
  m.def("h", &dolfin::mesh::h,
        "Compute maximum distance between any two vertices.");
  m.def("inradius", &dolfin::mesh::inradius, "Compute inradius of cells.");
  m.def("radius_ratio", &dolfin::mesh::radius_ratio);
  m.def("midpoints", &dolfin::mesh::midpoints);

  // dolfin::mesh::GhostMode enums
  py::enum_<dolfin::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfin::mesh::GhostMode::none)
      .value("shared_facet", dolfin::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfin::mesh::GhostMode::shared_vertex);

  // dolfin::mesh::CoordinateDofs class
  py::class_<dolfin::mesh::CoordinateDofs,
             std::shared_ptr<dolfin::mesh::CoordinateDofs>>(
      m, "CoordinateDofs", "CoordinateDofs object")
      .def("entity_points", [](const dolfin::mesh::CoordinateDofs& self) {
        const dolfin::mesh::Connectivity& connectivity = self.entity_points();
        Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
            connections = connectivity.connections();
        const int num_entities = connectivity.entity_positions().size() - 1;

        // FIXME: mesh::CoordinateDofs should know its dimension
        // (entity_size) to handle empty case on a process.
        int entity_size = 0;
        if (num_entities > 0)
        {
          assert(connections.size() % num_entities == 0);
          entity_size = connections.size() / num_entities;
        }
        return py::array({num_entities, entity_size}, connections.data());
      });

  // dolfin::mesh::Geometry class
  py::class_<dolfin::mesh::Geometry, std::shared_ptr<dolfin::mesh::Geometry>>(
      m, "Geometry", "Geometry object")
      .def_property_readonly("dim", &dolfin::mesh::Geometry::dim,
                             "Geometric dimension")
      .def("num_points", &dolfin::mesh::Geometry::num_points)
      .def("num_points_global", &dolfin::mesh::Geometry::num_points_global)
      .def("global_indices", &dolfin::mesh::Geometry::global_indices)
      .def("x", &dolfin::mesh::Geometry::x,
           py::return_value_policy::reference_internal,
           "Return coordinates of a point")
      .def_property(
          "points", py::overload_cast<>(&dolfin::mesh::Geometry::points),
          [](dolfin::mesh::Geometry& self, dolfin::EigenRowArrayXXd values) {
            self.points() = values;
          },
          "Return coordinates of all points")
      .def_readwrite("coord_mapping", &dolfin::mesh::Geometry::coord_mapping);

  // dolfin::mesh::Topology class
  py::class_<dolfin::mesh::Topology, std::shared_ptr<dolfin::mesh::Topology>>(
      m, "Topology", "DOLFIN Topology object")
      .def_property_readonly("dim", &dolfin::mesh::Topology::dim,
                             "Topological dimension")
      .def("connectivity",
           py::overload_cast<std::size_t, std::size_t>(
               &dolfin::mesh::Topology::connectivity, py::const_))
      .def("size", &dolfin::mesh::Topology::size)
      .def("hash", &dolfin::mesh::Topology::hash)
      .def("have_global_indices", &dolfin::mesh::Topology::have_global_indices)
      .def("ghost_offset", &dolfin::mesh::Topology::ghost_offset)
      .def("cell_owner",
           py::overload_cast<>(&dolfin::mesh::Topology::cell_owner, py::const_))
      .def("global_indices",
           [](const dolfin::mesh::Topology& self, int dim) {
             auto& indices = self.global_indices(dim);
             return py::array_t<std::int64_t>(indices.size(), indices.data());
           })
      .def("shared_entities",
           py::overload_cast<int>(&dolfin::mesh::Topology::shared_entities))
      .def("str", &dolfin::mesh::Topology::str);

  // dolfin::mesh::Mesh
  py::class_<dolfin::mesh::Mesh, std::shared_ptr<dolfin::mesh::Mesh>>(
      m, "Mesh", py::dynamic_attr(), "Mesh object")
      .def(py::init(
          [](const MPICommWrapper comm, dolfin::mesh::CellType type,
             const Eigen::Ref<const dolfin::EigenRowArrayXXd> geometry,
             const Eigen::Ref<const dolfin::EigenRowArrayXXi64> topology,
             const std::vector<std::int64_t>& global_cell_indices,
             const dolfin::mesh::GhostMode ghost_mode) {
            return std::make_unique<dolfin::mesh::Mesh>(
                comm.get(), type, geometry, topology, global_cell_indices,
                ghost_mode);
          }))
      .def(
          "cells",
          [](const dolfin::mesh::Mesh& self) {
            const std::uint32_t tdim = self.topology().dim();
            return py::array(
                {(std::int32_t)self.topology().size(tdim),
                 (std::int32_t)dolfin::mesh::num_cell_vertices(self.cell_type)},
                self.topology().connectivity(tdim, 0)->connections().data());
          })
      .def_property_readonly("geometry",
                             py::overload_cast<>(&dolfin::mesh::Mesh::geometry),
                             "Mesh geometry")
      .def(
          "coordinate_dofs",
          py::overload_cast<>(&dolfin::mesh::Mesh::coordinate_dofs, py::const_))
      .def("degree", &dolfin::mesh::Mesh::degree)
      .def("hash", &dolfin::mesh::Mesh::hash)
      .def("hmax", &dolfin::mesh::Mesh::hmax)
      .def("hmin", &dolfin::mesh::Mesh::hmin)
      .def("create_global_indices", &dolfin::mesh::Mesh::create_global_indices)
      .def("create_entities", &dolfin::mesh::Mesh::create_entities)
      .def("create_connectivity", &dolfin::mesh::Mesh::create_connectivity)
      .def("create_connectivity_all",
           &dolfin::mesh::Mesh::create_connectivity_all)
      .def("mpi_comm",
           [](dolfin::mesh::Mesh& self) {
             return MPICommWrapper(self.mpi_comm());
           })
      .def("num_entities", &dolfin::mesh::Mesh::num_entities,
           "Number of mesh entities")
      .def("rmax", &dolfin::mesh::Mesh::rmax)
      .def("rmin", &dolfin::mesh::Mesh::rmin)
      .def("num_entities_global", &dolfin::mesh::Mesh::num_entities_global)
      .def_property_readonly(
          "topology", py::overload_cast<>(&dolfin::mesh::Mesh::topology),
          "Mesh topology", py::return_value_policy::reference_internal)
      .def_readonly("cell_type", &dolfin::mesh::Mesh::cell_type)
      .def("ufl_id", &dolfin::mesh::Mesh::id)
      .def_property_readonly("id", &dolfin::mesh::Mesh::id)
      .def("cell_name", [](const dolfin::mesh::Mesh& self) {
        return dolfin::mesh::to_string(self.cell_type);
      });

  // dolfin::mesh::Connectivity class
  py::class_<dolfin::mesh::Connectivity,
             std::shared_ptr<dolfin::mesh::Connectivity>>(m, "Connectivity",
                                                          "Connectivity object")
      .def("connections",
           [](const dolfin::mesh::Connectivity& self, std::size_t i) {
             return Eigen::Map<const dolfin::EigenArrayXi32>(
                 self.connections(i), self.size(i));
           },
           "Connections for a single mesh entity",
           py::return_value_policy::reference_internal)
      .def("connections",
           py::overload_cast<>(&dolfin::mesh::Connectivity::connections),
           "Connections for all mesh entities")
      .def("pos",
           py::overload_cast<>(&dolfin::mesh::Connectivity::entity_positions),
           "Index to each entity in the connectivity array")
      .def("size", &dolfin::mesh::Connectivity::size);

  // dolfin::mesh::MeshEntity class
  py::class_<dolfin::mesh::MeshEntity,
             std::shared_ptr<dolfin::mesh::MeshEntity>>(m, "MeshEntity",
                                                        "MeshEntity object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t, std::size_t>())
      .def_property_readonly("dim", &dolfin::mesh::MeshEntity::dim,
                             "Topological dimension")
      .def("mesh", &dolfin::mesh::MeshEntity::mesh, "Associated mesh")
      .def("index",
           py::overload_cast<>(&dolfin::mesh::MeshEntity::index, py::const_),
           "Entity index")
      .def("entities",
           [](dolfin::mesh::MeshEntity& self, std::size_t dim) {
             if (self.dim() == dim)
               return py::array(1, self.entities(dim));
             else
             {
               assert(self.mesh.topology().connectivity(self.dim(), dim));
               const int num_entities = self.mesh()
                                            .topology()
                                            .connectivity(self.dim(), dim)
                                            ->size(self.index());
               return py::array(num_entities, self.entities(dim));
             }
           },
           py::return_value_policy::reference_internal)
      .def("__str__",
           [](dolfin::mesh::MeshEntity& self) { return self.str(false); });

  py::class_<dolfin::mesh::EntityRange,
             std::shared_ptr<dolfin::mesh::EntityRange>>(
      m, "EntityRange", "Range for iteration over entities of another entity")
      .def(py::init<const dolfin::mesh::MeshEntity&, int>())
      .def("__iter__", [](const dolfin::mesh::EntityRange& r) {
        return py::make_iterator(r.begin(), r.end());
      });

// dolfin::mesh::MeshFunction
#define MESHFUNCTION_MACRO(SCALAR, SCALAR_NAME)                                \
  py::class_<dolfin::mesh::MeshFunction<SCALAR>,                               \
             std::shared_ptr<dolfin::mesh::MeshFunction<SCALAR>>>(             \
      m, "MeshFunction" #SCALAR_NAME, "DOLFIN MeshFunction object")            \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>, std::size_t,    \
                    SCALAR>())                                                 \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>,                 \
                    const dolfin::mesh::MeshValueCollection<SCALAR>&,          \
                    const SCALAR&>())                                          \
      .def_property_readonly("dim", &dolfin::mesh::MeshFunction<SCALAR>::dim)  \
      .def_readwrite("name", &dolfin::mesh::MeshFunction<SCALAR>::name)        \
      .def("mesh", &dolfin::mesh::MeshFunction<SCALAR>::mesh)                  \
      .def("ufl_id",                                                           \
           [](const dolfin::mesh::MeshFunction<SCALAR>& self) {                \
             return self.id;                                                   \
           })                                                                  \
      .def("mark", &dolfin::mesh::MeshFunction<SCALAR>::mark)                  \
      .def_property_readonly(                                                  \
          "values",                                                            \
          py::overload_cast<>(&dolfin::mesh::MeshFunction<SCALAR>::values));

  MESHFUNCTION_MACRO(int, Int);
  MESHFUNCTION_MACRO(double, Double);
  MESHFUNCTION_MACRO(std::size_t, Sizet);
#undef MESHFUNCTION_MACRO

// dolfin::mesh::MeshValueCollection
#define MESHVALUECOLLECTION_MACRO(SCALAR, SCALAR_NAME)                         \
  py::class_<dolfin::mesh::MeshValueCollection<SCALAR>,                        \
             std::shared_ptr<dolfin::mesh::MeshValueCollection<SCALAR>>>(      \
      m, "MeshValueCollection_" #SCALAR_NAME,                                  \
      "DOLFIN MeshValueCollection object")                                     \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>, std::size_t>()) \
      .def_readwrite("name", &dolfin::mesh::MeshValueCollection<SCALAR>::name) \
      .def_property_readonly("dim",                                            \
                             &dolfin::mesh::MeshValueCollection<SCALAR>::dim)  \
      .def("size", &dolfin::mesh::MeshValueCollection<SCALAR>::size)           \
      .def("get_value", &dolfin::mesh::MeshValueCollection<SCALAR>::get_value) \
      .def("set_value",                                                        \
           (bool (dolfin::mesh::MeshValueCollection<SCALAR>::*)(               \
               std::size_t, const SCALAR&))                                    \
               & dolfin::mesh::MeshValueCollection<SCALAR>::set_value)         \
      .def("set_value",                                                        \
           (bool (dolfin::mesh::MeshValueCollection<SCALAR>::*)(               \
               std::size_t, std::size_t, const SCALAR&))                       \
               & dolfin::mesh::MeshValueCollection<SCALAR>::set_value)         \
      .def("values",                                                           \
           (std::map<                                                          \
                std::pair<std::size_t, std::size_t>,                           \
                SCALAR> & (dolfin::mesh::MeshValueCollection<SCALAR>::*)())    \
               & dolfin::mesh::MeshValueCollection<SCALAR>::values,            \
           py::return_value_policy::reference)                                 \
      .def("assign",                                                           \
           [](dolfin::mesh::MeshValueCollection<SCALAR>& self,                 \
              const dolfin::mesh::MeshFunction<SCALAR>& mf) { self = mf; })    \
      .def("assign",                                                           \
           [](dolfin::mesh::MeshValueCollection<SCALAR>& self,                 \
              const dolfin::mesh::MeshValueCollection<SCALAR>& other) {        \
             self = other;                                                     \
           })

  MESHVALUECOLLECTION_MACRO(bool, bool);
  MESHVALUECOLLECTION_MACRO(int, int);
  MESHVALUECOLLECTION_MACRO(double, double);
  MESHVALUECOLLECTION_MACRO(std::size_t, sizet);
#undef MESHVALUECOLLECTION_MACRO

  // dolfin::mesh::MeshQuality
  py::class_<dolfin::mesh::MeshQuality>(m, "MeshQuality", "MeshQuality class")
      .def_static("dihedral_angle_histogram_data",
                  &dolfin::mesh::MeshQuality::dihedral_angle_histogram_data,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("dihedral_angles_min_max",
                  &dolfin::mesh::MeshQuality::dihedral_angles_min_max);

  py::class_<dolfin::mesh::Ordering>(m, "Ordering", "Order mesh cell entities")
      .def_static("order_simplex", &dolfin::mesh::Ordering::order_simplex)
      .def_static("is_ordered_simplex",
                  &dolfin::mesh::Ordering::is_ordered_simplex);

} // namespace dolfin_wrappers
} // namespace dolfin_wrappers
