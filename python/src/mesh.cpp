// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cfloat>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshQuality.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Ordering.h>
#include <dolfin/mesh/Partitioning.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Topology.h>
#include <dolfin/mesh/Vertex.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{

void mesh(py::module& m)
{

  // dolfin::mesh::CellType
  py::class_<dolfin::mesh::CellType> celltype(m, "CellType");

  // dolfin::mesh::CellType enums
  py::enum_<dolfin::mesh::CellType::Type>(celltype, "Type")
      .value("point", dolfin::mesh::CellType::Type::point)
      .value("interval", dolfin::mesh::CellType::Type::interval)
      .value("triangle", dolfin::mesh::CellType::Type::triangle)
      .value("quadrilateral", dolfin::mesh::CellType::Type::quadrilateral)
      .value("tetrahedron", dolfin::mesh::CellType::Type::tetrahedron)
      .value("hexahedron", dolfin::mesh::CellType::Type::hexahedron);

  celltype.def("type2string", &dolfin::mesh::CellType::type2string)
      .def("string2type", &dolfin::mesh::CellType::string2type)
      .def("cell_type", &dolfin::mesh::CellType::cell_type)
      .def("num_entities", &dolfin::mesh::CellType::num_entities)
      .def("description", &dolfin::mesh::CellType::description)
      .def_property_readonly("is_simplex", &dolfin::mesh::CellType::is_simplex);

  // dolfin::mesh::GhostMode enums
  py::enum_<dolfin::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfin::mesh::GhostMode::none)
      .value("shared_facet", dolfin::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfin::mesh::GhostMode::shared_vertex);

  // dolfin::mesh::CoordinateDofs class
  py::class_<dolfin::mesh::CoordinateDofs,
             std::shared_ptr<dolfin::mesh::CoordinateDofs>>(
      m, "CoordinateDofs", "CoordinateDofs object")
      .def("entity_points",
           [](const dolfin::mesh::CoordinateDofs& self, std::size_t dim) {
             const dolfin::mesh::Connectivity& connectivity
                 = self.entity_points(dim);
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
      .def("have_shared_entities",
           &dolfin::mesh::Topology::have_shared_entities)
      .def("shared_entities",
           py::overload_cast<int>(&dolfin::mesh::Topology::shared_entities))
      .def("str", &dolfin::mesh::Topology::str);

  // dolfin::mesh::Mesh
  py::class_<dolfin::mesh::Mesh, std::shared_ptr<dolfin::mesh::Mesh>>(
      m, "Mesh", py::dynamic_attr(), "Mesh object")
      .def(py::init(
          [](const MPICommWrapper comm, dolfin::mesh::CellType::Type type,
             const Eigen::Ref<const dolfin::EigenRowArrayXXd> geometry,
             const Eigen::Ref<const dolfin::EigenRowArrayXXi64> topology,
             const std::vector<std::int64_t>& global_cell_indices,
             const dolfin::mesh::GhostMode ghost_mode) {
            return std::make_unique<dolfin::mesh::Mesh>(
                comm.get(), type, geometry, topology, global_cell_indices,
                ghost_mode);
          }))
      .def("cells",
           [](const dolfin::mesh::Mesh& self) {
             const std::uint32_t tdim = self.topology().dim();
             return py::array(
                 {(std::int32_t)self.topology().size(tdim),
                  (std::int32_t)self.type().num_vertices(tdim)},
                 self.topology().connectivity(tdim, 0)->connections().data());
           })
      .def_property_readonly("geometry",
                             py::overload_cast<>(&dolfin::mesh::Mesh::geometry),
                             "Mesh geometry")
      .def("coordinate_dofs", &dolfin::mesh::Mesh::coordinate_dofs)
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
      .def("type", py::overload_cast<>(&dolfin::mesh::Mesh::type, py::const_),
           py::return_value_policy::reference)
      .def("ufl_id", &dolfin::mesh::Mesh::id)
      .def_property_readonly("id", &dolfin::mesh::Mesh::id)
      .def("cell_name", [](const dolfin::mesh::Mesh& self) {
        return dolfin::mesh::CellType::type2string(self.type().cell_type());
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
           py::return_value_policy::reference_internal)
      .def("connections",
           py::overload_cast<>(&dolfin::mesh::Connectivity::connections),
           "Return all connectivities")
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
           "Index")
      .def("global_index", &dolfin::mesh::MeshEntity::global_index,
           "Global index")
      .def("num_entities", &dolfin::mesh::MeshEntity::num_entities,
           "Number of incident entities of given dimension")
      .def("num_global_entities",
           &dolfin::mesh::MeshEntity::num_global_entities,
           "Global number of incident entities of given dimension")
      .def("entities",
           [](dolfin::mesh::MeshEntity& self, std::size_t dim) {
             return Eigen::Map<const dolfin::EigenArrayXi32>(
                 self.entities(dim), self.num_entities(dim));
           },
           py::return_value_policy::reference_internal)
      .def("midpoint", &dolfin::mesh::MeshEntity::midpoint,
           "Midpoint of Entity")
      .def("sharing_processes", &dolfin::mesh::MeshEntity::sharing_processes)
      .def("is_shared", &dolfin::mesh::MeshEntity::is_shared)
      .def("is_ghost", &dolfin::mesh::MeshEntity::is_ghost)
      .def("__str__",
           [](dolfin::mesh::MeshEntity& self) { return self.str(false); });

  // dolfin::mesh::Vertex
  py::class_<dolfin::mesh::Vertex, std::shared_ptr<dolfin::mesh::Vertex>,
             dolfin::mesh::MeshEntity>(m, "Vertex", "Vertex object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def("point", &dolfin::mesh::Vertex::x);

  // dolfin::mesh::Edge
  py::class_<dolfin::mesh::Edge, std::shared_ptr<dolfin::mesh::Edge>,
             dolfin::mesh::MeshEntity>(m, "Edge", "Edge object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def("dot", &dolfin::mesh::Edge::dot)
      .def("length", &dolfin::mesh::Edge::length);

  // dolfin::mesh::Face
  py::class_<dolfin::mesh::Face, std::shared_ptr<dolfin::mesh::Face>,
             dolfin::mesh::MeshEntity>(m, "Face", "Face object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def("normal", &dolfin::mesh::Face::normal)
      .def("area", &dolfin::mesh::Face::area);

  // dolfin::mesh::Facet
  py::class_<dolfin::mesh::Facet, std::shared_ptr<dolfin::mesh::Facet>,
             dolfin::mesh::MeshEntity>(m, "Facet", "Facet object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def("exterior", &dolfin::mesh::Facet::exterior)
      .def("normal", &dolfin::mesh::Facet::normal);

  // dolfin::mesh::Cell
  py::class_<dolfin::mesh::Cell, std::shared_ptr<dolfin::mesh::Cell>,
             dolfin::mesh::MeshEntity>(m, "Cell", "DOLFIN Cell object")
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def("distance", &dolfin::mesh::Cell::distance)
      .def("facet_area", &dolfin::mesh::Cell::facet_area)
      .def("h", &dolfin::mesh::Cell::h)
      .def("inradius", &dolfin::mesh::Cell::inradius)
      .def("normal", &dolfin::mesh::Cell::normal)
      .def("circumradius", &dolfin::mesh::Cell::circumradius)
      .def("radius_ratio", &dolfin::mesh::Cell::radius_ratio)
      .def("volume", &dolfin::mesh::Cell::volume);

  py::class_<
      dolfin::mesh::MeshRange<dolfin::mesh::MeshEntity>,
      std::shared_ptr<dolfin::mesh::MeshRange<dolfin::mesh::MeshEntity>>>(
      m, "MeshEntities", "Range for iteration over entities of a Mesh")
      .def(py::init<const dolfin::mesh::Mesh&, int>())
      .def("__iter__",
           [](const dolfin::mesh::MeshRange<dolfin::mesh::MeshEntity>& r) {
             return py::make_iterator(r.begin(), r.end());
           });

  py::class_<
      dolfin::mesh::EntityRange<dolfin::mesh::MeshEntity>,
      std::shared_ptr<dolfin::mesh::EntityRange<dolfin::mesh::MeshEntity>>>(
      m, "EntityRange", "Range for iteration over entities of another entity")
      .def(py::init<const dolfin::mesh::MeshEntity&, int>())
      .def("__iter__",
           [](const dolfin::mesh::EntityRange<dolfin::mesh::MeshEntity>& r) {
             return py::make_iterator(r.begin(), r.end());
           });

  // dolfin::mesh::MeshRangeType enums
  py::enum_<dolfin::mesh::MeshRangeType>(m, "MeshRangeType")
      .value("REGULAR", dolfin::mesh::MeshRangeType::REGULAR)
      .value("ALL", dolfin::mesh::MeshRangeType::ALL)
      .value("GHOST", dolfin::mesh::MeshRangeType::GHOST);

// dolfin::mesh::MeshIterator (Cells, Facets, Faces, Edges, Vertices)
#define MESHITERATOR_MACRO(TYPE, ENTITYNAME)                                   \
  py::class_<dolfin::mesh::MeshRange<dolfin::ENTITYNAME>,                      \
             std::shared_ptr<dolfin::mesh::MeshRange<dolfin::ENTITYNAME>>>(    \
      m, #TYPE,                                                                \
      "Range for iterating over entities of type " #ENTITYNAME " of a Mesh")   \
      .def(py::init<const dolfin::mesh::Mesh&>())                              \
      .def(py::init<const dolfin::mesh::Mesh&, dolfin::mesh::MeshRangeType>()) \
      .def("__iter__",                                                         \
           [](const dolfin::mesh::MeshRange<dolfin::ENTITYNAME>& c) {          \
             return py::make_iterator(c.begin(), c.end());                     \
           });

  MESHITERATOR_MACRO(Cells, mesh::Cell);
  MESHITERATOR_MACRO(Facets, mesh::Facet);
  MESHITERATOR_MACRO(Faces, mesh::Face);
  MESHITERATOR_MACRO(Edges, mesh::Edge);
  MESHITERATOR_MACRO(Vertices, mesh::Vertex);
#undef MESHITERATOR_MACRO

#define MESHENTITYITERATOR_MACRO(TYPE, ENTITYNAME)                             \
  py::class_<dolfin::mesh::EntityRange<dolfin::ENTITYNAME>,                    \
             std::shared_ptr<dolfin::mesh::EntityRange<dolfin::ENTITYNAME>>>(  \
      m, #TYPE,                                                                \
      "Range for iterating over entities of type " #ENTITYNAME                 \
      " incident to a MeshEntity")                                             \
      .def(py::init<const dolfin::mesh::MeshEntity&>())                        \
      .def("__iter__",                                                         \
           [](const dolfin::mesh::EntityRange<dolfin::ENTITYNAME>& c) {        \
             return py::make_iterator(c.begin(), c.end());                     \
           });

  MESHENTITYITERATOR_MACRO(CellRange, mesh::Cell);
  MESHENTITYITERATOR_MACRO(FacetRange, mesh::Facet);
  MESHENTITYITERATOR_MACRO(FaceRange, mesh::Face);
  MESHENTITYITERATOR_MACRO(EdgeRange, mesh::Edge);
  MESHENTITYITERATOR_MACRO(VertexRange, mesh::Vertex);
#undef MESHENTITYITERATOR_MACRO

// dolfin::mesh::MeshFunction
#define MESHFUNCTION_MACRO(SCALAR, SCALAR_NAME)                                \
  py::class_<dolfin::mesh::MeshFunction<SCALAR>,                               \
             std::shared_ptr<dolfin::mesh::MeshFunction<SCALAR>>,              \
             dolfin::common::Variable>(m, "MeshFunction" #SCALAR_NAME,         \
                                       "DOLFIN MeshFunction object")           \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>, std::size_t,    \
                    SCALAR>())                                                 \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>,                 \
                    const dolfin::mesh::MeshValueCollection<SCALAR>&,          \
                    const SCALAR&>())                                          \
      .def("__getitem__",                                                      \
           (const SCALAR& (dolfin::mesh::MeshFunction<SCALAR>::*)(std::size_t) \
                const)                                                         \
               & dolfin::mesh::MeshFunction<SCALAR>::operator[])               \
      .def("__setitem__",                                                      \
           [](dolfin::mesh::MeshFunction<SCALAR>& self, std::size_t index,     \
              SCALAR value) { self.operator[](index) = value; })               \
      .def("__getitem__",                                                      \
           (const SCALAR& (dolfin::mesh::MeshFunction<                         \
                           SCALAR>::*)(const dolfin::mesh::MeshEntity&)const)  \
               & dolfin::mesh::MeshFunction<SCALAR>::operator[])               \
      .def("__setitem__",                                                      \
           [](dolfin::mesh::MeshFunction<SCALAR>& self,                        \
              const dolfin::mesh::MeshEntity& index,                           \
              SCALAR value) { self.operator[](index) = value; })               \
      .def("__len__", &dolfin::mesh::MeshFunction<SCALAR>::size)               \
      .def_property_readonly("dim", &dolfin::mesh::MeshFunction<SCALAR>::dim)  \
      .def("size", &dolfin::mesh::MeshFunction<SCALAR>::size)                  \
      .def("ufl_id", &dolfin::mesh::MeshFunction<SCALAR>::id)                  \
      .def("mesh", &dolfin::mesh::MeshFunction<SCALAR>::mesh)                  \
      .def("set_values", &dolfin::mesh::MeshFunction<SCALAR>::set_values)      \
      .def("set_all", [](dolfin::mesh::MeshFunction<SCALAR>& self,             \
                         const SCALAR& value) { self = value; })               \
      .def("where_equal", &dolfin::mesh::MeshFunction<SCALAR>::where_equal)    \
      .def("array",                                                            \
           [](dolfin::mesh::MeshFunction<SCALAR>& self) {                      \
             return Eigen::Map<Eigen::Array<SCALAR, Eigen::Dynamic, 1>>(       \
                 self.values(), self.size());                                  \
           },                                                                  \
           py::return_value_policy::reference_internal)

  MESHFUNCTION_MACRO(bool, Bool);
  MESHFUNCTION_MACRO(int, Int);
  MESHFUNCTION_MACRO(double, Double);
  MESHFUNCTION_MACRO(std::size_t, Sizet);
#undef MESHFUNCTION_MACRO

// dolfin::mesh::MeshValueCollection
#define MESHVALUECOLLECTION_MACRO(SCALAR, SCALAR_NAME)                         \
  py::class_<dolfin::mesh::MeshValueCollection<SCALAR>,                        \
             std::shared_ptr<dolfin::mesh::MeshValueCollection<SCALAR>>,       \
             dolfin::common::Variable>(m, "MeshValueCollection_" #SCALAR_NAME, \
                                       "DOLFIN MeshValueCollection object")    \
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>, std::size_t>()) \
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
      .def_static("radius_ratios", &dolfin::mesh::MeshQuality::radius_ratios)
      .def_static("radius_ratio_histogram_data",
                  &dolfin::mesh::MeshQuality::radius_ratio_histogram_data,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("dihedral_angle_histogram_data",
                  &dolfin::mesh::MeshQuality::dihedral_angle_histogram_data,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("radius_ratio_min_max",
                  &dolfin::mesh::MeshQuality::radius_ratio_min_max)
      .def_static("dihedral_angles_min_max",
                  &dolfin::mesh::MeshQuality::dihedral_angles_min_max);

  // dolfin::SubDomain trampoline class for user overloading from
  // Python
  class PySubDomain : public dolfin::mesh::SubDomain
  {
    using dolfin::mesh::SubDomain::SubDomain;

    dolfin::EigenArrayXb inside(Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
                                bool on_boundary) const override
    {
      PYBIND11_OVERLOAD(dolfin::EigenArrayXb, dolfin::mesh::SubDomain, inside,
                        x, on_boundary);
    }

    void map(Eigen::Ref<const dolfin::EigenArrayXd> x,
             Eigen::Ref<dolfin::EigenArrayXd> y) const override
    {
      PYBIND11_OVERLOAD(void, dolfin::mesh::SubDomain, map, x, y);
    }
  };

  py::class_<dolfin::mesh::Ordering>(m, "Ordering", "Order mesh cell entities")
      .def_static("order_simplex", &dolfin::mesh::Ordering::order_simplex)
      .def_static("is_ordered_simplex",
                  &dolfin::mesh::Ordering::is_ordered_simplex);

  // dolfin::mesh::SubDomain
  py::class_<dolfin::mesh::SubDomain, std::shared_ptr<dolfin::mesh::SubDomain>,
             PySubDomain>(m, "SubDomain", "Sub-domain object")
      .def(py::init<double>(), py::arg("map_tol") = DBL_EPSILON)
      .def("inside", &dolfin::mesh::SubDomain::inside, py::arg("x").noconvert(),
           py::arg("on_boundary"))
      .def("map", &dolfin::mesh::SubDomain::map, py::arg("x").noconvert(),
           py::arg("y").noconvert())
      .def("mark", &dolfin::mesh::SubDomain::mark<std::size_t>,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true)
      .def("mark", &dolfin::mesh::SubDomain::mark<bool>,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true)
      .def("mark", &dolfin::mesh::SubDomain::mark<int>, py::arg("meshfunction"),
           py::arg("marker"), py::arg("check_midpoint") = true)
      .def("mark", &dolfin::mesh::SubDomain::mark<double>,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true);

  // dolfin::mesh::PeriodicBoundaryComputation
  py::class_<dolfin::mesh::PeriodicBoundaryComputation>(
      m, "PeriodicBoundaryComputation")
      .def_static(
          "compute_periodic_pairs",
          &dolfin::mesh::PeriodicBoundaryComputation::compute_periodic_pairs)
      .def_static("masters_slaves",
                  &dolfin::mesh::PeriodicBoundaryComputation::masters_slaves);
} // namespace dolfin_wrappers
} // namespace dolfin_wrappers
