// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/common/Variable.h>
#include <dolfin/function/Expression.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshQuality.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/SubDomain.h>
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
  // Make dolfin::SubDomain from pointer
  m.def("make_dolfin_subdomain", [](std::uintptr_t e) {
    dolfin::SubDomain* p = reinterpret_cast<dolfin::SubDomain*>(e);
    return std::shared_ptr<const dolfin::SubDomain>(p);
  });

  // dolfin::CellType
  py::class_<dolfin::CellType> celltype(m, "CellType");

  // dolfin::CellType enums
  py::enum_<dolfin::CellType::Type>(celltype, "Type")
      .value("point", dolfin::CellType::Type::point)
      .value("interval", dolfin::CellType::Type::interval)
      .value("triangle", dolfin::CellType::Type::triangle)
      .value("quadrilateral", dolfin::CellType::Type::quadrilateral)
      .value("tetrahedron", dolfin::CellType::Type::tetrahedron)
      .value("hexahedron", dolfin::CellType::Type::hexahedron);

  celltype.def("type2string", &dolfin::CellType::type2string)
      .def("cell_type", &dolfin::CellType::cell_type)
      .def("description", &dolfin::CellType::description);

  // dolfin::MeshGeometry class
  py::class_<dolfin::MeshGeometry, std::shared_ptr<dolfin::MeshGeometry>>(
      m, "MeshGeometry", "DOLFIN MeshGeometry object")
      .def("dim", &dolfin::MeshGeometry::dim, "Geometrical dimension")
      .def("degree", &dolfin::MeshGeometry::degree, "Degree")
      .def("x", [](dolfin::MeshGeometry& self) {
        return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>(
            self.x().data(), self.num_points(), self.dim());
      });

  // dolfin::MeshTopology class
  py::class_<dolfin::MeshTopology, std::shared_ptr<dolfin::MeshTopology>,
             dolfin::Variable>(m, "MeshTopology", "DOLFIN MeshTopology object")
      .def("dim", &dolfin::MeshTopology::dim, "Topological dimension")
      .def("init",
           (void (dolfin::MeshTopology::*)(std::size_t))
               & dolfin::MeshTopology::init)
      .def("init",
           (void (dolfin::MeshTopology::*)(std::size_t, std::int32_t,
                                           std::int64_t))
               & dolfin::MeshTopology::init)
      .def("__call__",
           (const dolfin::MeshConnectivity& (
               dolfin::MeshTopology::*)(std::size_t, std::size_t) const)
               & dolfin::MeshTopology::operator(),
           py::return_value_policy::reference_internal)
      .def("size", &dolfin::MeshTopology::size)
      .def("hash", &dolfin::MeshTopology::hash)
      .def("init_global_indices", &dolfin::MeshTopology::init_global_indices)
      .def("have_global_indices", &dolfin::MeshTopology::have_global_indices)
      .def("ghost_offset", &dolfin::MeshTopology::ghost_offset)
      .def("cell_owner",
           (const std::vector<std::uint32_t>& (dolfin::MeshTopology::*)() const)
               & dolfin::MeshTopology::cell_owner)
      .def("set_global_index", &dolfin::MeshTopology::set_global_index)
      .def("global_indices",
           [](const dolfin::MeshTopology& self, int dim) {
             auto& indices = self.global_indices(dim);
             return py::array_t<std::int64_t>(indices.size(), indices.data());
           })
      .def("have_shared_entities", &dolfin::MeshTopology::have_shared_entities)
      .def(
          "shared_entities",
          (std::
               map<std::int32_t,
                   std::
                       set<std::
                               uint32_t>> & (dolfin::
                                                 MeshTopology::*)(std::
                                                                      uint32_t))
              & dolfin::MeshTopology::shared_entities)
      .def("str", &dolfin::MeshTopology::str);

  // dolfin::Mesh
  py::class_<dolfin::Mesh, std::shared_ptr<dolfin::Mesh>, dolfin::Variable>(
      m, "Mesh", py::dynamic_attr(), "DOLFIN Mesh object")
      .def(py::init<const dolfin::Mesh&>())
      .def(py::init([](const MPICommWrapper comm) {
        return std::unique_ptr<dolfin::Mesh>(new dolfin::Mesh(comm.get()));
      }))
      .def("bounding_box_tree", &dolfin::Mesh::bounding_box_tree)
      .def("cells",
           [](const dolfin::Mesh& self) {
             const std::uint32_t tdim = self.topology().dim();
             return py::array({(std::int32_t)self.topology().size(tdim),
                               (std::int32_t)self.type().num_vertices(tdim)},
                              self.topology()(tdim, 0)().data());
           })
      .def("geometry",
           (dolfin::MeshGeometry & (dolfin::Mesh::*)())
               & dolfin::Mesh::geometry,
           py::return_value_policy::reference, "Mesh geometry")
      .def("hash", &dolfin::Mesh::hash)
      .def("hmax", &dolfin::Mesh::hmax)
      .def("hmin", &dolfin::Mesh::hmin)
      .def("id", &dolfin::Mesh::id)
      .def("init_global", &dolfin::Mesh::init_global)
      .def("init", (void (dolfin::Mesh::*)() const) & dolfin::Mesh::init)
      .def("init",
           (std::size_t(dolfin::Mesh::*)(std::size_t) const)
               & dolfin::Mesh::init)
      .def("init",
           (void (dolfin::Mesh::*)(std::size_t, std::size_t) const)
               & dolfin::Mesh::init)
      .def("mpi_comm",
           [](dolfin::Mesh& self) { return MPICommWrapper(self.mpi_comm()); })
      .def("num_entities", &dolfin::Mesh::num_entities,
           "Number of mesh entities")
      .def("num_vertices", &dolfin::Mesh::num_vertices, "Number of vertices")
      .def("num_facets", &dolfin::Mesh::num_facets, "Number of facets")
      .def("num_cells", &dolfin::Mesh::num_cells, "Number of cells")
      .def("ordered", &dolfin::Mesh::ordered)
      .def("rmax", &dolfin::Mesh::rmax)
      .def("rmin", &dolfin::Mesh::rmin)
      .def("num_entities_global", &dolfin::Mesh::num_entities_global)
      .def("topology",
           (dolfin::MeshTopology & (dolfin::Mesh::*)())
               & dolfin::Mesh::topology,
           "Mesh topology", py::return_value_policy::reference_internal)
      .def("type",
           (const dolfin::CellType& (dolfin::Mesh::*)() const)
               & dolfin::Mesh::type,
           py::return_value_policy::reference)
      .def("ufl_id", [](const dolfin::Mesh& self) { return self.id(); })
      .def("cell_name", [](const dolfin::Mesh& self) {
        return dolfin::CellType::type2string(self.type().cell_type());
      });

  // dolfin::MeshConnectivity class
  py::class_<dolfin::MeshConnectivity,
             std::shared_ptr<dolfin::MeshConnectivity>>(
      m, "MeshConnectivity", "DOLFIN MeshConnectivity object")
      .def("__call__",
           [](const dolfin::MeshConnectivity& self, std::size_t i) {
             return Eigen::Map<const Eigen::Matrix<std::uint32_t,
                                                   Eigen::Dynamic, 1>>(
                 self(i), self.size(i));
           },
           py::return_value_policy::reference_internal)
      .def("size",
           (std::size_t(dolfin::MeshConnectivity::*)() const)
               & dolfin::MeshConnectivity::size)
      .def("size",
           (std::size_t(dolfin::MeshConnectivity::*)(std::size_t) const)
               & dolfin::MeshConnectivity::size);

  // dolfin::MeshEntity class
  py::class_<dolfin::MeshEntity, std::shared_ptr<dolfin::MeshEntity>>(
      m, "MeshEntity", "DOLFIN MeshEntity object")
      .def(py::init<const dolfin::Mesh&, std::size_t, std::size_t>())
      .def("dim", &dolfin::MeshEntity::dim, "Topological dimension")
      .def("mesh", &dolfin::MeshEntity::mesh, "Associated mesh")
      .def("index",
           (std::uint32_t (dolfin::MeshEntity::*)() const)
               & dolfin::MeshEntity::index, "Index")
      .def("global_index", &dolfin::MeshEntity::global_index, "Global index")
      .def("num_entities", &dolfin::MeshEntity::num_entities,
           "Number of incident entities of given dimension")
      .def("num_global_entities", &dolfin::MeshEntity::num_global_entities,
           "Global number of incident entities of given dimension")
      .def("entities",
           [](dolfin::MeshEntity& self, std::size_t dim) {
             return Eigen::Map<const Eigen::Matrix<std::uint32_t,
                                                   Eigen::Dynamic, 1>>(
                 self.entities(dim), self.num_entities(dim));
           })
      .def("midpoint", &dolfin::MeshEntity::midpoint, "Midpoint of Entity")
      .def("sharing_processes", &dolfin::MeshEntity::sharing_processes)
      .def("is_shared", &dolfin::MeshEntity::is_shared)
      .def("is_ghost", &dolfin::MeshEntity::is_ghost)
      .def("__str__", [](dolfin::MeshEntity& self) { return self.str(false); });

  // dolfin::Vertex
  py::class_<dolfin::Vertex, std::shared_ptr<dolfin::Vertex>,
             dolfin::MeshEntity>(m, "Vertex", "DOLFIN Vertex object")
      .def(py::init<const dolfin::Mesh&, std::size_t>())
      .def("point", &dolfin::Vertex::point);

  // dolfin::Edge
  py::class_<dolfin::Edge, std::shared_ptr<dolfin::Edge>, dolfin::MeshEntity>(
      m, "Edge", "DOLFIN Edge object")
      .def(py::init<const dolfin::Mesh&, std::size_t>())
      .def("dot", &dolfin::Edge::dot)
      .def("length", &dolfin::Edge::length);

  // dolfin::Face
  py::class_<dolfin::Face, std::shared_ptr<dolfin::Face>, dolfin::MeshEntity>(
      m, "Face", "DOLFIN Face object")
      .def(py::init<const dolfin::Mesh&, std::size_t>())
      .def("normal", &dolfin::Face::normal)
      .def("area", &dolfin::Face::area);

  // dolfin::Facet
  py::class_<dolfin::Facet, std::shared_ptr<dolfin::Facet>, dolfin::MeshEntity>(
      m, "Facet", "DOLFIN Facet object")
      .def(py::init<const dolfin::Mesh&, std::size_t>())
      .def("exterior", &dolfin::Facet::exterior)
      .def("normal", &dolfin::Facet::normal);

  // dolfin::Cell
  py::class_<dolfin::Cell, std::shared_ptr<dolfin::Cell>, dolfin::MeshEntity>(
      m, "Cell", "DOLFIN Cell object")
      .def(py::init<const dolfin::Mesh&, std::size_t>())
      .def("distance", &dolfin::Cell::distance)
      .def("facet_area", &dolfin::Cell::facet_area)
      .def("h", &dolfin::Cell::h)
      .def("inradius", &dolfin::Cell::inradius)
      .def("normal", &dolfin::Cell::normal)
      .def("circumradius", &dolfin::Cell::circumradius)
      .def("radius_ratio", &dolfin::Cell::radius_ratio)
      .def("volume", &dolfin::Cell::volume)
      .def("get_vertex_coordinates",
           [](const dolfin::Cell& self) {
             std::vector<double> x;
             self.get_vertex_coordinates(x);
             return x;
           },
           "Get cell vertex coordinates");

  py::class_<dolfin::MeshRange<dolfin::MeshEntity>, std::shared_ptr<dolfin::MeshRange<dolfin::MeshEntity>>>(
      m, "MeshEntities", "Range for iteration over entities of a Mesh")
      .def(py::init<const dolfin::Mesh&, int>())
      .def("__iter__", [](const dolfin::MeshRange<dolfin::MeshEntity>& r) {
        return py::make_iterator(r.begin(), r.end());
      });

  py::class_<dolfin::EntityRange<dolfin::MeshEntity>, std::shared_ptr<dolfin::EntityRange<dolfin::MeshEntity>>>(
      m, "EntityRange", "Range for iteration over entities of another entity")
      .def(py::init<const dolfin::MeshEntity&, int>())
      .def("__iter__", [](const dolfin::EntityRange<dolfin::MeshEntity>& r) {
        return py::make_iterator(r.begin(), r.end());
      });

// dolfin::MeshIterator (Cells, Facets, Faces, Edges, Vertices)
#define MESHITERATOR_MACRO(TYPE, ENTITYNAME)                                   \
  py::class_<dolfin::MeshRange<dolfin::ENTITYNAME>, std::shared_ptr<dolfin::MeshRange<dolfin::ENTITYNAME>>>( \
      m, #TYPE,                                                                \
      "Range for iterating over entities of type " #ENTITYNAME " of a Mesh")   \
      .def(py::init<const dolfin::Mesh&>())                                    \
      .def("__iter__", [](const dolfin::MeshRange<dolfin::ENTITYNAME>& c) {    \
        return py::make_iterator(c.begin(), c.end());                          \
      });

  MESHITERATOR_MACRO(Cells, Cell);
  MESHITERATOR_MACRO(Facets, Facet);
  MESHITERATOR_MACRO(Faces, Face);
  MESHITERATOR_MACRO(Edges, Edge);
  MESHITERATOR_MACRO(Vertices, Vertex);
#undef MESHITERATOR_MACRO

#define MESHENTITYITERATOR_MACRO(TYPE, ENTITYNAME)                             \
  py::class_<dolfin::EntityRange<dolfin::ENTITYNAME>, std::shared_ptr<dolfin::EntityRange<dolfin::ENTITYNAME>>>( \
      m, #TYPE, "Range for iterating over entities of type " #ENTITYNAME       \
                " incident to a MeshEntity")                                   \
      .def(py::init<const dolfin::MeshEntity&>())                              \
      .def("__iter__", [](const dolfin::EntityRange<dolfin::ENTITYNAME>& c) {        \
        return py::make_iterator(c.begin(), c.end());                          \
      });

  MESHENTITYITERATOR_MACRO(CellRange, Cell);
  MESHENTITYITERATOR_MACRO(FacetRange, Facet);
  MESHENTITYITERATOR_MACRO(FaceRange, Face);
  MESHENTITYITERATOR_MACRO(EdgeRange, Edge);
  MESHENTITYITERATOR_MACRO(VertexRange, Vertex);
#undef MESHENTITYITERATOR_MACRO

// dolfin::MeshFunction
#define MESHFUNCTION_MACRO(SCALAR, SCALAR_NAME)                                \
  py::class_<dolfin::MeshFunction<SCALAR>,                                     \
             std::shared_ptr<dolfin::MeshFunction<SCALAR>>, dolfin::Variable>( \
      m, "MeshFunction" #SCALAR_NAME, "DOLFIN MeshFunction object")            \
      .def(py::init(                                                           \
          [](std::shared_ptr<const dolfin::Mesh> mesh, std::size_t dim) {      \
            return dolfin::MeshFunction<SCALAR>(mesh, dim, 0);                 \
          }))                                                                  \
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::size_t,          \
                    SCALAR>())                                                 \
      .def(py::init<std::shared_ptr<const dolfin::Mesh>,                       \
                    const dolfin::MeshValueCollection<SCALAR>&>())             \
      .def(                                                                    \
          "__getitem__",                                                       \
          (const SCALAR& (dolfin::MeshFunction<SCALAR>::*)(std::size_t) const) \
              & dolfin::MeshFunction<SCALAR>::operator[])                      \
      .def("__setitem__",                                                      \
           [](dolfin::MeshFunction<SCALAR>& self, std::size_t index,           \
              SCALAR value) { self.operator[](index) = value; })               \
      .def("__getitem__",                                                      \
           (const SCALAR& (                                                    \
               dolfin::MeshFunction<SCALAR>::*)(const dolfin::MeshEntity&)     \
                const)                                                         \
               & dolfin::MeshFunction<SCALAR>::operator[])                     \
      .def("__setitem__",                                                      \
           [](dolfin::MeshFunction<SCALAR>& self,                              \
              const dolfin::MeshEntity& index,                                 \
              SCALAR value) { self.operator[](index) = value; })               \
      .def("__len__", &dolfin::MeshFunction<SCALAR>::size)                     \
      .def("dim", &dolfin::MeshFunction<SCALAR>::dim)                          \
      .def("size", &dolfin::MeshFunction<SCALAR>::size)                        \
      .def("id", &dolfin::MeshFunction<SCALAR>::id)                            \
      .def("ufl_id", &dolfin::MeshFunction<SCALAR>::id)                        \
      .def("mesh", &dolfin::MeshFunction<SCALAR>::mesh)                        \
      .def("set_values", &dolfin::MeshFunction<SCALAR>::set_values)            \
      .def("set_all", &dolfin::MeshFunction<SCALAR>::set_all)                  \
      .def("where_equal", &dolfin::MeshFunction<SCALAR>::where_equal)          \
      .def("array", [](dolfin::MeshFunction<SCALAR>& self) {                   \
        return Eigen::Map<Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>>(           \
            self.values(), self.size());                                       \
      })

  MESHFUNCTION_MACRO(bool, Bool);
  MESHFUNCTION_MACRO(int, Int);
  MESHFUNCTION_MACRO(double, Double);
  MESHFUNCTION_MACRO(std::size_t, Sizet);
#undef MESHFUNCTION_MACRO

// dolfin::MeshValueCollection
#define MESHVALUECOLLECTION_MACRO(SCALAR, SCALAR_NAME)                         \
  py::class_<dolfin::MeshValueCollection<SCALAR>,                              \
             std::shared_ptr<dolfin::MeshValueCollection<SCALAR>>,             \
             dolfin::Variable>(m, "MeshValueCollection_" #SCALAR_NAME,         \
                               "DOLFIN MeshValueCollection object")            \
      .def(py::init<std::shared_ptr<const dolfin::Mesh>>())                    \
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::size_t>())       \
      .def("dim", &dolfin::MeshValueCollection<SCALAR>::dim)                   \
      .def("size", &dolfin::MeshValueCollection<SCALAR>::size)                 \
      .def("get_value", &dolfin::MeshValueCollection<SCALAR>::get_value)       \
      .def("set_value",                                                        \
           (bool (dolfin::MeshValueCollection<SCALAR>::*)(std::size_t,         \
                                                          const SCALAR&))      \
               & dolfin::MeshValueCollection<SCALAR>::set_value)               \
      .def("set_value",                                                        \
           (bool (dolfin::MeshValueCollection<SCALAR>::*)(                     \
               std::size_t, std::size_t, const SCALAR&))                       \
               & dolfin::MeshValueCollection<SCALAR>::set_value)               \
      .def("values",                                                           \
           (std::map<std::pair<std::size_t, std::size_t>,                      \
                     SCALAR> & (dolfin::MeshValueCollection<SCALAR>::*)())     \
               & dolfin::MeshValueCollection<SCALAR>::values,                  \
           py::return_value_policy::reference)                                 \
      .def("assign",                                                           \
           [](dolfin::MeshValueCollection<SCALAR>& self,                       \
              const dolfin::MeshFunction<SCALAR>& mf) { self = mf; })          \
      .def("assign", [](dolfin::MeshValueCollection<SCALAR>& self,             \
                        const dolfin::MeshValueCollection<SCALAR>& other) {    \
        self = other;                                                          \
      })

  MESHVALUECOLLECTION_MACRO(bool, bool);
  MESHVALUECOLLECTION_MACRO(int, int);
  MESHVALUECOLLECTION_MACRO(double, double);
  MESHVALUECOLLECTION_MACRO(std::size_t, sizet);
#undef MESHVALUECOLLECTION_MACRO

  // dolfin::MeshEditor
  py::class_<dolfin::MeshEditor, std::shared_ptr<dolfin::MeshEditor>>(
      m, "MeshEditor", "DOLFIN MeshEditor object")
      .def(py::init<>())
      .def("open",
           (void (dolfin::MeshEditor::*)(dolfin::Mesh&, dolfin::CellType::Type,
                                         std::size_t, std::size_t, std::size_t))
               & dolfin::MeshEditor::open,
           py::arg("mesh"), py::arg("type"), py::arg("tdim"), py::arg("gdim"),
           py::arg("degree") = 1)
      .def("init_vertices_global", &dolfin::MeshEditor::init_vertices_global)
      .def("init_cells_global", &dolfin::MeshEditor::init_cells_global)
      .def("add_vertex",
           (void (dolfin::MeshEditor::*)(std::size_t, const dolfin::Point&))
               & dolfin::MeshEditor::add_vertex)
      .def("add_cell",
           (void (dolfin::MeshEditor::*)(std::size_t,
                                         const std::vector<std::size_t>&))
               & dolfin::MeshEditor::add_cell)
      .def("close", &dolfin::MeshEditor::close, py::arg("order") = true);

  // dolfin::MeshQuality
  py::class_<dolfin::MeshQuality>(m, "MeshQuality", "DOLFIN MeshQuality class")
      .def_static("radius_ratios", &dolfin::MeshQuality::radius_ratios)
      .def_static("radius_ratio_histogram_data",
                  &dolfin::MeshQuality::radius_ratio_histogram_data)
      .def_static("radius_ratio_min_max",
                  &dolfin::MeshQuality::radius_ratio_min_max)
      .def_static("radius_ratio_matplotlib_histogram",
                  &dolfin::MeshQuality::radius_ratio_matplotlib_histogram,
                  py::arg("mesh"), py::arg("num_bins") = 50)
      .def_static("dihedral_angles_min_max",
                  &dolfin::MeshQuality::dihedral_angles_min_max)
      .def_static("dihedral_angles_matplotlib_histogram",
                  &dolfin::MeshQuality::dihedral_angles_matplotlib_histogram);

  // dolfin::SubDomain trampoline class for user overloading from
  // Python
  class PySubDomain : public dolfin::SubDomain
  {
    using dolfin::SubDomain::SubDomain;

    bool inside(Eigen::Ref<const Eigen::VectorXd> x,
                bool on_boundary) const override
    {
      PYBIND11_OVERLOAD(bool, dolfin::SubDomain, inside, x, on_boundary);
    }

    void map(Eigen::Ref<const Eigen::VectorXd> x,
             Eigen::Ref<Eigen::VectorXd> y) const override
    {
      PYBIND11_OVERLOAD(void, dolfin::SubDomain, map, x, y);
    }
  };

  // dolfin::SubDomian
  py::class_<dolfin::SubDomain, std::shared_ptr<dolfin::SubDomain>,
             PySubDomain>(m, "SubDomain", "DOLFIN SubDomain object")
      .def(py::init<double>(), py::arg("map_tol") = DOLFIN_EPS)
      .def("inside",
           (bool (dolfin::SubDomain::*)(Eigen::Ref<const Eigen::VectorXd>, bool)
                const)
               & dolfin::SubDomain::inside)
      .def("map",
           (void (dolfin::SubDomain::*)(Eigen::Ref<const Eigen::VectorXd>,
                                        Eigen::Ref<Eigen::VectorXd>) const)
               & dolfin::SubDomain::map)
      .def("set_property", &dolfin::SubDomain::set_property)
      .def("get_property", &dolfin::SubDomain::get_property)
      .def("mark",
           (void (dolfin::SubDomain::*)(dolfin::MeshFunction<std::size_t>&,
                                        std::size_t, bool) const)
               & dolfin::SubDomain::mark,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true)
      .def("mark",
           (void (dolfin::SubDomain::*)(dolfin::MeshFunction<double>&, double,
                                        bool) const)
               & dolfin::SubDomain::mark,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true)
      .def("mark",
           (void (dolfin::SubDomain::*)(dolfin::MeshFunction<bool>&, bool, bool)
                const)
               & dolfin::SubDomain::mark,
           py::arg("meshfunction"), py::arg("marker"),
           py::arg("check_midpoint") = true);

  // dolfin::PeriodicBoundaryComputation
  py::class_<dolfin::PeriodicBoundaryComputation>(m,
                                                  "PeriodicBoundaryComputation")
      .def_static("compute_periodic_pairs",
                  &dolfin::PeriodicBoundaryComputation::compute_periodic_pairs)
      .def_static("masters_slaves",
                  &dolfin::PeriodicBoundaryComputation::masters_slaves);
}
}
