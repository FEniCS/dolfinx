// Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <cfloat>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <dolfinx/mesh/utils.h>
#include <iostream>
#include <memory>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>

namespace py = pybind11;

namespace
{
/// Wrap a Python graph partitioning function as a C++ function
template <typename Functor>
auto create_cell_partitioner_cpp(Functor p_py)
{
  return [p_py](MPI_Comm comm, int nparts,
                const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                bool ghosting)
  {
    return p_py(dolfinx_wrappers::MPICommWrapper(comm), nparts, local_graph,
                ghosting);
  };
}

/// Wrap a C++ cell partitioning function as a Python function
template <typename Functor>
auto create_cell_partitioner_py(Functor p_cpp)
{
  return [p_cpp](dolfinx_wrappers::MPICommWrapper comm, int n, int tdim,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
  { return p_cpp(comm.get(), n, tdim, cells); };
}

using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, int,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, int,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& partitioner)
{
  return [partitioner](MPI_Comm comm, int n, int tdim,
                       const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
  {
    return partitioner(dolfinx_wrappers::MPICommWrapper(comm), n, tdim, cells);
  };
}

} // namespace

namespace dolfinx_wrappers
{

template <typename T>
void declare_meshtags(py::module& m, std::string type)
{
  std::string pyclass_name = std::string("MeshTags_") + type;
  py::class_<dolfinx::mesh::MeshTags<T>,
             std::shared_ptr<dolfinx::mesh::MeshTags<T>>>(
      m, pyclass_name.c_str(), "MeshTags object")
      .def(py::init(
          [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh, int dim,
             const py::array_t<std::int32_t, py::array::c_style>& indices,
             const py::array_t<T, py::array::c_style>& values)
          {
            std::vector<std::int32_t> indices_vec(
                indices.data(), indices.data() + indices.size());
            std::vector<T> values_vec(values.data(),
                                      values.data() + values.size());
            return dolfinx::mesh::MeshTags<T>(mesh, dim, std::move(indices_vec),
                                              std::move(values_vec));
          }))
      .def_property_readonly("dtype", [](const dolfinx::mesh::MeshTags<T>& self)
                             { return py::dtype::of<T>(); })
      .def_readwrite("name", &dolfinx::mesh::MeshTags<T>::name)
      .def_property_readonly("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_property_readonly("mesh", &dolfinx::mesh::MeshTags<T>::mesh)
      .def_property_readonly("values",
                             [](dolfinx::mesh::MeshTags<T>& self)
                             {
                               return py::array_t<T>(self.values().size(),
                                                     self.values().data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("indices",
                             [](dolfinx::mesh::MeshTags<T>& self)
                             {
                               return py::array_t<std::int32_t>(
                                   self.indices().size(), self.indices().data(),
                                   py::cast(self));
                             })
      .def("find", [](dolfinx::mesh::MeshTags<T>& self, T value)
           { return as_pyarray(self.find(value)); });

  m.def("create_meshtags",
        [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh, int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           const py::array_t<T, py::array::c_style>& values)
        {
          return dolfinx::mesh::create_meshtags(
              mesh, dim, entities, std::span(values.data(), values.size()));
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
      .value("pyramid", dolfinx::mesh::CellType::pyramid)
      .value("prism", dolfinx::mesh::CellType::prism)
      .value("hexahedron", dolfinx::mesh::CellType::hexahedron);

  m.def("to_type", &dolfinx::mesh::to_type, py::arg("cell"));
  m.def("to_string", &dolfinx::mesh::to_string, py::arg("type"));
  m.def("is_simplex", &dolfinx::mesh::is_simplex, py::arg("type"));

  m.def("cell_entity_type", &dolfinx::mesh::cell_entity_type, py::arg("type"),
        py::arg("dim"), py::arg("index"));
  m.def("cell_dim", &dolfinx::mesh::cell_dim, py::arg("type"));
  m.def("cell_num_entities", &dolfinx::mesh::cell_num_entities, py::arg("type"),
        py::arg("dim"));
  m.def("cell_num_vertices", &dolfinx::mesh::num_cell_vertices,
        py::arg("type"));
  m.def(
      "cell_normals",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities)
      {
        std::vector<double> n = dolfinx::mesh::cell_normals(
            mesh, dim, std::span(entities.data(), entities.size()));
        return as_pyarray(std::move(n),
                          std::array<std::size_t, 2>{n.size() / 3, 3});
      },
      py::arg("mesh"), py::arg("dim"), py::arg("entities"));
  m.def("get_entity_vertices", &dolfinx::mesh::get_entity_vertices,
        py::arg("type"), py::arg("dim"));
  m.def("extract_topology", &dolfinx::mesh::extract_topology,
        py::arg("cell_type"), py::arg("layout"), py::arg("cells"));

  m.def(
      "h",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities)
      {
        return as_pyarray(dolfinx::mesh::h(
            mesh, std::span(entities.data(), entities.size()), dim));
      },
      py::arg("mesh"), py::arg("dim"), py::arg("entities"),
      "Compute maximum distance between any two vertices.");

  m.def(
      "compute_midpoints",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         py::array_t<std::int32_t, py::array::c_style> entities)
      {
        std::vector<double> x = dolfinx::mesh::compute_midpoints(
            mesh, dim, std::span(entities.data(), entities.size()));
        std::array<std::size_t, 2> shape = {(std::size_t)entities.size(), 3};
        return as_pyarray(std::move(x), shape);
      },
      py::arg("mesh"), py::arg("dim"), py::arg("entities"));

  using PythonPartitioningFunction
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

  m.def(
      "build_dual_graph",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells, int tdim)
      { return dolfinx::mesh::build_dual_graph(comm.get(), cells, tdim); },
      py::arg("comm"), py::arg("cells"), py::arg("tdim"),
      "Build dual graph for cells");

  m.def(
      "create_mesh",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         const dolfinx::fem::CoordinateElement& element,
         const py::array_t<double, py::array::c_style>& x,
         const PythonPartitioningFunction& partitioner)
      {
        auto partitioner_wrapper
            = [partitioner](
                  MPI_Comm comm, int n, int tdim,
                  const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
        { return partitioner(MPICommWrapper(comm), n, tdim, cells); };

        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape()[1];
        std::vector shape{std::size_t(x.shape(0)), shape1};
        return dolfinx::mesh::create_mesh(
            comm.get(), cells, element, std::span(x.data(), x.size()),
            {static_cast<std::size_t>(x.shape(0)),
             static_cast<std::size_t>(x.shape(1))},
            partitioner_wrapper);
      },
      py::arg("comm"), py::arg("cells"), py::arg("element"), py::arg("x"),
      py::arg("partitioner"), "Helper function for creating meshes.");

  m.def(
      "create_submesh",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities)
      {
        return dolfinx::mesh::create_submesh(
            mesh, dim, std::span(entities.data(), entities.size()));
      },
      py::arg("mesh"), py::arg("dim"), py::arg("entities"));

  // dolfinx::mesh::GhostMode enums
  py::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfinx::mesh::GhostMode::shared_vertex);

  // dolfinx::mesh::Geometry class
  py::class_<dolfinx::mesh::Geometry, std::shared_ptr<dolfinx::mesh::Geometry>>(
      m, "Geometry", "Geometry object")
      .def_property_readonly("dim", &dolfinx::mesh::Geometry::dim,
                             "Geometric dimension")
      .def_property_readonly("dofmap", &dolfinx::mesh::Geometry::dofmap)
      .def("index_map", &dolfinx::mesh::Geometry::index_map)
      .def_property_readonly(
          "x",
          [](const dolfinx::mesh::Geometry& self)
          {
            std::array<std::size_t, 2> shape = {self.x().size() / 3, 3};
            return py::array_t<double>(shape, self.x().data(), py::cast(self));
          },
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_property_readonly("cmap", &dolfinx::mesh::Geometry::cmap,
                             "The coordinate map")
      .def_property_readonly("input_global_indices",
                             &dolfinx::mesh::Geometry::input_global_indices);

  // dolfinx::mesh::TopologyComputation
  m.def(
      "compute_entities",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         int dim)
      { return dolfinx::mesh::compute_entities(comm.get(), topology, dim); },
      py::arg("comm"), py::arg("topology"), py::arg("dim"));
  m.def("compute_connectivity", &dolfinx::mesh::compute_connectivity,
        py::arg("topology"), py::arg("d0"), py::arg("d1"));

  // dolfinx::mesh::Topology class
  py::class_<dolfinx::mesh::Topology, std::shared_ptr<dolfinx::mesh::Topology>>(
      m, "Topology", py::dynamic_attr(), "Topology object")
      .def(py::init([](const MPICommWrapper comm,
                       const dolfinx::mesh::CellType cell_type)
                    { return dolfinx::mesh::Topology(comm.get(), cell_type); }),
           py::arg("comm"), py::arg("cell_type"))
      .def("set_connectivity", &dolfinx::mesh::Topology::set_connectivity,
           py::arg("c"), py::arg("d0"), py::arg("d1"))
      .def("set_index_map", &dolfinx::mesh::Topology::set_index_map,
           py::arg("dim"), py::arg("map"))
      .def("create_entities", &dolfinx::mesh::Topology::create_entities,
           py::arg("dim"))
      .def("create_entity_permutations",
           &dolfinx::mesh::Topology::create_entity_permutations)
      .def("create_connectivity", &dolfinx::mesh::Topology::create_connectivity,
           py::arg("d0"), py::arg("d1"))
      .def("get_facet_permutations",
           [](const dolfinx::mesh::Topology& self)
           {
             const std::vector<std::uint8_t>& p = self.get_facet_permutations();
             return py::array_t<std::uint8_t>(p.size(), p.data(),
                                              py::cast(self));
           })
      .def("get_cell_permutation_info",
           [](const dolfinx::mesh::Topology& self)
           {
             const std::vector<std::uint32_t>& p
                 = self.get_cell_permutation_info();
             return py::array_t<std::uint32_t>(p.size(), p.data(),
                                               py::cast(self));
           })
      .def_property_readonly("dim", &dolfinx::mesh::Topology::dim,
                             "Topological dimension")
      .def_property_readonly("original_cell_index",
                             [](const dolfinx::mesh::Topology& self)
                             {
                               return py::array_t<std::int64_t>(
                                   self.original_cell_index.size(),
                                   self.original_cell_index.data(),
                                   py::cast(self));
                             })
      .def("connectivity",
           py::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       py::const_),
           py::arg("d0"), py::arg("d1"))
      .def("index_map", &dolfinx::mesh::Topology::index_map, py::arg("dim"))
      .def_property_readonly("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def("cell_name", [](const dolfinx::mesh::Topology& self)
           { return dolfinx::mesh::to_string(self.cell_type()); })
      .def("interprocess_facets", &dolfinx::mesh::Topology::interprocess_facets)
      .def_property_readonly("comm", [](dolfinx::mesh::Mesh& self)
                             { return MPICommWrapper(self.comm()); });

  // dolfinx::mesh::Mesh
  py::class_<dolfinx::mesh::Mesh, std::shared_ptr<dolfinx::mesh::Mesh>>(
      m, "Mesh", py::dynamic_attr(), "Mesh object")
      .def(py::init(
               [](const MPICommWrapper comm,
                  const dolfinx::mesh::Topology& topology,
                  dolfinx::mesh::Geometry& geometry)
               { return dolfinx::mesh::Mesh(comm.get(), topology, geometry); }),
           py::arg("comm"), py::arg("topology"), py::arg("geometry"))
      .def_property_readonly(
          "geometry", py::overload_cast<>(&dolfinx::mesh::Mesh::geometry),
          "Mesh geometry")
      .def_property_readonly(
          "topology", py::overload_cast<>(&dolfinx::mesh::Mesh::topology),
          "Mesh topology", py::return_value_policy::reference_internal)
      .def_property_readonly("comm", [](dolfinx::mesh::Mesh& self)
                             { return MPICommWrapper(self.comm()); })
      .def_readwrite("name", &dolfinx::mesh::Mesh::name);

  // dolfinx::mesh::MeshTags

  declare_meshtags<std::int8_t>(m, "int8");
  declare_meshtags<std::int32_t>(m, "int32");
  declare_meshtags<std::int64_t>(m, "int64");
  declare_meshtags<double>(m, "float64");

  // Partitioning interface using
  using PythonCellPartitionFunction
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&)>;
  m.def("create_cell_partitioner",
        [](dolfinx::mesh::GhostMode gm) -> PythonCellPartitionFunction
        {
          return create_cell_partitioner_py(
              dolfinx::mesh::create_cell_partitioner(gm));
        });
  m.def(
      "create_cell_partitioner",
      [](const std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
             MPICommWrapper comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             bool ghosting)>& part) -> PythonCellPartitionFunction
      {
        return create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(
                dolfinx::mesh::GhostMode::none,
                create_cell_partitioner_cpp(part)));
      },
      py::arg("part"));

  m.def(
      "locate_entities",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const std::function<py::array_t<bool>(
             const py::array_t<double, py::array::c_style>&)>& marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          std::array<std::size_t, 2> shape = {x.extent(0), x.extent(1)};
          py::array_t<double> x_view(shape, x.data_handle(), py::none());
          py::array_t<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return as_pyarray(
            dolfinx::mesh::locate_entities(mesh, dim, cpp_marker));
      },
      py::arg("mesh"), py::arg("dim"), py::arg("marker"));

  m.def(
      "locate_entities_boundary",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const std::function<py::array_t<bool>(
             const py::array_t<double, py::array::c_style>&)>& marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          std::array<std::size_t, 2> shape = {x.extent(0), x.extent(1)};
          py::array_t<double> x_view(shape, x.data_handle(), py::none());
          py::array_t<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };
        return as_pyarray(
            dolfinx::mesh::locate_entities_boundary(mesh, dim, cpp_marker));
      },
      py::arg("mesh"), py::arg("dim"), py::arg("marker"));

  m.def(
      "entities_to_geometry",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         py::array_t<std::int32_t, py::array::c_style> entities, bool orient)
      {
        std::vector<std::int32_t> idx = dolfinx::mesh::entities_to_geometry(
            mesh, dim, std::span(entities.data(), entities.size()), orient);
        dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
        std::size_t num_vertices = dolfinx::mesh::num_cell_vertices(
            cell_entity_type(cell_type, dim, 0));
        std::array<std::size_t, 2> shape
            = {(std::size_t)entities.size(), num_vertices};
        return as_pyarray(std::move(idx), shape);
      },
      py::arg("mesh"), py::arg("dim"), py::arg("entities"), py::arg("orient"));
  m.def(
      "exterior_facet_indices",
      [](const dolfinx::mesh::Topology& t)
      { return as_pyarray(dolfinx::mesh::exterior_facet_indices(t)); },
      py::arg("topology"));
  m.def(
      "compute_incident_entities",
      [](const dolfinx::mesh::Mesh& mesh,
         py::array_t<std::int32_t, py::array::c_style> entities, int d0, int d1)
      {
        return as_pyarray(dolfinx::mesh::compute_incident_entities(
            mesh, std::span(entities.data(), entities.size()), d0, d1));
      },
      py::arg("mesh"), py::arg("entities"), py::arg("d0"), py::arg("d1"));

  // Mesh generation
  py::enum_<dolfinx::mesh::DiagonalType>(m, "DiagonalType")
      .value("left", dolfinx::mesh::DiagonalType::left)
      .value("right", dolfinx::mesh::DiagonalType::right)
      .value("crossed", dolfinx::mesh::DiagonalType::crossed)
      .value("left_right", dolfinx::mesh::DiagonalType::left_right)
      .value("right_left", dolfinx::mesh::DiagonalType::right_left);

  m.def(
      "create_interval",
      [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::mesh::create_interval(
            comm.get(), n, p, create_cell_partitioner_cpp(partitioner));
      },
      py::arg("comm"), py::arg("n"), py::arg("p"), py::arg("ghost_mode"),
      py::arg("partitioner"));

  m.def(
      "create_rectangle",
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 2>, 2>& p,
         std::array<std::size_t, 2> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& partitioner,
         dolfinx::mesh::DiagonalType diagonal)
      {
        return dolfinx::mesh::create_rectangle(
            comm.get(), p, n, celltype,
            create_cell_partitioner_cpp(partitioner), diagonal);
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("celltype"),
      py::arg("partitioner"), py::arg("diagonal"));

  m.def(
      "create_box",
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 3>, 2>& p,
         std::array<std::size_t, 3> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::mesh::create_box(
            comm.get(), p, n, celltype,
            create_cell_partitioner_cpp(partitioner));
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("celltype"),
      py::arg("partitioner"));
}
} // namespace dolfinx_wrappers
