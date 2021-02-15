// Copyright (C) 2017-2020 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <cfloat>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <dolfinx/mesh/utils.h>
#include <iostream>
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
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh, int dim,
             const py::array_t<std::int32_t, py::array::c_style>& indices,
             const py::array_t<T, py::array::c_style>& values) {
            std::vector<std::int32_t> indices_vec(
                indices.data(), indices.data() + indices.size());
            std::vector<T> values_vec(values.data(),
                                      values.data() + values.size());
            return dolfinx::mesh::MeshTags<T>(mesh, dim, std::move(indices_vec),
                                              std::move(values_vec));
          }))
      .def_readwrite("name", &dolfinx::mesh::MeshTags<T>::name)
      .def_property_readonly("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_property_readonly("mesh", &dolfinx::mesh::MeshTags<T>::mesh)
      .def("ufl_id", &dolfinx::mesh::MeshTags<T>::id)
      .def_property_readonly("values",
                             [](dolfinx::mesh::MeshTags<T>& self) {
                               return py::array_t<T>(self.values().size(),
                                                     self.values().data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("indices", [](dolfinx::mesh::MeshTags<T>& self) {
        return py::array_t<std::int32_t>(self.indices().size(),
                                         self.indices().data(), py::cast(self));
      });

  m.def("create_meshtags",
        [](const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh, int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           const py::array_t<T, py::array::c_style>& values) {
          return dolfinx::mesh::create_meshtags(
              mesh, dim, entities, tcb::span(values.data(), values.size()));
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
  m.def("cell_normals",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           const py::array_t<std::int32_t, py::array::c_style>& entities) {
          return as_pyarray2d(dolfinx::mesh::cell_normals(
              mesh, dim, tcb::span(entities.data(), entities.size())));
        });
  m.def("get_entity_vertices", &dolfinx::mesh::get_entity_vertices);

  m.def("extract_topology", &dolfinx::mesh::extract_topology);

  m.def(
      "h",
      [](const dolfinx::mesh::Mesh& mesh, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities) {
        return as_pyarray(dolfinx::mesh::h(
            mesh, tcb::span(entities.data(), entities.size()), dim));
      },
      "Compute maximum distance between any two vertices.");
  m.def("midpoints", &dolfinx::mesh::midpoints);

  m.def("midpoints",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           py::array_t<std::int32_t, py::array::c_style> entity_list) {
          return as_pyarray2d(dolfinx::mesh::midpoints(
              mesh, dim, tcb::span(entity_list.data(), entity_list.size())));
        });
  m.def("compute_boundary_facets", &dolfinx::mesh::compute_boundary_facets);

  using PythonPartitioningFunction
      = std::function<const dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int, const dolfinx::mesh::CellType,
          const dolfinx::graph::AdjacencyList<std::int64_t>&,
          dolfinx::mesh::GhostMode)>;

  m.def(
      "create_mesh",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         const dolfinx::fem::CoordinateElement& element,
         const Eigen::Ref<const Eigen::Array<
             double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x,
         dolfinx::mesh::GhostMode ghost_mode,
         PythonPartitioningFunction partitioner) {
        auto partitioner_wrapper
            = [partitioner](
                  MPI_Comm comm, int n, const dolfinx::mesh::CellType cell_type,
                  const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
                  dolfinx::mesh::GhostMode ghost_mode) {
                return partitioner(MPICommWrapper(comm), n, cell_type, cells,
                                   ghost_mode);
              };
        return dolfinx::mesh::create_mesh(comm.get(), cells, element, x,
                                          ghost_mode, partitioner_wrapper);
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
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::common::IndexMap>& map,
             const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap,
             const dolfinx::fem::CoordinateElement& element,
             const py::array_t<double, py::array::c_style>& x,
             const py::array_t<std::int64_t, py::array::c_style>&
                 global_indices) {
            std::vector<std::int64_t> indices(global_indices.data(),
                                              global_indices.data()
                                                  + global_indices.size());
            assert(x.ndim() <= 2);
            if (x.ndim() == 1)
            {
              dolfinx::common::array2d<double> _x(x.shape()[0], 1);
              std::copy(x.data(), x.data() + x.size(), _x.data());
              return dolfinx::mesh::Geometry(map, dofmap, element,
                                             std::move(_x), std::move(indices));
            }
            else
            {
              dolfinx::common::array2d<double> _x(x.shape()[0], x.shape()[1]);
              std::copy(x.data(), x.data() + x.size(), _x.data());
              return dolfinx::mesh::Geometry(map, dofmap, element,
                                             std::move(_x), std::move(indices));
            }
          }))
      .def_property_readonly("dim", &dolfinx::mesh::Geometry::dim,
                             "Geometric dimension")
      .def_property_readonly("dofmap", &dolfinx::mesh::Geometry::dofmap)
      .def("index_map", &dolfinx::mesh::Geometry::index_map)
      .def_property_readonly(
          "x",
          [](const dolfinx::mesh::Geometry& self) {
            const dolfinx::common::array2d<double>& x = self.x();
            return py::array_t<double>(x.shape, x.strides(), x.data(),
                                       py::cast(self));
          },
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_property_readonly("cmap", &dolfinx::mesh::Geometry::cmap,
                             "The coordinate map")
      .def_property_readonly("input_global_indices",
                             &dolfinx::mesh::Geometry::input_global_indices);

  // dolfinx::mesh::TopologyComputation
  m.def("compute_entities",
        [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
           int dim) {
          return dolfinx::mesh::compute_entities(comm.get(), topology, dim);
        });
  m.def("compute_connectivity", &dolfinx::mesh::compute_connectivity);

  // dolfinx::mesh::Topology class
  py::class_<dolfinx::mesh::Topology, std::shared_ptr<dolfinx::mesh::Topology>>(
      m, "Topology", "Topology object")
      .def(py::init([](const MPICommWrapper comm,
                       const dolfinx::mesh::CellType cell_type) {
        return dolfinx::mesh::Topology(comm.get(), cell_type);
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
           [](const dolfinx::mesh::Topology& self) {
             const std::vector<std::uint8_t>& p = self.get_facet_permutations();
             return py::array_t<std::uint8_t>(p.size(), p.data(),
                                              py::cast(self));
           })
      .def("get_cell_permutation_info",
           [](const dolfinx::mesh::Topology& self) {
             const std::vector<std::uint32_t>& p
                 = self.get_cell_permutation_info();
             return py::array_t<std::uint32_t>(p.size(), p.data(),
                                               py::cast(self));
           })
      .def_property_readonly("dim", &dolfinx::mesh::Topology::dim,
                             "Topological dimension")
      .def("connectivity",
           py::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       py::const_))
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
        return dolfinx::mesh::Mesh(comm.get(), topology, geometry);
      }))
      .def_property_readonly(
          "geometry", py::overload_cast<>(&dolfinx::mesh::Mesh::geometry),
          "Mesh geometry")
      .def("mpi_comm",
           [](dolfinx::mesh::Mesh& self) {
             return MPICommWrapper(self.mpi_comm());
           })
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
  m.def("partition_cells_graph",
        [](const MPICommWrapper comm, int nparts,
           dolfinx::mesh::CellType cell_type,
           const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
           dolfinx::mesh::GhostMode ghost_mode)
            -> dolfinx::graph::AdjacencyList<std::int32_t> {
          return dolfinx::mesh::partition_cells_graph(
              comm.get(), nparts, cell_type, cells, ghost_mode);
        });

  m.def("locate_entities",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           const std::function<py::array_t<bool>(
               const py::array_t<double, py::array::c_style>&)>& marker) {
          auto cpp_marker
              = [&marker](const dolfinx::common::array2d<double>& x) {
                  py::array_t<double> x_view(x.shape, x.strides(), x.data(),
                                             py::none());
                  py::array_t<bool> marked = marker(x_view);
                  return std::vector<bool>(marked.data(),
                                           marked.data() + marked.size());
                };
          return as_pyarray(
              dolfinx::mesh::locate_entities(mesh, dim, cpp_marker));
        });

  m.def("locate_entities_boundary",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           const std::function<py::array_t<bool>(
               const py::array_t<double, py::array::c_style>&)>& marker) {
          auto cpp_marker
              = [&marker](const dolfinx::common::array2d<double>& x) {
                  py::array_t<double> x_view(x.shape, x.strides(), x.data(),
                                             py::none());
                  py::array_t<bool> marked = marker(x_view);
                  return std::vector<bool>(marked.data(),
                                           marked.data() + marked.size());
                };
          return as_pyarray(
              dolfinx::mesh::locate_entities_boundary(mesh, dim, cpp_marker));
        });

  m.def("entities_to_geometry",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           py::array_t<std::int32_t, py::array::c_style> entity_list,
           bool orient) {
          return as_pyarray2d(dolfinx::mesh::entities_to_geometry(
              mesh, dim, tcb::span(entity_list.data(), entity_list.size()),
              orient));
        });
  m.def("exterior_facet_indices", &dolfinx::mesh::exterior_facet_indices);
}
} // namespace dolfinx_wrappers