// Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "array.h"
#include "caster_mpi.h"
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
// #include <nanobind/eval.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

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

using PythonPartitioningFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, int,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

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
void declare_meshtags(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("MeshTags_") + type;
  nb::class_<dolfinx::mesh::MeshTags<T>>(m, pyclass_name.c_str(),
                                         "MeshTags object")
      .def("__init__",
           [](dolfinx::mesh::MeshTags<T>* self,
              std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
              const nb::ndarray<std::int32_t, nb::numpy>& indices,
              const nb::ndarray<T, nb::numpy>& values)
           {
             std::vector<std::int32_t> indices_vec(
                 indices.data(), indices.data() + indices.size());
             std::vector<T> values_vec(values.data(),
                                       values.data() + values.size());
             new (self) dolfinx::mesh::MeshTags<T>(
                 topology, dim, std::move(indices_vec), std::move(values_vec));
           })
      .def_prop_ro("dtype", [](const dolfinx::mesh::MeshTags<T>& self)
                   { return nb::dtype<T>(); })
      .def_rw("name", &dolfinx::mesh::MeshTags<T>::name)
      .def_prop_ro("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_prop_ro("topology", &dolfinx::mesh::MeshTags<T>::topology)
      .def_prop_ro("values",
                   [](dolfinx::mesh::MeshTags<T>& self)
                   {
                     const std::size_t size = self.values().size();
                     return nb::ndarray<const T>(self.values().data(), 1,
                                                 &size);
                   })
      .def_prop_ro("indices",
                   [](dolfinx::mesh::MeshTags<T>& self)
                   {
                     const std::size_t size = self.indices().size();
                     return nb::ndarray<const std::int32_t>(
                         self.indices().data(), 1, &size);
                   })
      .def("find", [](dolfinx::mesh::MeshTags<T>& self, T value)
           { return as_nbarray(self.find(value)); });

  m.def("create_meshtags",
        [](std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           const nb::ndarray<T, nb::numpy>& values)
        {
          return dolfinx::mesh::create_meshtags(
              topology, dim, entities, std::span(values.data(), values.size()));
        });
}

template <typename T>
void declare_mesh(nb::module_& m, std::string type)
{
  std::string pyclass_geometry_name = std::string("Geometry_") + type;
  nb::class_<dolfinx::mesh::Geometry<T>>(m, pyclass_geometry_name.c_str(),
                                         "Geometry object")
      .def_prop_ro("dim", &dolfinx::mesh::Geometry<T>::dim,
                   "Geometric dimension")
      .def_prop_ro("dofmap",
                   [](dolfinx::mesh::Geometry<T>& self)
                   {
                     auto dofs = self.dofmap();
                     std::array shape{dofs.extent(0), dofs.extent(1)};
                     return nb::ndarray<const std::int32_t>(dofs.data_handle(),
                                                            2, shape.data());
                   })
      .def("index_map", &dolfinx::mesh::Geometry<T>::index_map)
      .def_prop_ro(
          "x",
          [](const dolfinx::mesh::Geometry<T>& self)
          {
            std::array<std::size_t, 2> shape{self.x().size() / 3, 3};
            return nb::ndarray<const T>(self.x().data(), 2, shape.data());
          },
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_prop_ro("cmaps", &dolfinx::mesh::Geometry<T>::cmaps,
                   "The coordinate maps")
      .def_prop_ro("input_global_indices",
                   &dolfinx::mesh::Geometry<T>::input_global_indices);

  std::string pyclass_mesh_name = std::string("Mesh_") + type;
  nb::class_<dolfinx::mesh::Mesh<T>>(m, pyclass_mesh_name.c_str(),
                                     nb::dynamic_attr(), "Mesh object")
      .def(
          "__init__",
          [](dolfinx::mesh::Mesh<T>* mesh, const MPICommWrapper comm,
             std::shared_ptr<dolfinx::mesh::Topology> topology,
             dolfinx::mesh::Geometry<T>& geometry) {
            new (mesh) dolfinx::mesh::Mesh<T>(comm.get(), topology, geometry);
          },
          nb::arg("comm"), nb::arg("topology"), nb::arg("geometry"))
      .def_prop_ro("geometry",
                   nb::overload_cast<>(&dolfinx::mesh::Mesh<T>::geometry),
                   "Mesh geometry")
      .def_prop_ro("topology",
                   nb::overload_cast<>(&dolfinx::mesh::Mesh<T>::topology),
                   "Mesh topology")
      .def_prop_ro("comm", [](dolfinx::mesh::Mesh<T>& self)
                   { return MPICommWrapper(self.comm()); })
      .def_rw("name", &dolfinx::mesh::Mesh<T>::name);

  m.def(
      std::string("create_interval_" + type).c_str(),
      [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::mesh::create_interval<T>(
            comm.get(), n, p, create_cell_partitioner_cpp(partitioner));
      },
      nb::arg("comm"), nb::arg("n"), nb::arg("p"), nb::arg("ghost_mode"),
      nb::arg("partitioner"));
  m.def(
      std::string("create_rectangle_" + type).c_str(),
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 2>, 2>& p,
         std::array<std::size_t, 2> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& partitioner,
         dolfinx::mesh::DiagonalType diagonal)
      {
        return dolfinx::mesh::create_rectangle<T>(
            comm.get(), p, n, celltype,
            create_cell_partitioner_cpp(partitioner), diagonal);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner"), nb::arg("diagonal"));
  m.def(
      std::string("create_box_" + type).c_str(),
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 3>, 2>& p,
         std::array<std::size_t, 3> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::mesh::create_box<T>(
            comm.get(), p, n, celltype,
            create_cell_partitioner_cpp(partitioner));
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner"));
  m.def(
      "create_mesh",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         const dolfinx::fem::CoordinateElement<T>& element,
         const nb::ndarray<T, nb::numpy>& x,
         const PythonPartitioningFunction& p)
      {
        auto p_wrap
            = [p](MPI_Comm comm, int n, int tdim,
                  const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
        { return p(MPICommWrapper(comm), n, tdim, cells); };

        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);
        std::vector shape{std::size_t(x.shape(0)), shape1};

        return dolfinx::mesh::create_mesh(
            comm.get(), cells, {element}, std::span(x.data(), x.size()),
            {static_cast<std::size_t>(x.shape(0)),
             static_cast<std::size_t>(x.shape(1))},
            p_wrap);
      },
      nb::arg("comm"), nb::arg("cells"), nb::arg("element"),
      nb::arg("x").noconvert(), nb::arg("partitioner"),
      "Helper function for creating meshes.");
  m.def(
      "create_submesh",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         const nb::ndarray<std::int32_t, nb::numpy>& entities)
      {
        return dolfinx::mesh::create_submesh(
            mesh, dim, std::span(entities.data(), entities.size()));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));

  m.def(
      "cell_normals",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         const nb::ndarray<std::int32_t, nb::numpy>& entities)
      {
        std::vector<T> n = dolfinx::mesh::cell_normals(
            mesh, dim, std::span(entities.data(), entities.size()));
        return as_nbarray(std::move(n),
                          std::array<std::size_t, 2>{n.size() / 3, 3});
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));
  m.def(
      "h",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         const nb::ndarray<std::int32_t, nb::numpy>& entities)
      {
        return as_nbarray(dolfinx::mesh::h(
            mesh, std::span(entities.data(), entities.size()), dim));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"),
      "Compute maximum distsance between any two vertices.");
  m.def(
      "compute_midpoints",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<std::int32_t, nb::numpy> entities)
      {
        std::vector<T> x = dolfinx::mesh::compute_midpoints(
            mesh, dim, std::span(entities.data(), entities.size()));
        std::array<std::size_t, 2> shape{(std::size_t)entities.size(), 3};
        return as_nbarray(std::move(x), shape);
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));

  m.def(
      "locate_entities",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         const std::function<nb::ndarray<bool>(
             const nb::ndarray<const T, nb::numpy>&)>& marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          std::array shape{x.extent(0), x.extent(1)};
          nb::ndarray<const T, nb::numpy> x_view(x.data_handle(), 2,
                                                 shape.data());
          nb::ndarray<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return as_nbarray(
            dolfinx::mesh::locate_entities(mesh, dim, cpp_marker));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("marker"));

  m.def(
      "locate_entities_boundary",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         const std::function<nb::ndarray<bool>(
             const nb::ndarray<const T, nb::numpy>&)>& marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          std::array shape{x.extent(0), x.extent(1)};
          nb::ndarray<const T, nb::numpy> x_view(x.data_handle(), 2,
                                                 shape.data(), nb::none());
          nb::ndarray<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };
        return as_nbarray(
            dolfinx::mesh::locate_entities_boundary(mesh, dim, cpp_marker));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("marker"));

  m.def(
      "entities_to_geometry",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<std::int32_t, nb::numpy> entities, bool orient)
      {
        std::vector<std::int32_t> idx = dolfinx::mesh::entities_to_geometry(
            mesh, dim, std::span(entities.data(), entities.size()), orient);

        auto topology = mesh.topology();
        assert(topology);
        if (topology->cell_types().size() > 1)
          throw std::runtime_error("Multiple cell type not supported.");
        dolfinx::mesh::CellType cell_type = topology->cell_types()[0];
        std::size_t num_vertices = dolfinx::mesh::num_cell_vertices(
            cell_entity_type(cell_type, dim, 0));
        std::array<std::size_t, 2> shape{(std::size_t)entities.size(),
                                         num_vertices};
        return as_nbarray(std::move(idx), shape);
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"), nb::arg("orient"));
}

void mesh(nb::module_& m)
{
  nb::enum_<dolfinx::mesh::CellType>(m, "CellType")
      .value("point", dolfinx::mesh::CellType::point)
      .value("interval", dolfinx::mesh::CellType::interval)
      .value("triangle", dolfinx::mesh::CellType::triangle)
      .value("quadrilateral", dolfinx::mesh::CellType::quadrilateral)
      .value("tetrahedron", dolfinx::mesh::CellType::tetrahedron)
      .value("pyramid", dolfinx::mesh::CellType::pyramid)
      .value("prism", dolfinx::mesh::CellType::prism)
      .value("hexahedron", dolfinx::mesh::CellType::hexahedron)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  m.def("to_type", &dolfinx::mesh::to_type, nb::arg("cell"));
  m.def("to_string", &dolfinx::mesh::to_string, nb::arg("type"));
  m.def("is_simplex", &dolfinx::mesh::is_simplex, nb::arg("type"));

  m.def("cell_entity_type", &dolfinx::mesh::cell_entity_type, nb::arg("type"),
        nb::arg("dim"), nb::arg("index"));
  m.def("cell_dim", &dolfinx::mesh::cell_dim, nb::arg("type"));
  m.def("cell_num_entities", &dolfinx::mesh::cell_num_entities, nb::arg("type"),
        nb::arg("dim"));
  m.def("cell_num_vertices", &dolfinx::mesh::num_cell_vertices,
        nb::arg("type"));
  m.def("get_entity_vertices", &dolfinx::mesh::get_entity_vertices,
        nb::arg("type"), nb::arg("dim"));
  m.def("extract_topology", &dolfinx::mesh::extract_topology,
        nb::arg("cell_type"), nb::arg("layout"), nb::arg("cells"));

  m.def(
      "build_dual_graph",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells, int tdim)
      { return dolfinx::mesh::build_dual_graph(comm.get(), cells, tdim); },
      nb::arg("comm"), nb::arg("cells"), nb::arg("tdim"),
      "Build dual graph for cells");

  // dolfinx::mesh::GhostMode enums
  nb::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfinx::mesh::GhostMode::shared_vertex);

  // dolfinx::mesh::TopologyComputation
  m.def(
      "compute_entities",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         int dim)
      { return dolfinx::mesh::compute_entities(comm.get(), topology, dim); },
      nb::arg("comm"), nb::arg("topology"), nb::arg("dim"));
  m.def("compute_connectivity", &dolfinx::mesh::compute_connectivity,
        nb::arg("topology"), nb::arg("d0"), nb::arg("d1"));

  // dolfinx::mesh::Topology class
  nb::class_<dolfinx::mesh::Topology>(m, "Topology", nb::dynamic_attr(),
                                      "Topology object")
      .def(
          "__init__",
          [](dolfinx::mesh::Topology* t, const MPICommWrapper comm,
             const std::vector<dolfinx::mesh::CellType> cell_type)
          { new (t) dolfinx::mesh::Topology(comm.get(), cell_type); },
          nb::arg("comm"), nb::arg("cell_type"))
      .def("entity_group_offsets",
           &dolfinx::mesh::Topology::entity_group_offsets)
      .def("set_connectivity", &dolfinx::mesh::Topology::set_connectivity,
           nb::arg("c"), nb::arg("d0"), nb::arg("d1"))
      .def("set_index_map", &dolfinx::mesh::Topology::set_index_map,
           nb::arg("dim"), nb::arg("map"))
      .def("create_entities", &dolfinx::mesh::Topology::create_entities,
           nb::arg("dim"))
      .def("create_entity_permutations",
           &dolfinx::mesh::Topology::create_entity_permutations)
      .def("create_connectivity", &dolfinx::mesh::Topology::create_connectivity,
           nb::arg("d0"), nb::arg("d1"))
      .def("get_facet_permutations",
           [](const dolfinx::mesh::Topology& self)
           {
             const std::vector<std::uint8_t>& p = self.get_facet_permutations();
             const std::size_t size = p.size();
             return nb::ndarray<const std::uint8_t>(p.data(), 1, &size);
           })
      .def("get_cell_permutation_info",
           [](const dolfinx::mesh::Topology& self)
           {
             const std::vector<std::uint32_t>& p
                 = self.get_cell_permutation_info();
             const std::size_t size = p.size();
             return nb::ndarray<const std::uint32_t>(p.data(), 1, &size);
           })
      .def_prop_ro("dim", &dolfinx::mesh::Topology::dim,
                   "Topological dimension")
      .def_prop_ro("original_cell_index",
                   [](const dolfinx::mesh::Topology& self)
                   {
                     const std::size_t size = self.original_cell_index.size();
                     return nb::ndarray<const std::int64_t>(
                         self.original_cell_index.data(), 1, &size);
                   })
      .def("connectivity",
           nb::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       nb::const_),
           nb::arg("d0"), nb::arg("d1"))
      .def("index_map", &dolfinx::mesh::Topology::index_map, nb::arg("dim"))
      .def_prop_ro("cell_types", &dolfinx::mesh::Topology::cell_types)
      .def("cell_name",
           [](const dolfinx::mesh::Topology& self)
           {
             if (self.cell_types().size() > 1)
               throw std::runtime_error("Multiple cell types not supported");
             return dolfinx::mesh::to_string(self.cell_types()[0]);
           })
      .def("interprocess_facets", &dolfinx::mesh::Topology::interprocess_facets)
      .def_prop_ro("comm", [](dolfinx::mesh::Topology& self)
                   { return MPICommWrapper(self.comm()); });

  // dolfinx::mesh::MeshTags

  declare_meshtags<std::int8_t>(m, "int8");
  declare_meshtags<std::int32_t>(m, "int32");
  declare_meshtags<std::int64_t>(m, "int64");
  declare_meshtags<double>(m, "float64");

  declare_mesh<float>(m, "float32");
  declare_mesh<double>(m, "float64");

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
             bool ghosting)>& part,
         dolfinx::mesh::GhostMode ghost_mode) -> PythonCellPartitionFunction
      {
        return create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(
                ghost_mode, create_cell_partitioner_cpp(part)));
      },
      nb::arg("part"), nb::arg("ghost_mode") = dolfinx::mesh::GhostMode::none);

  m.def(
      "exterior_facet_indices",
      [](const dolfinx::mesh::Topology& t)
      { return as_nbarray(dolfinx::mesh::exterior_facet_indices(t)); },
      nb::arg("topology"));
  m.def(
      "compute_incident_entities",
      [](const dolfinx::mesh::Topology& topology,
         nb::ndarray<std::int32_t, nb::numpy> entities, int d0, int d1)
      {
        return as_nbarray(dolfinx::mesh::compute_incident_entities(
            topology, std::span(entities.data(), entities.size()), d0, d1));
      },
      nb::arg("mesh"), nb::arg("entities"), nb::arg("d0"), nb::arg("d1"));

  // Mesh generation
  nb::enum_<dolfinx::mesh::DiagonalType>(m, "DiagonalType")
      .value("left", dolfinx::mesh::DiagonalType::left)
      .value("right", dolfinx::mesh::DiagonalType::right)
      .value("crossed", dolfinx::mesh::DiagonalType::crossed)
      .value("left_right", dolfinx::mesh::DiagonalType::left_right)
      .value("right_left", dolfinx::mesh::DiagonalType::right_left);
}
} // namespace dolfinx_wrappers
