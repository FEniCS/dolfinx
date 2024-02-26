// Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "array.h"
#include "caster_mpi.h"
#include "numpy_dtype.h"
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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

namespace
{
/// Wrap a Python graph partitioning function as a C++ function
template <typename Functor>
auto create_partitioner_cpp(Functor p)
{
  return [p](MPI_Comm comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             bool ghosting)
  {
    return p(dolfinx_wrappers::MPICommWrapper(comm), nparts, local_graph,
             ghosting);
  };
}

/// Wrap a C++ cell partitioning function as a Python function
template <typename Functor>
auto create_cell_partitioner_py(Functor p)
{
  return [p](dolfinx_wrappers::MPICommWrapper comm, int n,
             dolfinx::mesh::CellType cell_type,
             const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
  { return p(comm.get(), n, cell_type, cells); };
}

using PythonPartitioningFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, dolfinx::mesh::CellType,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, dolfinx::mesh::CellType,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, dolfinx::mesh::CellType,
        const dolfinx::graph::AdjacencyList<std::int64_t>&)>;

/// Wrap a Python cell graph partitioning function as a C++ function
CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& p)
{
  if (p)
  {
    return [p](MPI_Comm comm, int n, dolfinx::mesh::CellType cell_type,
               const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
    { return p(dolfinx_wrappers::MPICommWrapper(comm), n, cell_type, cells); };
  }
  else
    return nullptr;
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
      .def(
          "__init__",
          [](dolfinx::mesh::MeshTags<T>* self,
             std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indices,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> values)
          {
            std::vector<std::int32_t> indices_vec(
                indices.data(), indices.data() + indices.size());
            std::vector<T> values_vec(values.data(),
                                      values.data() + values.size());
            new (self) dolfinx::mesh::MeshTags<T>(
                topology, dim, std::move(indices_vec), std::move(values_vec));
          })
      .def_prop_ro("dtype", [](const dolfinx::mesh::MeshTags<T>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_rw("name", &dolfinx::mesh::MeshTags<T>::name)
      .def_prop_ro("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_prop_ro("topology", &dolfinx::mesh::MeshTags<T>::topology)
      .def_prop_ro(
          "values",
          [](dolfinx::mesh::MeshTags<T>& self)
          {
            return nb::ndarray<const T, nb::numpy>(self.values().data(),
                                                   {self.values().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indices",
          [](dolfinx::mesh::MeshTags<T>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.indices().data(), {self.indices().size()});
          },
          nb::rv_policy::reference_internal)
      .def("find", [](dolfinx::mesh::MeshTags<T>& self, T value)
           { return as_nbarray(self.find(value)); });

  m.def("create_meshtags",
        [](std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           nb::ndarray<const T, nb::ndim<1>, nb::c_contig> values)
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
      .def_prop_ro(
          "dofmap",
          [](dolfinx::mesh::Geometry<T>& self)
          {
            auto dofs = self.dofmap();
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data_handle(), {dofs.extent(0), dofs.extent(1)});
          },
          nb::rv_policy::reference_internal)
      .def(
          "dofmaps",
          [](dolfinx::mesh::Geometry<T>& self, int i)
          {
            auto dofs = self.dofmap(i);
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data_handle(), {dofs.extent(0), dofs.extent(1)});
          },
          nb::rv_policy::reference_internal, nb::arg("i"),
          "Get the geometry dofmap associated with coordinate element i (mixed "
          "topology)")
      .def("index_map", &dolfinx::mesh::Geometry<T>::index_map)
      .def_prop_ro(
          "x",
          [](dolfinx::mesh::Geometry<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(self.x().data(),
                                             {self.x().size() / 3, 3});
          },
          nb::rv_policy::reference_internal,
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_prop_ro(
          "cmap", [](dolfinx::mesh::Geometry<T>& self) { return self.cmap(); },
          "The coordinate map")
      .def_prop_ro("input_global_indices",
                   &dolfinx::mesh::Geometry<T>::input_global_indices);

  std::string pyclass_mesh_name = std::string("Mesh_") + type;
  nb::class_<dolfinx::mesh::Mesh<T>>(m, pyclass_mesh_name.c_str(),
                                     nb::dynamic_attr(), "Mesh object")
      .def(
          "__init__",
          [](dolfinx::mesh::Mesh<T>* mesh, MPICommWrapper comm,
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
      .def_prop_ro(
          "comm",
          [](dolfinx::mesh::Mesh<T>& self)
          { return MPICommWrapper(self.comm()); },
          nb::keep_alive<0, 1>())
      .def_rw("name", &dolfinx::mesh::Mesh<T>::name);

  std::string create_interval("create_interval_" + type);
  m.def(
      create_interval.c_str(),
      [](MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& part)
      {
        return dolfinx::mesh::create_interval<T>(
            comm.get(), n, p, create_cell_partitioner_cpp(part));
      },
      nb::arg("comm"), nb::arg("n"), nb::arg("p"), nb::arg("ghost_mode"),
      nb::arg("partitioner").none());

  std::string create_rectangle("create_rectangle_" + type);
  m.def(
      create_rectangle.c_str(),
      [](MPICommWrapper comm, std::array<std::array<double, 2>, 2> p,
         std::array<std::size_t, 2> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& part,
         dolfinx::mesh::DiagonalType diagonal)
      {
        return dolfinx::mesh::create_rectangle<T>(
            comm.get(), p, n, celltype, create_cell_partitioner_cpp(part),
            diagonal);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none(), nb::arg("diagonal"));

  std::string create_box("create_box_" + type);
  m.def(
      create_box.c_str(),
      [](MPICommWrapper comm, std::array<std::array<double, 3>, 2> p,
         std::array<std::size_t, 3> n, dolfinx::mesh::CellType celltype,
         const PythonCellPartitionFunction& part)
      {
        MPI_Comm _comm = comm.get();
        return dolfinx::mesh::create_box<T>(_comm, _comm, p, n, celltype,
                                            create_cell_partitioner_cpp(part));
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none());
  m.def(
      "create_mesh",
      [](MPICommWrapper comm,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> cells,
         const dolfinx::fem::CoordinateElement<T>& element,
         nb::ndarray<const T, nb::c_contig> x,
         const PythonPartitioningFunction& p)
      {
        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);
        if (p)
        {
          auto p_wrap
              = [p](MPI_Comm comm, int n, dolfinx::mesh::CellType cell_type,
                    const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
          { return p(MPICommWrapper(comm), n, cell_type, cells); };
          return dolfinx::mesh::create_mesh(
              comm.get(), comm.get(), std::span(cells.data(), cells.size()),
              element, comm.get(), std::span(x.data(), x.size()),
              {x.shape(0), shape1}, p_wrap);
        }
        else
        {
          return dolfinx::mesh::create_mesh(
              comm.get(), comm.get(), std::span(cells.data(), cells.size()),
              element, comm.get(), std::span(x.data(), x.size()),
              {x.shape(0), shape1}, p);
        }
      },
      nb::arg("comm"), nb::arg("cells"), nb::arg("element"),
      nb::arg("x").noconvert(), nb::arg("partitioner").none(),
      "Helper function for creating meshes.");
  m.def(
      "create_submesh",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        return dolfinx::mesh::create_submesh(
            mesh, dim, std::span(entities.data(), entities.size()));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));

  m.def(
      "cell_normals",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        std::vector<T> n = dolfinx::mesh::cell_normals(
            mesh, dim, std::span(entities.data(), entities.size()));
        return as_nbarray(std::move(n), {n.size() / 3, 3});
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));
  m.def(
      "h",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        return as_nbarray(dolfinx::mesh::h(
            mesh, std::span(entities.data(), entities.size()), dim));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"),
      "Compute maximum distsance between any two vertices.");
  m.def(
      "compute_midpoints",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        std::vector<T> x = dolfinx::mesh::compute_midpoints(
            mesh, dim, std::span(entities.data(), entities.size()));
        return as_nbarray(std::move(x), {entities.size(), 3});
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"));

  m.def(
      "locate_entities",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)});
          auto marked = marker(x_view);
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
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        auto cpp_marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)});
          auto marked = marker(x_view);
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
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         bool orient)
      {
        std::vector<std::int32_t> idx = dolfinx::mesh::entities_to_geometry(
            mesh, dim, std::span(entities.data(), entities.size()), orient);

        auto topology = mesh.topology();
        assert(topology);
        dolfinx::mesh::CellType cell_type = topology->cell_type();
        std::size_t num_vertices = dolfinx::mesh::num_cell_vertices(
            cell_entity_type(cell_type, dim, 0));
        return as_nbarray(std::move(idx), {entities.size(), num_vertices});
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
      [](const MPICommWrapper comm, dolfinx::mesh::CellType cell_type,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
      { return dolfinx::mesh::build_dual_graph(comm.get(), cell_type, cells); },
      nb::arg("comm"), nb::arg("cell_type"), nb::arg("cells"),
      "Build dual graph for cells");

  // dolfinx::mesh::GhostMode enums
  nb::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet)
      .value("shared_vertex", dolfinx::mesh::GhostMode::shared_vertex);

  // dolfinx::mesh::TopologyComputation
  m.def(
      "compute_entities",
      [](MPICommWrapper comm, const dolfinx::mesh::Topology& topology, int dim,
         int index) {
        return dolfinx::mesh::compute_entities(comm.get(), topology, dim,
                                               index);
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("dim"), nb::arg("index"));
  m.def("compute_connectivity", &dolfinx::mesh::compute_connectivity,
        nb::arg("topology"), nb::arg("d0"), nb::arg("d1"));

  // dolfinx::mesh::Topology class
  nb::class_<dolfinx::mesh::Topology>(m, "Topology", nb::dynamic_attr(),
                                      "Topology object")
      .def(
          "__init__",
          [](dolfinx::mesh::Topology* t, MPICommWrapper comm,
             dolfinx::mesh::CellType cell_type)
          { new (t) dolfinx::mesh::Topology(comm.get(), cell_type); },
          nb::arg("comm"), nb::arg("cell_type"))
      .def("set_connectivity",
           nb::overload_cast<
               std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
               int, int>(&dolfinx::mesh::Topology::set_connectivity),
           nb::arg("c"), nb::arg("d0"), nb::arg("d1"))
      .def("set_index_map",
           nb::overload_cast<int,
                             std::shared_ptr<const dolfinx::common::IndexMap>>(
               &dolfinx::mesh::Topology::set_index_map),
           nb::arg("dim"), nb::arg("map"))
      .def("create_entities", &dolfinx::mesh::Topology::create_entities,
           nb::arg("dim"))
      .def("create_entity_permutations",
           &dolfinx::mesh::Topology::create_entity_permutations)
      .def("create_connectivity", &dolfinx::mesh::Topology::create_connectivity,
           nb::arg("d0"), nb::arg("d1"))
      .def(
          "get_facet_permutations",
          [](const dolfinx::mesh::Topology& self)
          {
            const std::vector<std::uint8_t>& p = self.get_facet_permutations();
            return nb::ndarray<const std::uint8_t, nb::numpy>(p.data(),
                                                              {p.size()});
          },
          nb::rv_policy::reference_internal)
      .def(
          "get_cell_permutation_info",
          [](const dolfinx::mesh::Topology& self)
          {
            const std::vector<std::uint32_t>& p
                = self.get_cell_permutation_info();
            return nb::ndarray<const std::uint32_t, nb::numpy>(p.data(),
                                                               {p.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("dim", &dolfinx::mesh::Topology::dim,
                   "Topological dimension")
      .def_prop_ro(
          "original_cell_index",
          [](const dolfinx::mesh::Topology& self)
          {
            if (self.original_cell_index.size() != 1)
              throw std::runtime_error("Mixed topology unsupported");
            return nb::ndarray<const std::int64_t, nb::numpy>(
                self.original_cell_index[0].data(),
                {self.original_cell_index[0].size()});
          },
          nb::rv_policy::reference_internal)
      .def("connectivity",
           nb::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       nb::const_),
           nb::arg("d0"), nb::arg("d1"))
      .def("connectivity",
           nb::overload_cast<std::pair<std::int8_t, std::int8_t>,
                             std::pair<std::int8_t, std::int8_t>>(
               &dolfinx::mesh::Topology::connectivity, nb::const_),
           nb::arg("d0"), nb::arg("d1"))
      .def("index_map", &dolfinx::mesh::Topology::index_map, nb::arg("dim"))
      .def("index_maps", &dolfinx::mesh::Topology::index_maps, nb::arg("dim"))
      .def_prop_ro("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def_prop_ro(
          "entity_types",
          [](const dolfinx::mesh::Topology& self)
          {
            std::vector<std::vector<dolfinx::mesh::CellType>> entity_types;
            for (int i = 0; i <= self.dim(); ++i)
              entity_types.push_back(self.entity_types(i));
            return entity_types;
          })
      .def("cell_name", [](const dolfinx::mesh::Topology& self)
           { return dolfinx::mesh::to_string(self.cell_type()); })
      .def("interprocess_facets",
           nb::overload_cast<>(&dolfinx::mesh::Topology::interprocess_facets,
                               nb::const_))
      .def("interprocess_facets",
           nb::overload_cast<std::int8_t>(
               &dolfinx::mesh::Topology::interprocess_facets, nb::const_))
      .def_prop_ro(
          "comm",
          [](dolfinx::mesh::Topology& self)
          { return MPICommWrapper(self.comm()); },
          nb::keep_alive<0, 1>());

  m.def("create_geometry",
        [](const dolfinx::mesh::Topology& topology,
           const std::vector<dolfinx::fem::CoordinateElement<double>>& elements,
           const std::vector<std::int64_t>& nodes,
           const std::vector<std::int64_t>& xdofs, const std::vector<double>& x,
           int dim)
        {
          return dolfinx::mesh::create_geometry(
              topology, elements, std::span<const std::int64_t>(nodes),
              std::span<const std::int64_t>(xdofs), x, dim);
        });

  m.def("create_topology",
        [](MPICommWrapper comm,
           const std::vector<dolfinx::mesh::CellType>& cell_type,
           const std::vector<std::vector<std::int64_t>>& cells,
           const std::vector<std::vector<std::int64_t>>& original_cell_index,
           const std::vector<std::vector<int>>& ghost_owners,
           const std::vector<std::int64_t>& boundary_vertices)
        {
          std::vector<std::span<const std::int64_t>> cells_span(cells.begin(),
                                                                cells.end());
          std::vector<std::span<const std::int64_t>> original_cell_index_span(
              original_cell_index.begin(), original_cell_index.end());
          std::vector<std::span<const int>> ghost_owners_span(
              ghost_owners.begin(), ghost_owners.end());
          std::span<const std::int64_t> boundary_vertices_span(
              boundary_vertices.begin(), boundary_vertices.end());

          return dolfinx::mesh::create_topology(
              comm.get(), cell_type, cells_span, original_cell_index_span,
              ghost_owners_span, boundary_vertices_span);
        });

  // dolfinx::mesh::MeshTags

  declare_meshtags<std::int8_t>(m, "int8");
  declare_meshtags<std::int32_t>(m, "int32");
  declare_meshtags<std::int64_t>(m, "int64");
  declare_meshtags<double>(m, "float64");

  declare_mesh<float>(m, "float32");
  declare_mesh<double>(m, "float64");

  m.def(
      "create_cell_partitioner",
      [](dolfinx::mesh::GhostMode gm) -> PythonCellPartitionFunction
      {
        return create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(gm));
      },
      "Create default cell partitioner.");
  m.def(
      "create_cell_partitioner",
      [](std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
             MPICommWrapper comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             bool ghosting)>
             part,
         dolfinx::mesh::GhostMode ghost_mode) -> PythonCellPartitionFunction
      {
        return create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(
                ghost_mode, create_partitioner_cpp(part)));
      },
      nb::arg("part"), nb::arg("ghost_mode") = dolfinx::mesh::GhostMode::none,
      "Create a cell partitioner from a graph partitioning function.");

  m.def(
      "exterior_facet_indices",
      [](const dolfinx::mesh::Topology& t)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::mesh::exterior_facet_indices(t));
      },
      nb::arg("topology"));
  m.def(
      "compute_incident_entities",
      [](const dolfinx::mesh::Topology& topology,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int d0, int d1)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::mesh::compute_incident_entities(
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
