// Copyright (C) 2017-2025 Chris N. Richardson, Garth N. Wells and Jørgen S.
// Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/mesh.h"
#include "dolfinx_wrappers/MPICommWrapper.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
namespace part::impl
{
CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& p)
{
  if (p)
  {
    return [p](MPI_Comm comm, int n,
               const std::vector<dolfinx::mesh::CellType>& cell_types,
               const std::vector<std::span<const std::int64_t>>& cells)
    {
      std::vector<nb::ndarray<const std::int64_t, nb::numpy>> cells_nb;
      std::ranges::transform(
          cells, std::back_inserter(cells_nb),
          [](auto c)
          {
            return nb::ndarray<const std::int64_t, nb::numpy>(c.data(),
                                                              {c.size()});
          });
      return p(dolfinx_wrappers::MPICommWrapper(comm), n, cell_types, cells_nb);
    };
  }
  else
    return nullptr;
}
} // namespace part::impl

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
      .def_prop_ro("name", [](const nb::object& obj)
                   { return nb::getattr(obj, "__name__"); });

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
  m.def(
      "extract_topology",
      [](dolfinx::mesh::CellType cell_type,
         const dolfinx::fem::ElementDofLayout& layout,
         nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> cells)
      {
        return dolfinx_wrappers::as_nbarray(dolfinx::mesh::extract_topology(
            cell_type, layout, std::span(cells.data(), cells.size())));
      },
      nb::arg("cell_type"), nb::arg("layout"), nb::arg("cells"));

  m.def(
      "build_dual_graph",
      [](const MPICommWrapper comm, dolfinx::mesh::CellType cell_type,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
         std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
      {
        std::vector<dolfinx::mesh::CellType> c = {cell_type};
        return dolfinx::mesh::build_dual_graph(
            comm.get(), std::span{c}, {cells.array()}, max_facet_to_cell_links,
            num_threads);
      },
      nb::arg("comm"), nb::arg("cell_type"), nb::arg("cells"),
      nb::arg("max_facet_to_cell_links").none(), nb::arg("num_threads"),
      "Build dual graph for cells");

  m.def(
      "build_dual_graph",
      [](const MPICommWrapper comm,
         std::vector<dolfinx::mesh::CellType>& cell_types,
         const std::vector<
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>>& cells,
         std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
      {
        std::vector<std::span<const std::int64_t>> cell_span(cells.size());
        for (std::size_t i = 0; i < cells.size(); ++i)
        {
          cell_span[i]
              = std::span<const std::int64_t>(cells[i].data(), cells[i].size());
        }
        return dolfinx::mesh::build_dual_graph(
            comm.get(), cell_types, cell_span, max_facet_to_cell_links,
            num_threads);
      },
      nb::arg("comm"), nb::arg("cell_types"), nb::arg("cells"),
      nb::arg("max_facet_to_cell_links").none(), nb::arg("num_threads"),
      "Build dual graph for cells");

  // dolfinx::mesh::GhostMode enums
  nb::enum_<dolfinx::mesh::GhostMode>(m, "GhostMode")
      .value("none", dolfinx::mesh::GhostMode::none)
      .value("shared_facet", dolfinx::mesh::GhostMode::shared_facet);

  // dolfinx::mesh::TopologyComputation
  m.def(
      "compute_entities",
      [](const dolfinx::mesh::Topology& topology, int dim,
         dolfinx::mesh::CellType entity_type, int num_threads)
      {
        return dolfinx::mesh::compute_entities(topology, dim, entity_type,
                                               num_threads);
      },
      nb::arg("topology"), nb::arg("dim"), nb::arg("entity_type"),
      nb::arg("num_threads") = 1);
  m.def("compute_connectivity", &dolfinx::mesh::compute_connectivity,
        nb::arg("topology"), nb::arg("d0"), nb::arg("d1"));

  // dolfinx::mesh::EntityMap class
  nb::class_<dolfinx::mesh::EntityMap>(m, "EntityMap", "EntityMap object")
      .def(
          "__init__",
          [](dolfinx::mesh::EntityMap* self,
             const std::shared_ptr<const dolfinx::mesh::Topology>& topology,
             const std::shared_ptr<const dolfinx::mesh::Topology>& sub_topology,
             int dim,
             const nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>&
                 sub_topology_to_topology)
          {
            new (self) dolfinx::mesh::EntityMap(
                topology, sub_topology, dim,
                std::vector(sub_topology_to_topology.data(),
                            sub_topology_to_topology.data()
                                + sub_topology_to_topology.size()));
          },
          nb::arg("topology"), nb::arg("sub_topology"), nb::arg("dim"),
          nb::arg("sub_topology_to_topology"))
      .def(
          "sub_topology_to_topology",
          [](const dolfinx::mesh::EntityMap& self,
             const nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>&
                 entities,
             bool inverse)
          {
            std::vector<std::int32_t> mapped_entities
                = self.sub_topology_to_topology(
                    std::span(entities.data(), entities.size()), inverse);
            return as_nbarray(std::move(mapped_entities));
          },
          nb::arg("entities"), nb::arg("inverse"))
      .def_prop_ro("dim", &dolfinx::mesh::EntityMap::dim)
      .def_prop_ro("topology", &dolfinx::mesh::EntityMap::topology)
      .def_prop_ro("sub_topology", &dolfinx::mesh::EntityMap::sub_topology);

  // dolfinx::mesh::Topology class
  nb::class_<dolfinx::mesh::Topology>(m, "Topology", nb::dynamic_attr(),
                                      "Topology object")
      .def(
          "__init__",
          [](dolfinx::mesh::Topology* t, dolfinx::mesh::CellType cell_type,
             std::shared_ptr<const dolfinx::common::IndexMap> vertex_map,
             std::shared_ptr<const dolfinx::common::IndexMap> cell_map,
             std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> cells,
             std::optional<
                 nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>>
                 original_index)
          {
            using U = std::vector<std::vector<std::int64_t>>;
            using V = std::optional<U>;
            V idx = original_index
                        ? U(1, std::vector(original_index->data(),
                                           original_index->data()
                                               + original_index->size()))
                        : V(std::nullopt);
            new (t) dolfinx::mesh::Topology({cell_type}, vertex_map, {cell_map},
                                            {cells}, idx);
          },
          nb::arg("cell_type"), nb::arg("vertex_map"), nb::arg("cell_map"),
          nb::arg("cells"), nb::arg("original_index").none())
      .def("create_entities", &dolfinx::mesh::Topology::create_entities,
           nb::arg("dim"), nb::arg("num_threads") = 1)
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
      .def_prop_rw(
          "original_cell_index",
          [](const dolfinx::mesh::Topology& self)
          {
            if (self.original_cell_index.size() != 1)
              throw std::runtime_error("Mixed topology unsupported.");
            const std::vector<std::vector<std::int64_t>>& idx
                = self.original_cell_index;
            return nb::ndarray<const std::int64_t, nb::numpy>(
                idx.front().data(), {idx.front().size()});
          },
          [](dolfinx::mesh::Topology& self,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>
                 original_cell_indices)
          {
            self.original_cell_index.resize(1);
            self.original_cell_index.front().assign(
                original_cell_indices.data(),
                original_cell_indices.data() + original_cell_indices.size());
          },
          nb::arg("original_cell_indices"))
      .def_prop_ro(
          "original_cell_indices",
          [](const dolfinx::mesh::Topology& self)
          {
            const std::vector<std::vector<std::int64_t>>& indices
                = self.original_cell_index;
            std::vector<nb::ndarray<const std::int64_t, nb::numpy>> idx_nb;
            for (auto& oci : indices)
            {
              idx_nb.push_back(nb::ndarray<const std::int64_t, nb::numpy>(
                  oci.data(), {oci.size()}));
            }
            return idx_nb;
          })
      .def("connectivity",
           nb::overload_cast<int, int>(&dolfinx::mesh::Topology::connectivity,
                                       nb::const_),
           nb::arg("d0"), nb::arg("d1"))
      .def("connectivity",
           nb::overload_cast<std::array<int, 2>, std::array<int, 2>>(
               &dolfinx::mesh::Topology::connectivity, nb::const_),
           nb::arg("d0"), nb::arg("d1"))
      .def("index_map", &dolfinx::mesh::Topology::index_map, nb::arg("dim"))
      .def("index_maps", &dolfinx::mesh::Topology::index_maps, nb::arg("dim"))
      .def_prop_ro("cell_type", &dolfinx::mesh::Topology::cell_type)
      .def_prop_ro("cell_types", &dolfinx::mesh::Topology::cell_types)
      .def_prop_ro(
          "entity_types",
          [](const dolfinx::mesh::Topology& self)
          {
            std::vector<std::vector<dolfinx::mesh::CellType>> entity_types;
            for (int i = 0; i <= self.dim(); ++i)
              entity_types.push_back(self.entity_types(i));
            return entity_types;
          })
      .def(
          "interprocess_facets",
          [](const dolfinx::mesh::Topology& self)
          {
            const std::vector<std::int32_t>& facets
                = self.interprocess_facets();
            return nb::ndarray<const std::int32_t, nb::numpy>(facets.data(),
                                                              {facets.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "comm", [](dolfinx::mesh::Topology& self)
          { return MPICommWrapper(self.comm()); }, nb::keep_alive<0, 1>());

  m.def(
      "create_topology",
      [](MPICommWrapper comm,
         const std::vector<dolfinx::mesh::CellType>& cell_type,
         const std::vector<std::vector<std::int64_t>>& cells,
         const std::vector<std::vector<std::int64_t>>& original_cell_index,
         const std::vector<std::vector<int>>& ghost_owners,
         const std::vector<std::int64_t>& boundary_vertices, int num_threads)
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
            ghost_owners_span, boundary_vertices_span, num_threads);
      },
      "Create a Topology object.");

  m.def("compute_mixed_cell_pairs", &dolfinx::mesh::compute_mixed_cell_pairs);

  m.def(
      "create_cell_partitioner",
      [](dolfinx::mesh::GhostMode mode,
         std::optional<std::int32_t> max_facet_to_cell_links)
          -> part::impl::PythonCellPartitionFunction
      {
        return part::impl::create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(
                mode, &dolfinx::graph::partition_graph,
                max_facet_to_cell_links));
      },
      nb::arg("mode"), nb::arg("max_facet_to_cell_links").none(),
      "Create default cell partitioner.");
  m.def(
      "create_cell_partitioner",
      [](const std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
             MPICommWrapper comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             bool ghosting)>& part,
         dolfinx::mesh::GhostMode mode,
         std::optional<std::int32_t> max_facet_to_cell_links)
          -> part::impl::PythonCellPartitionFunction
      {
        return part::impl::create_cell_partitioner_py(
            dolfinx::mesh::create_cell_partitioner(
                mode, part::impl::create_partitioner_cpp(part),
                max_facet_to_cell_links));
      },
      nb::arg("part"), nb::arg("ghost_mode"),
      nb::arg("max_facet_to_cell_links").none(),
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

  declare_meshtags<std::int8_t>(m, "int8");
  declare_meshtags<std::int32_t>(m, "int32");
  declare_meshtags<std::int64_t>(m, "int64");
  declare_meshtags<double>(m, "float64");

  declare_mesh<float>(m, "float32");
  declare_mesh<double>(m, "float64");
}
} // namespace dolfinx_wrappers
