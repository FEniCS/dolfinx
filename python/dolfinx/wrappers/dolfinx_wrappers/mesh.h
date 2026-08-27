// Copyright (C) 2017-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include "array.h"
#include "numpy_dtype.h"
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/partition.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <ranges>
#include <span>
#include <variant>

namespace nb = nanobind;

namespace dolfinx_wrappers::part::impl
{
/// Wrap a Python graph partitioning function as a C++ function.
template <typename Functor>
auto create_partitioner_cpp(Functor&& p)
{
  return [p](MPI_Comm comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             std::span<const std::int32_t> node_weights,
             std::span<const std::int32_t> edge_weights, bool ghosting)
  {
    nb::ndarray<const std::int32_t, nb::numpy> node_weights_nb(
        node_weights.data(), {node_weights.size()});
    nb::ndarray<const std::int32_t, nb::numpy> edge_weights_nb(
        edge_weights.data(), {edge_weights.size()});
    return p(dolfinx_wrappers::MPICommWrapper(comm), nparts, local_graph,
             node_weights_nb, edge_weights_nb, ghosting);
  };
}

using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int,
        const dolfinx::graph::AdjacencyList<std::int64_t>&,
        nb::ndarray<const std::int32_t, nb::numpy>,
        nb::ndarray<const std::int32_t, nb::numpy>, bool)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, const dolfinx::graph::AdjacencyList<std::int64_t>&,
        std::span<const std::int32_t>, std::span<const std::int32_t>, bool)>;

/// Opaque handle wrapping a partitioning function `Fn`, bound to
/// Python as its own type -- see GeometricPartitioner/HybridPartitioner
/// below.
template <typename Fn>
struct OpaquePartitioner
{
  Fn fn;
};

/// Opaque handles for a dolfinx::graph::geom_partition_fn and a
/// dolfinx::graph::hybrid_partition_fn, bound to Python as their own
/// types so that create_mesh can distinguish them from a
/// PythonCellPartitionFunction (and from each other): nanobind cannot
/// tell which of several different std::function signatures a bare
/// Python callable is meant to satisfy, so distinct callable shapes must
/// be distinct Python types, not disambiguated by argument count.
using GeometricPartitioner
    = OpaquePartitioner<dolfinx::graph::geom_partition_fn>;
using HybridPartitioner
    = OpaquePartitioner<dolfinx::graph::hybrid_partition_fn>;

/// The Python-visible partitioner argument accepted by create_mesh: a
/// GeometricPartitioner, a HybridPartitioner, or a plain graph
/// partitioner (std::nullopt disables redistribution, as for
/// PythonCellPartitionFunction alone). The two opaque handle types must
/// be listed first: nanobind tries variant alternatives in order and
/// stops at the first that converts, and their __call__ makes them
/// callable, so a raw PythonCellPartitionFunction-shaped std::function
/// would also happily (and wrongly) construct from either if that
/// alternative were tried first.
using PythonMeshPartitioner
    = std::variant<GeometricPartitioner, HybridPartitioner,
                   std::optional<PythonCellPartitionFunction>>;

/// Convert create_mesh's Python-visible partitioner argument to the
/// dolfinx::mesh::AnyCellPartitionFunction that dolfinx::mesh::create_mesh
/// expects.
inline dolfinx::mesh::AnyCellPartitionFunction
to_any_cell_partitioner(const PythonMeshPartitioner& p)
{
  if (const auto* geo = std::get_if<GeometricPartitioner>(&p))
    return dolfinx::mesh::AnyCellPartitionFunction(geo->fn);
  if (const auto* hyb = std::get_if<HybridPartitioner>(&p))
    return dolfinx::mesh::AnyCellPartitionFunction(hyb->fn);

  const auto& opt = std::get<std::optional<PythonCellPartitionFunction>>(p);
  return dolfinx::mesh::AnyCellPartitionFunction(
      opt ? create_partitioner_cpp(*opt) : CppCellPartitionFunction(nullptr));
}
} // namespace dolfinx_wrappers::part::impl

namespace dolfinx_wrappers
{

template <typename T>
void declare_meshtags(nb::module_& m, const std::string& type)
{
  std::string pyclass_name = std::string("MeshTags_") + type;
  nb::class_<dolfinx::mesh::MeshTags<T>>(m, pyclass_name.c_str(),
                                         "MeshTags object")
      .def(
          "__init__",
          [](dolfinx::mesh::MeshTags<T>* self,
             std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indices,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> values,
             std::string name)
          {
            std::vector<std::int32_t> indices_vec(
                indices.data(), indices.data() + indices.size());
            std::vector<T> values_vec(values.data(),
                                      values.data() + values.size());
            new (self) dolfinx::mesh::MeshTags<T>(
                topology, dim, std::move(indices_vec), std::move(values_vec),
                std::move(name));
          })
      .def_prop_ro("dtype", [](const dolfinx::mesh::MeshTags<T>&)
                   { return dolfinx_wrappers::numpy_dtype_v<T>; })
      .def_prop_rw(
          "name",
          [](const dolfinx::mesh::MeshTags<T>& self) { return self.name(); },
          [](dolfinx::mesh::MeshTags<T>& self, std::string name)
          { self.name(name); })
      .def_prop_ro("dim", &dolfinx::mesh::MeshTags<T>::dim)
      .def_prop_ro("topology", &dolfinx::mesh::MeshTags<T>::topology)
      .def_prop_ro(
          "values",
          [](const dolfinx::mesh::MeshTags<T>& self)
          {
            std::span<const T> v = self.values();
            return nb::ndarray<const T, nb::numpy>(v.data(), {v.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indices",
          [](const dolfinx::mesh::MeshTags<T>& self)
          {
            std::span<const std::int32_t> idx = self.indices();
            return nb::ndarray<const std::int32_t, nb::numpy>(idx.data(),
                                                              {idx.size()});
          },
          nb::rv_policy::reference_internal)
      .def("find", [](dolfinx::mesh::MeshTags<T>& self, T value)
           { return as_nbarray(self.find(value)); });

  m.def("create_meshtags",
        [](std::shared_ptr<const dolfinx::mesh::Topology> topology, int dim,
           const dolfinx::graph::AdjacencyList<std::int32_t>& entities,
           nb::ndarray<const T, nb::ndim<1>, nb::c_contig> values,
           std::string name)
        {
          return dolfinx::mesh::create_meshtags(
              topology, dim, entities, std::span(values.data(), values.size()),
              std::move(name));
        });
  std::string pyfunc_name = "transfer_meshtags_to_submesh_" + type;
  m.def(
      pyfunc_name.c_str(),
      [](const dolfinx::mesh::MeshTags<T>& tags,
         std::shared_ptr<const dolfinx::mesh::Topology> submesh_topology,
         const dolfinx::mesh::EntityMap& vertex_map,
         const dolfinx::mesh::EntityMap& cell_map)
      {
        return dolfinx::mesh::transfer_meshtags_to_submesh<T>(
            tags, submesh_topology, vertex_map, cell_map);
      },
      nanobind::arg("tags"), nanobind::arg("submesh_topology"),
      nanobind::arg("vertex_map"), nanobind::arg("cell_map"));
}

template <typename T>
void declare_mesh(nb::module_& m, std::string type)
{
  std::string pyclass_geometry_name = std::string("Geometry_") + type;
  nb::class_<dolfinx::mesh::Geometry<T>>(m, pyclass_geometry_name.c_str(),
                                         "Geometry object")
      .def(
          "__init__",
          [](dolfinx::mesh::Geometry<T>* self,
             std::shared_ptr<const dolfinx::common::IndexMap> index_map,
             nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
             const dolfinx::fem::CoordinateElement<T>& element,
             nb::ndarray<const T, nb::ndim<2>> x,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>
                 input_global_indices)
          {
            std::size_t shape1 = x.shape(1);
            if (shape1 == 0 or shape1 > 3)
            {
              throw std::runtime_error(
                  "Geometry point array must have shape (num_points, dim) "
                  "with 0 < dim <= 3.");
            }

            std::vector<T> x_vec;
            if (shape1 == 3 and x.stride(0) == 3 and x.stride(1) == 1)
              x_vec.assign(x.data(), x.data() + x.size());
            else
            {
              // Pad geometry to be 3D
              x_vec.assign(3 * x.shape(0), 0);
              auto _x = x.view();
              for (std::size_t i = 0; i < x.shape(0); ++i)
                for (std::size_t j = 0; j < shape1; ++j)
                  x_vec[3 * i + j] = _x(i, j);
            }

            new (self) dolfinx::mesh::Geometry<T>(
                std::move(index_map),
                std::vector<std::vector<std::int32_t>>(
                    1, std::vector<std::int32_t>(
                           dofmap.data(), dofmap.data() + dofmap.size())),
                {element}, std::move(x_vec), shape1,
                std::vector(input_global_indices.data(),
                            input_global_indices.data()
                                + input_global_indices.size()));
          },
          nb::arg("index_map"), nb::arg("dofmap"), nb::arg("element"),
          nb::arg("x"), nb::arg("input_global_indices"))
      .def_prop_ro("dim", &dolfinx::mesh::Geometry<T>::dim,
                   "Geometric dimension")
      .def_prop_ro(
          "dofmaps",
          [](const dolfinx::mesh::Geometry<T>& self)
          {
            auto dms = self.dofmaps();
            std::vector<nb::ndarray<const std::int32_t, nb::numpy>> result;
            result.reserve(dms.size());
            for (auto& dm : dms)
            {
              result.push_back(nb::ndarray<const std::int32_t, nb::numpy>(
                  dm.data_handle(), {dm.extent(0), dm.extent(1)}));
            }
            return result;
          },
          nb::rv_policy::reference_internal, "The geometry dofmaps")
      .def("index_map", &dolfinx::mesh::Geometry<T>::index_map)
      .def_prop_ro(
          "x",
          [](dolfinx::mesh::Geometry<T>& self)
          {
            std::span<T> x = self.x();
            return nb::ndarray<T, nb::shape<-1, 3>, nb::numpy>(
                x.data(), {x.size() / 3, 3});
          },
          nb::rv_policy::reference_internal,
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_prop_ro(
          "cmaps", [](dolfinx::mesh::Geometry<T>& self)
          { return self.cmaps(); }, "The coordinate maps")
      .def_prop_ro(
          "input_global_indices",
          [](const dolfinx::mesh::Geometry<T>& self)
          {
            const std::vector<std::int64_t>& id_to_global
                = self.input_global_indices();
            return nb::ndarray<const std::int64_t, nb::numpy>(
                id_to_global.data(), {id_to_global.size()});
          },
          nb::rv_policy::reference_internal);

  std::string pyclass_mesh_name = std::string("Mesh_") + type;
  nb::class_<dolfinx::mesh::Mesh<T>>(m, pyclass_mesh_name.c_str(),
                                     nb::dynamic_attr(), "Mesh object")
      .def(
          "__init__",
          [](dolfinx::mesh::Mesh<T>* mesh, MPICommWrapper comm,
             std::shared_ptr<dolfinx::mesh::Topology> topology,
             dolfinx::mesh::Geometry<T>& geometry)
          {
            new (mesh) dolfinx::mesh::Mesh<T>(comm.get(), std::move(topology),
                                              geometry);
          },
          nb::arg("comm"), nb::arg("topology"), nb::arg("geometry"))
      .def_prop_ro("geometry",
                   nb::overload_cast<>(&dolfinx::mesh::Mesh<T>::geometry),
                   "Mesh geometry")
      .def_prop_ro("topology",
                   nb::overload_cast<>(&dolfinx::mesh::Mesh<T>::topology),
                   "Mesh topology")
      .def_prop_ro(
          "comm", [](dolfinx::mesh::Mesh<T>& self)
          { return MPICommWrapper(self.comm()); }, nb::keep_alive<0, 1>())
      .def_rw("name", &dolfinx::mesh::Mesh<T>::name);

  std::string create_interval("create_interval_" + type);
  m.def(
      create_interval.c_str(),
      [](MPICommWrapper comm, std::int64_t n, std::array<T, 2> p,
         dolfinx::mesh::GhostMode mode,
         const part::impl::PythonMeshPartitioner& part, int gdim)
      {
        return dolfinx::mesh::create_interval<T>(
            comm.get(), n, p, mode, part::impl::to_any_cell_partitioner(part),
            gdim);
      },
      nb::arg("comm"), nb::arg("n"), nb::arg("p"), nb::arg("ghost_mode"),
      nb::arg("partitioner").none(), nb::arg("gdim"));

  std::string create_rectangle("create_rectangle_" + type);
  m.def(
      create_rectangle.c_str(),
      [](MPICommWrapper comm, std::array<std::array<T, 2>, 2> p,
         std::array<std::int64_t, 2> n, dolfinx::mesh::CellType celltype,
         const part::impl::PythonMeshPartitioner& part,
         dolfinx::mesh::DiagonalType diagonal, int gdim,
         dolfinx::mesh::GhostMode ghost_mode)
      {
        return dolfinx::mesh::create_rectangle<T>(
            comm.get(), p, n, celltype,
            part::impl::to_any_cell_partitioner(part), diagonal, gdim,
            ghost_mode);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none(), nb::arg("diagonal"), nb::arg("gdim"),
      nb::arg("ghost_mode"));

  std::string create_box("create_box_" + type);
  m.def(
      create_box.c_str(),
      [](MPICommWrapper comm, std::array<std::array<T, 3>, 2> p,
         std::array<std::int64_t, 3> n, dolfinx::mesh::CellType celltype,
         const part::impl::PythonMeshPartitioner& part,
         dolfinx::mesh::GhostMode ghost_mode)
      {
        MPI_Comm _comm = comm.get();
        return dolfinx::mesh::create_box<T>(
            _comm, _comm, p, n, celltype,
            part::impl::to_any_cell_partitioner(part), ghost_mode);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none(), nb::arg("ghost_mode"));

  m.def(
      "create_mesh",
      [](MPICommWrapper comm,
         const std::vector<nb::ndarray<const std::int64_t, nb::ndim<1>,
                                       nb::c_contig>>& cells_nb,
         const std::vector<dolfinx::fem::CoordinateElement<T>>& elements,
         nb::ndarray<const T, nb::c_contig> x,
         const part::impl::PythonMeshPartitioner& p,
         dolfinx::mesh::GhostMode ghost_mode,
         std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
         std::optional<
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
             cell_weights)
      {
        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);

        std::vector<std::span<const std::int64_t>> cells;
        std::ranges::transform(
            cells_nb, std::back_inserter(cells), [](auto& c)
            { return std::span<const std::int64_t>(c.data(), c.size()); });

        std::span<const std::int32_t> cell_weights_span
            = cell_weights ? std::span<const std::int32_t>(cell_weights->data(),
                                                           cell_weights->size())
                           : std::span<const std::int32_t>();

        return dolfinx::mesh::create_mesh(
            comm.get(), comm.get(), cells, cell_weights_span, elements,
            comm.get(), std::span(x.data(), x.size()), {x.shape(0), shape1},
            part::impl::to_any_cell_partitioner(p), ghost_mode,
            max_facet_to_cell_links, num_threads);
      },
      nb::arg("comm"), nb::arg("cells"), nb::arg("elements"),
      nb::arg("x").noconvert(), nb::arg("partitioner").none(),
      nb::arg("ghost_mode"), nb::arg("max_facet_to_cell_links").none(),
      nb::arg("num_threads"), nb::arg("cell_weights").none(),
      "Helper function for creating a mixed topology mesh.");

  m.def(
      "create_mesh",
      [](MPICommWrapper comm,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> cells,
         const dolfinx::fem::CoordinateElement<T>& element,
         nb::ndarray<const T, nb::c_contig> x,
         const part::impl::PythonMeshPartitioner& p,
         dolfinx::mesh::GhostMode ghost_mode,
         std::optional<std::int32_t> max_facet_to_cell_links, int num_threads,
         std::optional<
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
             cell_weights)
      {
        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);
        std::span<const std::int32_t> cell_weights_span
            = cell_weights ? std::span<const std::int32_t>(cell_weights->data(),
                                                           cell_weights->size())
                           : std::span<const std::int32_t>();
        return dolfinx::mesh::create_mesh(
            comm.get(), comm.get(), std::span(cells.data(), cells.size()),
            cell_weights_span, element, comm.get(),
            std::span(x.data(), x.size()), {x.shape(0), shape1},
            part::impl::to_any_cell_partitioner(p), ghost_mode,
            max_facet_to_cell_links, num_threads);
      },
      nb::arg("comm"), nb::arg("cells"), nb::arg("element"),
      nb::arg("x").noconvert(), nb::arg("partitioner").none(),
      nb::arg("ghost_mode"), nb::arg("max_facet_to_cell_links").none(),
      nb::arg("num_threads"), nb::arg("cell_weights").none(),
      "Helper function for creating meshes.");
  m.def(
      "create_submesh",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        auto [submesh, e_map, v_map, g_map] = dolfinx::mesh::create_submesh(
            mesh, dim, std::span(entities.data(), entities.size()));
        auto _g_map = as_nbarray(std::move(g_map));

        return std::tuple(std::move(submesh), std::move(e_map),
                          std::move(v_map), _g_map);
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
      "locate_entities",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker,
         int entity_type_idx)
      {
        auto cpp_marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)});
          auto marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return as_nbarray(dolfinx::mesh::locate_entities(mesh, dim, cpp_marker,
                                                         entity_type_idx));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("marker"),
      nb::arg("entity_type_idx"));

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
         bool permute)
      {
        auto [geom_indices, idx_shape] = dolfinx::mesh::entities_to_geometry(
            mesh, dim, std::span(entities.data(), entities.size()), permute);
        return as_nbarray(std::move(geom_indices), idx_shape);
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"), nb::arg("permute"));

  m.def("create_geometry",
        [](const dolfinx::mesh::Topology& topology,
           const std::vector<dolfinx::fem::CoordinateElement<T>>& elements,
           nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> nodes,
           nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> xdofs,
           nb::ndarray<const T, nb::ndim<1>, nb::c_contig> x, int dim)
        {
          return dolfinx::mesh::create_geometry(
              topology, elements,
              std::span<const std::int64_t>(nodes.data(), nodes.size()),
              std::span<const std::int64_t>(xdofs.data(), xdofs.size()),
              std::span<const T>(x.data(), x.size()), dim);
        });
}

} // namespace dolfinx_wrappers
