// Copyright (C) 2017-2024 Chris N. Richardson and Garth N. Wells
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
#include <dolfinx/mesh/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <ranges>

namespace nb = nanobind;

namespace dolfinx_wrappers::part::impl
{
/// Wrap a Python graph partitioning function as a C++ function
template <typename Functor>
auto create_partitioner_cpp(Functor&& p)
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
auto create_cell_partitioner_py(Functor&& p)
{
  return [p](dolfinx_wrappers::MPICommWrapper comm, int n,
             const std::vector<dolfinx::mesh::CellType>& cell_types,
             std::vector<nb::ndarray<const std::int64_t, nb::numpy>> cells_nb)
  {
    std::vector<std::span<const std::int64_t>> cells;
    std::ranges::transform(
        cells_nb, std::back_inserter(cells), [](auto& c)
        { return std::span<const std::int64_t>(c.data(), c.size()); });
    return p(comm.get(), n, cell_types, cells);
  };
}

using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int,
        const std::vector<dolfinx::mesh::CellType>&,
        std::vector<nb::ndarray<const std::int64_t, nb::numpy>>)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, const std::vector<dolfinx::mesh::CellType>& q,
        const std::vector<std::span<const std::int64_t>>&)>;

/// Wrap a Python cell graph partitioning function as a C++ function
CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& p);
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
                   { return dolfinx_wrappers::numpy_dtype_v<T>; })
      .def_rw("name", &dolfinx::mesh::MeshTags<T>::name)
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
            int shape1 = x.shape(1);
            std::vector<T> x_vec;
            if (shape1 == 3 and x.stride(0) == 3 and x.stride(1) == 1)
              x_vec.assign(x.data(), x.data() + x.size());
            else
            {
              // Pad geometry to be 3D
              x_vec.assign(3 * x.shape(0), 0);
              auto _x = x.view();
              for (std::size_t i = 0; i < x.shape(0); ++i)
                for (int j = 0; j < shape1; ++j)
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
            std::span<T> x = self.x();
            return nb::ndarray<T, nb::shape<-1, 3>, nb::numpy>(
                x.data(), {x.size() / 3, 3});
          },
          nb::rv_policy::reference_internal,
          "Return coordinates of all geometry points. Each row is the "
          "coordinate of a point.")
      .def_prop_ro(
          "cmap", [](dolfinx::mesh::Geometry<T>& self) { return self.cmap(); },
          "The coordinate map")
      .def(
          "cmaps", [](dolfinx::mesh::Geometry<T>& self, int i)
          { return self.cmaps()[i]; }, "The ith coordinate map")
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
         const part::impl::PythonCellPartitionFunction& part)
      {
        return dolfinx::mesh::create_interval<T>(
            comm.get(), n, p, mode,
            part::impl::create_cell_partitioner_cpp(part));
      },
      nb::arg("comm"), nb::arg("n"), nb::arg("p"), nb::arg("ghost_mode"),
      nb::arg("partitioner").none());

  std::string create_rectangle("create_rectangle_" + type);
  m.def(
      create_rectangle.c_str(),
      [](MPICommWrapper comm, std::array<std::array<T, 2>, 2> p,
         std::array<std::int64_t, 2> n, dolfinx::mesh::CellType celltype,
         const part::impl::PythonCellPartitionFunction& part,
         dolfinx::mesh::DiagonalType diagonal)
      {
        return dolfinx::mesh::create_rectangle<T>(
            comm.get(), p, n, celltype,
            part::impl::create_cell_partitioner_cpp(part), diagonal);
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none(), nb::arg("diagonal"));

  std::string create_box("create_box_" + type);
  m.def(
      create_box.c_str(),
      [](MPICommWrapper comm, std::array<std::array<T, 3>, 2> p,
         std::array<std::int64_t, 3> n, dolfinx::mesh::CellType celltype,
         const part::impl::PythonCellPartitionFunction& part)
      {
        MPI_Comm _comm = comm.get();
        return dolfinx::mesh::create_box<T>(
            _comm, _comm, p, n, celltype,
            part::impl::create_cell_partitioner_cpp(part));
      },
      nb::arg("comm"), nb::arg("p"), nb::arg("n"), nb::arg("celltype"),
      nb::arg("partitioner").none());

  m.def("create_mesh",
        [](MPICommWrapper comm,
           const std::vector<nb::ndarray<const std::int64_t, nb::ndim<1>,
                                         nb::c_contig>>& cells_nb,
           const std::vector<dolfinx::fem::CoordinateElement<T>>& elements,
           nb::ndarray<const T, nb::c_contig> x,
           const part::impl::PythonCellPartitionFunction& p,
           std::optional<std::int32_t> max_facet_to_cell_links)
        {
          std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);

          std::vector<std::span<const std::int64_t>> cells;
          std::ranges::transform(
              cells_nb, std::back_inserter(cells), [](auto& c)
              { return std::span<const std::int64_t>(c.data(), c.size()); });

          if (p)
          {
            auto p_wrap
                = [p](MPI_Comm comm, int n,
                      const std::vector<dolfinx::mesh::CellType>& cell_types,
                      const std::vector<std::span<const std::int64_t>>& cells)
            {
              std::vector<nb::ndarray<const std::int64_t, nb::numpy>> cells_nb;
              std::ranges::transform(
                  cells, std::back_inserter(cells_nb),
                  [](auto c)
                  {
                    return nb::ndarray<const std::int64_t, nb::numpy>(
                        c.data(), {c.size()});
                  });
              return p(MPICommWrapper(comm), n, cell_types, cells_nb);
            };
            return dolfinx::mesh::create_mesh(
                comm.get(), comm.get(), cells, elements, comm.get(),
                std::span(x.data(), x.size()), {x.shape(0), shape1}, p_wrap,
                max_facet_to_cell_links);
          }
          else
            return dolfinx::mesh::create_mesh(
                comm.get(), comm.get(), cells, elements, comm.get(),
                std::span(x.data(), x.size()), {x.shape(0), shape1}, nullptr,
                max_facet_to_cell_links);
        });

  m.def(
      "create_mesh",
      [](MPICommWrapper comm,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> cells,
         const dolfinx::fem::CoordinateElement<T>& element,
         nb::ndarray<const T, nb::c_contig> x,
         const part::impl::PythonCellPartitionFunction& p,
         std::optional<std::int32_t> max_facet_to_cell_links)
      {
        std::size_t shape1 = x.ndim() == 1 ? 1 : x.shape(1);
        if (p)
        {
          auto p_wrap
              = [p](MPI_Comm comm, int n,
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
            return p(MPICommWrapper(comm), n, cell_types, cells_nb);
          };
          return dolfinx::mesh::create_mesh(
              comm.get(), comm.get(), std::span(cells.data(), cells.size()),
              element, comm.get(), std::span(x.data(), x.size()),
              {x.shape(0), shape1}, p_wrap, max_facet_to_cell_links);
        }
        else
        {
          return dolfinx::mesh::create_mesh(
              comm.get(), comm.get(), std::span(cells.data(), cells.size()),
              element, comm.get(), std::span(x.data(), x.size()),
              {x.shape(0), shape1}, nullptr, max_facet_to_cell_links);
        }
      },
      nb::arg("comm"), nb::arg("cells"), nb::arg("element"),
      nb::arg("x").noconvert(), nb::arg("partitioner").none(),
      nb::arg("max_facet_to_cell_links").none(),
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