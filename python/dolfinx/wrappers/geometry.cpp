// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <dolfinx/common/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_bbtree(nb::module_& m, std::string type)
{
  // dolfinx::geometry::BoundingBoxTree
  std::string pyclass_name = "BoundingBoxTree_" + type;
  nb::class_<dolfinx::geometry::BoundingBoxTree<T>>(m, pyclass_name.c_str())
      .def(
          "__init__",
          [](dolfinx::geometry::BoundingBoxTree<T>* bbt,
             const dolfinx::mesh::Mesh<T>& mesh, int dim,
             const nb::ndarray<std::int32_t, nb::numpy>& entities,
             double padding = 0.0)
          {
            new (bbt) dolfinx::geometry::BoundingBoxTree<T>(
                mesh, dim,
                std::span<const std::int32_t>(entities.data(), entities.size()),
                padding);
          },
          nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"),
          nb::arg("padding"))
      .def_prop_ro("num_bboxes",
                   &dolfinx::geometry::BoundingBoxTree<T>::num_bboxes)
      .def(
          "get_bbox",
          [](const dolfinx::geometry::BoundingBoxTree<T>& self,
             const std::size_t i)
          {
            std::array<T, 6> bbox = self.get_bbox(i);
            std::array<std::size_t, 2> shape{2, 3};
            return nb::ndarray<T, nb::numpy>(bbox.data(), 2, shape.data());
          },
          nb::arg("i"))
      .def("__repr__", &dolfinx::geometry::BoundingBoxTree<T>::str)
      .def(
          "create_global_tree",
          [](const dolfinx::geometry::BoundingBoxTree<T>& self,
             const dolfinx_wrappers::MPICommWrapper comm)
          { return self.create_global_tree(comm.get()); },
          nb::arg("comm"));

  m.def(
      "compute_collisions_points",
      [](const dolfinx::geometry::BoundingBoxTree<T>& tree,
         const nb::ndarray<T>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);

        return dolfinx::geometry::compute_collisions<T>(tree, _p);
      },
      nb::arg("tree"), nb::arg("points"));
  m.def(
      "compute_collisions_trees",
      [](const dolfinx::geometry::BoundingBoxTree<T>& treeA,
         const dolfinx::geometry::BoundingBoxTree<T>& treeB)
      {
        std::vector coll
            = dolfinx::geometry::compute_collisions<T>(treeA, treeB);
        std::array<std::size_t, 2> shape{std::size_t(coll.size() / 2), 2};
        return dolfinx_wrappers::as_nbarray(std::move(coll), shape);
      },
      nb::arg("tree0"), nb::arg("tree1"));
  m.def(
      "compute_closest_entity",
      [](const dolfinx::geometry::BoundingBoxTree<T>& tree,
         const dolfinx::geometry::BoundingBoxTree<T>& midpoint_tree,
         const dolfinx::mesh::Mesh<T>& mesh,
         const nb::ndarray<T, nb::numpy>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);

        return dolfinx_wrappers::as_nbarray(
            dolfinx::geometry::compute_closest_entity<T>(tree, midpoint_tree,
                                                         mesh, _p));
      },
      nb::arg("tree"), nb::arg("midpoint_tree"), nb::arg("mesh"),
      nb::arg("points"));
  m.def(
      "create_midpoint_tree",
      [](const dolfinx::mesh::Mesh<T>& mesh, int tdim,
         const nb::ndarray<std::int32_t, nb::numpy>& entities)
      {
        return dolfinx::geometry::create_midpoint_tree(
            mesh, tdim,
            std::span<const std::int32_t>(entities.data(), entities.size()));
      },
      nb::arg("mesh"), nb::arg("tdim"), nb::arg("entities"));
  m.def(
      "compute_colliding_cells",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         const dolfinx::graph::AdjacencyList<int>& candidate_cells,
         const nb::ndarray<T>& points)
          -> std::variant<dolfinx::graph::AdjacencyList<std::int32_t>,
                          nb::ndarray<std::int32_t>>
      {
        const int gdim = mesh.geometry().dim();
        std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);

        return dolfinx::geometry::compute_colliding_cells<T>(
            mesh, candidate_cells, _p);
      },
      nb::arg("mesh"), nb::arg("candidate_cells"), nb::arg("points"));

  m.def(
      "compute_distance_gjk",
      [](const nb::ndarray<T, nb::numpy>& p, const nb::ndarray<T, nb::numpy>& q)
      {
        const std::size_t p_s0 = p.ndim() == 1 ? 1 : p.shape(0);
        const std::size_t q_s0 = q.ndim() == 1 ? 1 : q.shape(0);
        std::span<const T> _p(p.data(), 3 * p_s0), _q(q.data(), 3 * q_s0);

        const std::array<T, 3> d
            = dolfinx::geometry::compute_distance_gjk<T>(_p, _q);

        std::vector<T> _d(d.begin(), d.end());

        return dolfinx_wrappers::as_nbarray(std::move(_d));
      },
      nb::arg("p"), nb::arg("q"));

  m.def(
      "squared_distance",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         std::vector<std::int32_t> indices,
         const nb::ndarray<T, nb::numpy>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);

        return dolfinx_wrappers::as_nbarray(
            dolfinx::geometry::squared_distance<T>(mesh, dim, indices, _p));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("indices"), nb::arg("points"));
  m.def("determine_point_ownership",
        [](const dolfinx::mesh::Mesh<T>& mesh, const nb::ndarray<T>& points)
        {
          const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          std::span<const T> _p(points.data(), 3 * p_s0);

          return dolfinx::geometry::determine_point_ownership<T>(mesh, _p);
        });
}
} // namespace

namespace dolfinx_wrappers
{
void geometry(nb::module_& m)
{
  declare_bbtree<float>(m, "float32");
  declare_bbtree<double>(m, "float64");
}
} // namespace dolfinx_wrappers
