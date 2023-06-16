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
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void geometry(nb::module_& m)
{
  m.def(
      "create_midpoint_tree",
      [](const dolfinx::mesh::Mesh<double>& mesh, int tdim,
         const nb::ndarray<std::int32_t, nb::numpy>& entities)
      {
        std::size_t size = std::reduce(entities.shape_ptr(),
                                       entities.shape_ptr() + entities.ndim(),
                                       1, std::multiplies<std::size_t>());

        return dolfinx::geometry::create_midpoint_tree(
            mesh, tdim, std::span<const std::int32_t>(entities.data(), size));
      },
      nb::arg("mesh"), nb::arg("tdim"), nb::arg("entities"));

  m.def(
      "compute_closest_entity",
      [](const dolfinx::geometry::BoundingBoxTree<double>& tree,
         const dolfinx::geometry::BoundingBoxTree<double>& midpoint_tree,
         const dolfinx::mesh::Mesh<double>& mesh,
         const nb::ndarray<double, nb::numpy>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const double> p(points.data(), 3 * p_s0);
        return as_nbarray(dolfinx::geometry::compute_closest_entity<double>(
            tree, midpoint_tree, mesh, p));
      },
      nb::arg("tree"), nb::arg("midpoint_tree"), nb::arg("mesh"),
      nb::arg("points"));
  m.def("determine_point_ownership",
        [](const dolfinx::mesh::Mesh<double>& mesh,
           const nb::ndarray<double>& points)
        {
          if (points.ndim() > 2)
            throw std::runtime_error("Array has wrong ndim.");
          const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          std::span<const double> p(points.data(), 3 * p_s0);
          return dolfinx::geometry::determine_point_ownership<double>(mesh, p);
        });

  m.def(
      "compute_collisions",
      [](const dolfinx::geometry::BoundingBoxTree<double>& tree,
         const nb::ndarray<double>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const double> _p(points.data(), 3 * p_s0);
        return dolfinx::geometry::compute_collisions<double>(tree, _p);
      },
      nb::arg("tree"), nb::arg("points"));
  m.def(
      "compute_collisions",
      [](const dolfinx::geometry::BoundingBoxTree<double>& treeA,
         const dolfinx::geometry::BoundingBoxTree<double>& treeB)
      {
        std::vector coll
            = dolfinx::geometry::compute_collisions<double>(treeA, treeB);
        std::array<std::size_t, 2> shape{std::size_t(coll.size() / 2), 2};
        return as_nbarray(std::move(coll), shape);
      },
      nb::arg("tree0"), nb::arg("tree1"));

  m.def(
      "compute_distance_gjk",
      [](const nb::ndarray<double>& p, const nb::ndarray<double>& q)
      {
        const std::size_t p_s0 = p.ndim() == 1 ? 1 : p.shape(0);
        const std::size_t q_s0 = q.ndim() == 1 ? 1 : q.shape(0);
        std::span<const double> _p(p.data(), 3 * p_s0);
        std::span<const double> _q(q.data(), 3 * q_s0);

        const std::array<double, 3> d
            = dolfinx::geometry::compute_distance_gjk<double>(_p, _q);
        const std::size_t size = 3;
        return nb::ndarray<const double>(d.data(), 1, &size);
      },
      nb::arg("p"), nb::arg("q"));

  m.def(
      "squared_distance",
      [](const dolfinx::mesh::Mesh<double>& mesh, int dim,
         std::vector<std::int32_t> indices, const nb::ndarray<double>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const double> _p(points.data(), 3 * p_s0);

        return as_nbarray(dolfinx::geometry::squared_distance<double>(
            mesh, dim, indices, _p));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("indices"), nb::arg("points"));
  m.def(
      "compute_colliding_cells",
      [](const dolfinx::mesh::Mesh<double>& mesh,
         const dolfinx::graph::AdjacencyList<int>& candidate_cells,
         const nb::ndarray<double>& points)
          -> std::variant<dolfinx::graph::AdjacencyList<std::int32_t>,
                          nb::ndarray<std::int32_t>>
      {
        const int gdim = mesh.geometry().dim();
        std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const double> _p(points.data(), 3 * p_s0);

        if (gdim > 1 and points.ndim() == 1)
        {
          // Single point in 2D/3D
          assert(points.shape(0) <= 3);
          auto cells = dolfinx::geometry::compute_colliding_cells<double>(
              mesh, candidate_cells, _p);
          const std::size_t size = cells.array().size();
          return nb::ndarray<std::int32_t>(cells.array().data(), 1, &size);
        }

        return dolfinx::geometry::compute_colliding_cells<double>(
            mesh, candidate_cells, _p);
      },
      nb::arg("mesh"), nb::arg("candidate_cells"), nb::arg("points"));

  // dolfinx::geometry::BoundingBoxTree
  nb::class_<dolfinx::geometry::BoundingBoxTree<double>>(m, "BoundingBoxTree")
      .def(
          "__init__",
          [](dolfinx::geometry::BoundingBoxTree<double>* bbt,
             const dolfinx::mesh::Mesh<double>& mesh, int dim,
             const nb::ndarray<std::int32_t, nb::numpy>& entities,
             double padding)
          {
            new (bbt) dolfinx::geometry::BoundingBoxTree(
                mesh, dim,
                std::span<const std::int32_t>(entities.data(), entities.size()),
                padding);
          },
          nb::arg("mesh"), nb::arg("dim"), nb::arg("entities"),
          nb::arg("padding"))
      .def_prop_ro("num_bboxes",
                   &dolfinx::geometry::BoundingBoxTree<double>::num_bboxes)
      .def(
          "get_bbox",
          [](const dolfinx::geometry::BoundingBoxTree<double>& self,
             const std::size_t i)
          {
            std::array<double, 6> bbox = self.get_bbox(i);
            std::array<std::size_t, 2> shape{2, 3};
            return nb::ndarray<double>(bbox.data(), 2, shape.data());
          },
          nb::arg("i"))
      .def("__repr__", &dolfinx::geometry::BoundingBoxTree<double>::str)
      .def(
          "create_global_tree",
          [](const dolfinx::geometry::BoundingBoxTree<double>& self,
             const MPICommWrapper comm)
          { return self.create_global_tree(comm.get()); },
          nb::arg("comm"));
}
} // namespace dolfinx_wrappers
