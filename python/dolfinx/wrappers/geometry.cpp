// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <array>
#include <dolfinx/common/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
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
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 entities,
             double padding)
          {
            std::optional<std::span<const std::int32_t>> ents
                = entities ? std::span<const std::int32_t>(
                                 entities->data(),
                                 entities->data() + entities->size())
                           : std::optional<std::span<const std::int32_t>>(
                                 std::nullopt);

            new (bbt)
                dolfinx::geometry::BoundingBoxTree<T>(mesh, dim, ents, padding);
          },
          nb::arg("mesh"), nb::arg("dim"), nb::arg("entities").none(),
          nb::arg("padding") = 0.0)
      .def_prop_ro("num_bboxes",
                   &dolfinx::geometry::BoundingBoxTree<T>::num_bboxes)
      .def(
          "get_bbox",
          [](const dolfinx::geometry::BoundingBoxTree<T>& self,
             const std::size_t i)
          {
            std::array<T, 6> bbox = self.get_bbox(i);
            return nb::ndarray<T, nb::shape<2, 3>, nb::numpy>(bbox.data())
                .cast();
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
         nb::ndarray<const T, nb::shape<3>, nb::c_contig> points)
      {
        return dolfinx::geometry::compute_collisions<T>(
            tree, std::span(points.data(), 3));
      },
      nb::arg("tree"), nb::arg("points"));
  m.def(
      "compute_collisions_points",
      [](const dolfinx::geometry::BoundingBoxTree<T>& tree,
         nb::ndarray<const T, nb::shape<-1, 3>, nb::c_contig> points)
      {
        return dolfinx::geometry::compute_collisions<T>(
            tree, std::span(points.data(), points.size()));
      },
      nb::arg("tree"), nb::arg("points"));
  m.def(
      "compute_collisions_trees",
      [](const dolfinx::geometry::BoundingBoxTree<T>& treeA,
         const dolfinx::geometry::BoundingBoxTree<T>& treeB)
      {
        std::vector coll
            = dolfinx::geometry::compute_collisions<T>(treeA, treeB);
        return dolfinx_wrappers::as_nbarray(std::move(coll),
                                            {coll.size() / 2, 2});
      },
      nb::arg("tree0"), nb::arg("tree1"));
  m.def(
      "compute_closest_entity",
      [](const dolfinx::geometry::BoundingBoxTree<T>& tree,
         const dolfinx::geometry::BoundingBoxTree<T>& midpoint_tree,
         const dolfinx::mesh::Mesh<T>& mesh,
         nb::ndarray<const T, nb::shape<3>, nb::c_contig> points)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::geometry::compute_closest_entity<T>(
                tree, midpoint_tree, mesh,
                std::span(points.data(), points.size())));
      },
      nb::arg("tree"), nb::arg("midpoint_tree"), nb::arg("mesh"),
      nb::arg("points"));
  m.def(
      "compute_closest_entity",
      [](const dolfinx::geometry::BoundingBoxTree<T>& tree,
         const dolfinx::geometry::BoundingBoxTree<T>& midpoint_tree,
         const dolfinx::mesh::Mesh<T>& mesh,
         nb::ndarray<const T, nb::shape<-1, 3>, nb::c_contig> points)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::geometry::compute_closest_entity<T>(
                tree, midpoint_tree, mesh,
                std::span(points.data(), points.size())));
      },
      nb::arg("tree"), nb::arg("midpoint_tree"), nb::arg("mesh"),
      nb::arg("points"));
  m.def(
      "create_midpoint_tree",
      [](const dolfinx::mesh::Mesh<T>& mesh, int tdim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
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
         nb::ndarray<const T, nb::shape<3>, nb::c_contig> points)
      {
        return dolfinx::geometry::compute_colliding_cells<T>(
            mesh, candidate_cells, std::span(points.data(), points.size()));
      },
      nb::arg("mesh"), nb::arg("candidate_cells"), nb::arg("points"));
  m.def(
      "compute_colliding_cells",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         const dolfinx::graph::AdjacencyList<int>& candidate_cells,
         nb::ndarray<const T, nb::shape<-1, 3>, nb::c_contig> points)
      {
        return dolfinx::geometry::compute_colliding_cells<T>(
            mesh, candidate_cells, std::span(points.data(), points.size()));
      },
      nb::arg("mesh"), nb::arg("candidate_cells"), nb::arg("points"));

  m.def(
      "compute_distance_gjk",
      [](nb::ndarray<const T, nb::c_contig> p,
         nb::ndarray<const T, nb::c_contig> q)
      {
        std::size_t p_s0 = p.ndim() == 1 ? 1 : p.shape(0);
        std::size_t q_s0 = q.ndim() == 1 ? 1 : q.shape(0);
        std::span<const T> _p(p.data(), 3 * p_s0), _q(q.data(), 3 * q_s0);
        std::array<T, 3> d = dolfinx::geometry::compute_distance_gjk<T>(_p, _q);
        return nb::ndarray<T, nb::numpy>(d.data(), {d.size()}).cast();
      },
      //   nb::rv_policy::copy,
      nb::arg("p"), nb::arg("q"));

  m.def(
      "squared_distance",
      [](const dolfinx::mesh::Mesh<T>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indices,
         nb::ndarray<const T, nb::c_contig> points)
      {
        std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);
        return dolfinx_wrappers::as_nbarray(
            dolfinx::geometry::squared_distance<T>(
                mesh, dim, std::span(indices.data(), indices.size()), _p));
      },
      nb::arg("mesh"), nb::arg("dim"), nb::arg("indices"), nb::arg("points"));
  m.def("determine_point_ownership",
        [](const dolfinx::mesh::Mesh<T>& mesh,
           nb::ndarray<const T, nb::c_contig> points, const T padding)
        {
          std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          std::span<const T> _p(points.data(), 3 * p_s0);
          return dolfinx::geometry::determine_point_ownership<T>(mesh, _p,
                                                                 padding);
        });

  std::string pod_pyclass_name = "PointOwnershipData_" + type;
  nb::class_<dolfinx::geometry::PointOwnershipData<T>>(m,
                                                       pod_pyclass_name.c_str())
      .def(
          "__init__",
          [](dolfinx::geometry::PointOwnershipData<T>* self,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
                 src_owner,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
                 dest_owners,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> dest_points,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
                 dest_cells)
          {
            new (self) dolfinx::geometry::PointOwnershipData<T>{
                .src_owner = std::vector(src_owner.data(),
                                         src_owner.data() + src_owner.size()),
                .dest_owners
                = std::vector(dest_owners.data(),
                              dest_owners.data() + dest_owners.size()),
                .dest_points
                = std::vector(dest_points.data(),
                              dest_points.data() + dest_points.size()),
                .dest_cells = std::vector(
                    dest_cells.data(), dest_cells.data() + dest_cells.size())};
          },
          nb::arg("src_owner"), nb::arg("dest_owners"), nb::arg("dest_points"),
          nb::arg("dest_cells"))
      .def_prop_ro(
          "src_owner",
          [](const dolfinx::geometry::PointOwnershipData<T>& self)
          {
            return nb::ndarray<const int, nb::numpy>(self.src_owner.data(),
                                                     {self.src_owner.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "dest_owners",
          [](const dolfinx::geometry::PointOwnershipData<T>& self)
          {
            return nb::ndarray<const int, nb::numpy>(self.dest_owners.data(),
                                                     {self.dest_owners.size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "dest_points",
          [](const dolfinx::geometry::PointOwnershipData<T>& self)
          {
            return nb::ndarray<const T, nb::shape<-1, 3>, nb::numpy>(
                self.dest_points.data(), {self.dest_points.size() / 3, 3});
          },
          nb::rv_policy::reference_internal, "Destination point")
      .def_prop_ro(
          "dest_cells",
          [](const dolfinx::geometry::PointOwnershipData<T>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.dest_cells.data(), {self.dest_cells.size()});
          },
          nb::rv_policy::reference_internal);
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
