// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void geometry(py::module& m)
{
  m.def("create_midpoint_tree",
        [](const dolfinx::mesh::Mesh& mesh, int tdim,
           const py::array_t<std::int32_t, py::array::c_style>& entities)
        {
          return dolfinx::geometry::create_midpoint_tree(
              mesh, tdim,
              xtl::span<const std::int32_t>(entities.data(), entities.size()));
        });

  m.def(
      "compute_closest_entity",
      [](const dolfinx::geometry::BoundingBoxTree& tree,
         const dolfinx::geometry::BoundingBoxTree& midpoint_tree,
         const dolfinx::mesh::Mesh& mesh,
         const py::array_t<double, py::array::c_style>& points)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        xt::xtensor<double, 2> p
            = xt::zeros<double>({p_s0, static_cast<std::size_t>(3)});
        auto px = points.unchecked();
        if (px.ndim() == 1)
        {
          for (py::ssize_t i = 0; i < px.shape(0); i++)
            p(0, i) = px(i);
        }
        else if (px.ndim() == 2)
        {
          for (py::ssize_t i = 0; i < px.shape(0); i++)
            for (py::ssize_t j = 0; j < px.shape(1); j++)
              p(i, j) = px(i, j);
        }
        else
          throw std::runtime_error("Array has wrong ndim.");

        return as_pyarray(dolfinx::geometry::compute_closest_entity(
            tree, midpoint_tree, mesh, p));
      },
      py::arg("tree"), py::arg("midpoint_tree"), py::arg("mesh"),
      py::arg("points"));

  m.def("compute_collisions",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const py::array_t<double>& points)
        {
          const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          xt::xtensor<double, 2> _p
              = xt::zeros<double>({p_s0, static_cast<std::size_t>(3)});
          auto px = points.unchecked();
          if (px.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(0, i) = px(i);
          }
          else if (px.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              for (py::ssize_t j = 0; j < px.shape(1); j++)
                _p(i, j) = px(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          return dolfinx::geometry::compute_collisions(tree, _p);
        });
  m.def("compute_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const dolfinx::geometry::BoundingBoxTree&>(
            &dolfinx::geometry::compute_collisions));

  m.def("compute_distance_gjk",
        [](const py::array_t<double>& p, const py::array_t<double>& q)
        {
          const std::size_t p_s0 = p.ndim() == 1 ? 1 : p.shape(0);
          const std::size_t q_s0 = q.ndim() == 1 ? 1 : q.shape(0);
          xt::xtensor<double, 2> _p
              = xt::zeros<double>({p_s0, static_cast<std::size_t>(3)});
          xt::xtensor<double, 2> _q
              = xt::zeros<double>({q_s0, static_cast<std::size_t>(3)});

          auto px = p.unchecked();
          if (px.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(0, i) = px(i);
          }
          else if (px.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              for (py::ssize_t j = 0; j < px.shape(1); j++)
                _p(i, j) = px(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          auto qx = q.unchecked();
          if (qx.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < qx.shape(0); i++)
              _q(0, i) = qx(i);
          }
          else if (qx.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < qx.shape(0); i++)
              for (py::ssize_t j = 0; j < qx.shape(1); j++)
                _q(i, j) = qx(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          const xt::xtensor_fixed<double, xt::xshape<3>> d
              = dolfinx::geometry::compute_distance_gjk(_p, _q);
          return py::array_t<double>(d.shape(), d.data());
        });

  m.def("squared_distance",
        [](const dolfinx::mesh::Mesh& mesh, int dim,
           std::vector<std::int32_t> indices, const py::array_t<double>& points)
        {
          const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          xt::xtensor<double, 2> _p
              = xt::zeros<double>({p_s0, static_cast<std::size_t>(3)});
          auto px = points.unchecked();
          if (px.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(0, i) = px(i);
          }
          else if (px.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              for (py::ssize_t j = 0; j < px.shape(1); j++)
                _p(i, j) = px(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          return xt_as_pyarray(
              dolfinx::geometry::squared_distance(mesh, dim, indices, _p));
        });
  m.def("compute_colliding_cells",
        [](const dolfinx::mesh::Mesh& mesh,
           const dolfinx::graph::AdjacencyList<int>& candidate_cells,
           const py::array_t<double>& points)
            -> std::variant<dolfinx::graph::AdjacencyList<std::int32_t>,
                            py::array_t<std::int32_t>>
        {
          const int gdim = mesh.geometry().dim();
          std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
          xt::xtensor<double, 2> _p = xt::zeros<double>({p_s0, std::size_t(3)});
          auto px = points.unchecked();
          if (gdim > 1 and px.ndim() == 1)
          {
            // Single point in 2D/3D
            assert(px.shape(0) <= 3);
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(0, i) = px(i);
            auto cells = dolfinx::geometry::compute_colliding_cells(
                mesh, candidate_cells, _p);
            return py::array_t<std::int32_t>(cells.array().size(),
                                             cells.array().data());
          }
          else if (gdim == 1 and px.ndim() == 1)
          {
            // 1D problem
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(i, 0) = px(i);
          }
          else if (px.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              for (py::ssize_t j = 0; j < px.shape(1); j++)
                _p(i, j) = px(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          return dolfinx::geometry::compute_colliding_cells(
              mesh, candidate_cells, _p);
        });

  // dolfinx::geometry::BoundingBoxTree
  py::class_<dolfinx::geometry::BoundingBoxTree,
             std::shared_ptr<dolfinx::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init(
               [](const dolfinx::mesh::Mesh& mesh, int dim,
                  const py::array_t<std::int32_t, py::array::c_style>& entities,
                  double padding)
               {
                 return dolfinx::geometry::BoundingBoxTree(
                     mesh, dim,
                     xtl::span<const std::int32_t>(entities.data(),
                                                   entities.size()),
                     padding);
               }),
           py::arg("mesh"), py::arg("dim"), py::arg("entities"),
           py::arg("padding"))
      .def_property_readonly("num_bboxes",
                             &dolfinx::geometry::BoundingBoxTree::num_bboxes)
      .def("get_bbox",
           [](const dolfinx::geometry::BoundingBoxTree& self,
              const std::size_t i)
           {
             xt::xtensor<double, 2> bbox = self.get_bbox(i);
             return xt_as_pyarray(std::move(bbox));
           })
      .def("__repr__", &dolfinx::geometry::BoundingBoxTree::str)
      .def("create_global_tree",
           [](const dolfinx::geometry::BoundingBoxTree& self,
              const MPICommWrapper comm)
           { return self.create_global_tree(comm.get()); });
}
} // namespace dolfinx_wrappers
