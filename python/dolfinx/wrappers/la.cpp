// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/span.hpp>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void la(py::module& m)
{
  // dolfinx::la::SparsityPattern
  py::class_<dolfinx::la::SparsityPattern,
             std::shared_ptr<dolfinx::la::SparsityPattern>>(m,
                                                            "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), maps, bs);
          }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             const std::array<std::vector<int>, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), patterns, maps, bs);
          }))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map)
      .def("assemble", &dolfinx::la::SparsityPattern::assemble)
      .def("num_nonzeros", &dolfinx::la::SparsityPattern::num_nonzeros)
      .def("insert", &dolfinx::la::SparsityPattern::insert)
      .def("insert_diagonal", &dolfinx::la::SparsityPattern::insert_diagonal)
      .def_property_readonly("diagonal_pattern",
                             &dolfinx::la::SparsityPattern::diagonal_pattern,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "off_diagonal_pattern",
          &dolfinx::la::SparsityPattern::off_diagonal_pattern,
          py::return_value_policy::reference_internal);

  // dolfinx::la::VectorSpaceBasis
  py::class_<dolfinx::la::VectorSpaceBasis,
             std::shared_ptr<dolfinx::la::VectorSpaceBasis>>(m,
                                                             "VectorSpaceBasis")
      .def(py::init([](const std::vector<Vec> x) {
        std::vector<std::shared_ptr<dolfinx::la::PETScVector>> _x;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
          assert(x[i]);
          _x.push_back(std::make_shared<dolfinx::la::PETScVector>(x[i], true));
        }
        return dolfinx::la::VectorSpaceBasis(_x);
      }))
      .def("is_orthonormal", &dolfinx::la::VectorSpaceBasis::is_orthonormal,
           py::arg("tol") = 1.0e-10)
      .def("is_orthogonal", &dolfinx::la::VectorSpaceBasis::is_orthogonal,
           py::arg("tol") = 1.0e-10)
      .def("in_nullspace", &dolfinx::la::VectorSpaceBasis::in_nullspace,
           py::arg("A"), py::arg("tol") = 1.0e-10)
      .def("orthogonalize", &dolfinx::la::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfinx::la::VectorSpaceBasis::orthonormalize,
           py::arg("tol") = 1.0e-10)
      .def("dim", &dolfinx::la::VectorSpaceBasis::dim)
      .def("__getitem__", [](const dolfinx::la::VectorSpaceBasis& self, int i) {
        return self[i]->vec();
      });

  // dolfinx::la::Vector
  py::class_<dolfinx::la::Vector<PetscScalar>,
             std::shared_ptr<dolfinx::la::Vector<PetscScalar>>>(m, "Vector")
      .def_property_readonly("array",
                             &dolfinx::la::Vector<PetscScalar>::mutable_array);

  // utils
  m.def("scatter_forward", &dolfinx::la::scatter_fwd<PetscScalar>);
  m.def("scatter_reverse", &dolfinx::la::scatter_rev<PetscScalar>);

  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::create_petsc_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_matrix",
      [](const MPICommWrapper comm, const dolfinx::la::SparsityPattern& p,
         const std::string& type) {
        return dolfinx::la::create_petsc_matrix(comm.get(), p, type);
      },
      py::return_value_policy::take_ownership, py::arg("comm"), py::arg("p"),
      py::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");
  // TODO: check reference counting for index sets
  m.def("create_petsc_index_sets", &dolfinx::la::create_petsc_index_sets,
        py::return_value_policy::take_ownership);
  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x_b,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps) {
        std::vector<tcb::span<const PetscScalar>> _x_b;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        dolfinx::la::scatter_local_vectors(x, _x_b, maps);
      },
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");
  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps) {
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::get_local_vectors(x, maps);
        std::vector<py::array> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(as_pyarray(std::move(v)));
        return ret;
      },
      "Gather an (ordered) list of sub vectors from a block vector.");
  // NOTE: Enabling the below requires adding a C API for MatNullSpace to
  // petsc4py
  //   m.def("create_nullspace",
  //         [](const MPICommWrapper comm, MPI_Comm comm,
  //            const dolfinx::la::VectorSpaceBasis& nullspace) {
  //           return dolfinx::la::create_petsc_nullspace(comm.get(),
  //           nullspace);
  //         },
  //         py::return_value_policy::take_ownership,
  //         "Create a PETSc MatNullSpace.");
}
} // namespace dolfinx_wrappers
