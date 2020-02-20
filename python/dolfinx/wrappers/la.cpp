// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
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
             std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
                 index_maps) {
            return dolfinx::la::SparsityPattern(comm.get(), index_maps);
          }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns) {
            return std::make_unique<dolfinx::la::SparsityPattern>(comm.get(),
                                                                  patterns);
          }))
      .def("local_range", &dolfinx::la::SparsityPattern::local_range)
      .def("index_map", &dolfinx::la::SparsityPattern::index_map)
      .def("assemble", &dolfinx::la::SparsityPattern::assemble)
      .def("str", &dolfinx::la::SparsityPattern::str)
      .def("num_nonzeros", &dolfinx::la::SparsityPattern::num_nonzeros)
      .def("num_nonzeros_diagonal",
           &dolfinx::la::SparsityPattern::num_nonzeros_diagonal)
      .def("num_nonzeros_off_diagonal",
           &dolfinx::la::SparsityPattern::num_nonzeros_off_diagonal)
      .def("num_local_nonzeros",
           &dolfinx::la::SparsityPattern::num_local_nonzeros)
      .def("insert", &dolfinx::la::SparsityPattern::insert)
      .def("insert_diagonal", &dolfinx::la::SparsityPattern::insert_diagonal);

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

  // utils
  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&>(
            &dolfinx::la::create_petsc_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_vector",
      [](const MPICommWrapper comm, std::array<std::int64_t, 2> range,
         const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghost_indices,
         int block_size) {
        return dolfinx::la::create_petsc_vector(comm.get(), range,
                                                ghost_indices, block_size);
      },
      py::return_value_policy::take_ownership, "Create a PETSc Vec.");
  m.def(
      "create_matrix",
      [](const MPICommWrapper comm, const dolfinx::la::SparsityPattern& p) {
        return dolfinx::la::create_petsc_matrix(comm.get(), p);
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat from sparsity pattern.");
  m.def("create_petsc_index_sets", &dolfinx::la::create_petsc_index_sets,
        py::return_value_policy::take_ownership);
  m.def("scatter_local_vectors", &dolfinx::la::scatter_local_vectors,
        "Scatter the (ordered) list of sub vectors into a block vector.");
  m.def("get_local_vectors", &dolfinx::la::get_local_vectors,
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
