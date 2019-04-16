// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/VectorSpaceBasis.h>
#include <dolfin/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

void la(py::module& m)
{
  // dolfin::la::SparsityPattern
  py::class_<dolfin::la::SparsityPattern,
             std::shared_ptr<dolfin::la::SparsityPattern>>(m, "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             std::array<std::shared_ptr<const dolfin::common::IndexMap>, 2>
                 index_maps) {
            return dolfin::la::SparsityPattern(comm.get(), index_maps);
          }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfin::la::SparsityPattern*>>
                 patterns) {
            return std::make_unique<dolfin::la::SparsityPattern>(comm.get(),
                                                                 patterns);
          }))
      .def("local_range", &dolfin::la::SparsityPattern::local_range)
      .def("index_map", &dolfin::la::SparsityPattern::index_map)
      .def("assemble", &dolfin::la::SparsityPattern::assemble)
      .def("str", &dolfin::la::SparsityPattern::str)
      .def("num_nonzeros", &dolfin::la::SparsityPattern::num_nonzeros)
      .def("num_nonzeros_diagonal",
           &dolfin::la::SparsityPattern::num_nonzeros_diagonal)
      .def("num_nonzeros_off_diagonal",
           &dolfin::la::SparsityPattern::num_nonzeros_off_diagonal)
      .def("num_local_nonzeros",
           &dolfin::la::SparsityPattern::num_local_nonzeros)
      .def("insert_local", &dolfin::la::SparsityPattern::insert_local)
      .def("insert_global", &dolfin::la::SparsityPattern::insert_global);

  py::class_<dolfin::la::PETScOptions>(m, "PETScOptions")
      .def_static("set",
                  (void (*)(std::string)) & dolfin::la::PETScOptions::set)
      .def_static("set", &dolfin::la::PETScOptions::set<bool>)
      .def_static("set", &dolfin::la::PETScOptions::set<int>)
      .def_static("set", &dolfin::la::PETScOptions::set<double>)
      .def_static("set", &dolfin::la::PETScOptions::set<std::string>)
      .def_static("clear", py::overload_cast<std::string>(
                               &dolfin::la::PETScOptions::clear))
      .def_static("clear",
                  py::overload_cast<>(&dolfin::la::PETScOptions::clear));

  // dolfin::la::PETScKrylovSolver
  py::class_<dolfin::la::PETScKrylovSolver,
             std::shared_ptr<dolfin::la::PETScKrylovSolver>>(
      m, "PETScKrylovSolver", "PETScKrylovSolver object")
      .def(py::init([](const MPICommWrapper comm) {
             return std::make_unique<dolfin::la::PETScKrylovSolver>(comm.get());
           }),
           py::arg("comm"))
      .def(py::init<KSP, bool>(), py::arg("comm"),
           py::arg("inc_ref_count") = true)
      .def("get_options_prefix",
           &dolfin::la::PETScKrylovSolver::get_options_prefix)
      .def("set_options_prefix",
           &dolfin::la::PETScKrylovSolver::set_options_prefix)
      .def("set_operator", &dolfin::la::PETScKrylovSolver::set_operator)
      .def("set_operators", &dolfin::la::PETScKrylovSolver::set_operators)
      .def("solve", &dolfin::la::PETScKrylovSolver::solve,
           "Solve linear system", py::arg("x"), py::arg("b"),
           py::arg("transpose") = false)
      .def("set_from_options", &dolfin::la::PETScKrylovSolver::set_from_options)
      .def("set_dm", &dolfin::la::PETScKrylovSolver::set_dm)
      .def("set_dm_active", &dolfin::la::PETScKrylovSolver::set_dm_active)
      .def("ksp", &dolfin::la::PETScKrylovSolver::ksp);

  // dolfin::la::VectorSpaceBasis
  py::class_<dolfin::la::VectorSpaceBasis,
             std::shared_ptr<dolfin::la::VectorSpaceBasis>>(m,
                                                            "VectorSpaceBasis")
      .def(py::init([](const std::vector<Vec> x) {
        std::vector<std::shared_ptr<dolfin::la::PETScVector>> _x;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
          assert(x[i]);
          _x.push_back(std::make_shared<dolfin::la::PETScVector>(x[i]));
        }
        return dolfin::la::VectorSpaceBasis(_x);
      }))
      .def("is_orthonormal", &dolfin::la::VectorSpaceBasis::is_orthonormal,
           py::arg("tol") = 1.0e-10)
      .def("is_orthogonal", &dolfin::la::VectorSpaceBasis::is_orthogonal,
           py::arg("tol") = 1.0e-10)
      .def("in_nullspace",
           [](const dolfin::la::VectorSpaceBasis& self, Mat A, double tol) {
             dolfin::la::PETScMatrix _A(A);
             return self.in_nullspace(_A, tol);
           },
           py::arg("A"), py::arg("tol") = 1.0e-10)
      .def("orthogonalize", &dolfin::la::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfin::la::VectorSpaceBasis::orthonormalize,
           py::arg("tol") = 1.0e-10)
      .def("dim", &dolfin::la::VectorSpaceBasis::dim)
      .def("__getitem__", [](const dolfin::la::VectorSpaceBasis& self, int i) {
        return self[i]->vec();
      });

  // utils
  m.def("create_vector",
        py::overload_cast<const dolfin::common::IndexMap&>(
            &dolfin::la::create_petsc_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def("create_vector",
        [](const MPICommWrapper comm, std::array<std::int64_t, 2> range,
           const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
           int block_size) {
          return dolfin::la::create_petsc_vector(comm.get(), range,
                                                 ghost_indices, block_size);
        },
        py::return_value_policy::take_ownership, "Create a PETSc Vec.");
  m.def("create_matrix",
        [](const MPICommWrapper comm, const dolfin::la::SparsityPattern& p) {
          return dolfin::la::create_petsc_matrix(comm.get(), p);
        },
        py::return_value_policy::take_ownership,
        "Create a PETSc Mat from sparsity pattern.");
  // NOTE: Enabling the below requires adding a C API for MatNullSpace to
  // petsc4py
  //   m.def("create_nullspace",
  //         [](const MPICommWrapper comm, MPI_Comm comm,
  //            const dolfin::la::VectorSpaceBasis& nullspace) {
  //           return dolfin::la::create_petsc_nullspace(comm.get(), nullspace);
  //         },
  //         py::return_value_policy::take_ownership,
  //         "Create a PETSc MatNullSpace.");
}
} // namespace dolfin_wrappers
