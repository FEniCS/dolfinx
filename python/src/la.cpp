// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casters.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/VectorSpaceBasis.h>
#include <dolfin/la/utils.h>

namespace py = pybind11;

namespace
{

// Linear operator trampoline class
template <typename LinearOperatorBase>
class PyLinearOperator : public LinearOperatorBase
{
  using LinearOperatorBase::LinearOperatorBase;

  // pybdind11 has some issues when passing by reference (due to the
  // return value policy), so the below is non-standard.  See
  // https://github.com/pybind/pybind11/issues/250.

  std::size_t size(std::size_t dim) const
  {
    PYBIND11_OVERLOAD_PURE(std::size_t, LinearOperatorBase, size, );
  }

  void mult(const dolfin::la::PETScVector& x, dolfin::la::PETScVector& y) const
  {
    PYBIND11_OVERLOAD_INT(void, LinearOperatorBase, "mult", &x, &y);
  }
};

// Linear operator trampoline class (with pure virtual 'mult' function)
template <typename LinearOperatorBase>
class PyLinearOperatorPure : public LinearOperatorBase
{
  using LinearOperatorBase::LinearOperatorBase;

  std::size_t size(std::size_t dim) const
  {
    PYBIND11_OVERLOAD_PURE(std::size_t, LinearOperatorBase, size, );
  }

  void mult(const dolfin::la::PETScVector& x, dolfin::la::PETScVector& y) const
  {
    PYBIND11_OVERLOAD_INT(void, LinearOperatorBase, "mult", &x, &y);
    py::pybind11_fail("Tried to call pure virtual function \'mult\'");
  }
};
} // namespace

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
      .def("apply", &dolfin::la::SparsityPattern::apply)
      .def("str", &dolfin::la::SparsityPattern::str)
      .def("num_nonzeros", &dolfin::la::SparsityPattern::num_nonzeros)
      .def("num_nonzeros_diagonal",
           &dolfin::la::SparsityPattern::num_nonzeros_diagonal)
      .def("num_nonzeros_off_diagonal",
           &dolfin::la::SparsityPattern::num_nonzeros_off_diagonal)
      .def("num_local_nonzeros",
           &dolfin::la::SparsityPattern::num_local_nonzeros)
      .def("insert_local", &dolfin::la::SparsityPattern::insert_local)
      .def("insert_global", &dolfin::la::SparsityPattern::insert_global)
      .def("insert_local_global",
           &dolfin::la::SparsityPattern::insert_local_global);

  py::class_<dolfin::la::PETScOptions>(m, "PETScOptions")
      .def_static("set",
                  (void (*)(std::string)) & dolfin::la::PETScOptions::set)
      .def_static("set",
                  (void (*)(std::string, bool)) & dolfin::la::PETScOptions::set)
      .def_static("set",
                  (void (*)(std::string, int)) & dolfin::la::PETScOptions::set)
      .def_static("set", (void (*)(std::string, double))
                             & dolfin::la::PETScOptions::set)
      .def_static("set", (void (*)(std::string, std::string))
                             & dolfin::la::PETScOptions::set)
      .def_static("clear",
                  (void (*)(std::string)) & dolfin::la::PETScOptions::clear)
      .def_static("clear", (void (*)()) & dolfin::la::PETScOptions::clear);

  // dolfin::la::PETScVector
  py::class_<dolfin::la::PETScVector, std::shared_ptr<dolfin::la::PETScVector>>(
      m, "PETScVector", py::dynamic_attr(), "PETScVector object")
      .def(py::init<>())
      .def(py::init<const dolfin::common::IndexMap&>())
      .def(py::init(
          [](const MPICommWrapper comm, std::array<std::int64_t, 2> range,
             const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
                 ghost_indices,
             int block_size) {
            return dolfin::la::PETScVector(comm.get(), range, ghost_indices,
                                           block_size);
          }))
      .def(py::init<Vec>())
      .def(py::init<const dolfin::la::PETScVector&>())
      .def("apply", &dolfin::la::PETScVector::apply)
      .def("apply_ghosts", &dolfin::la::PETScVector::apply_ghosts)
      .def("update_ghosts", &dolfin::la::PETScVector::update_ghosts)
      .def("get_options_prefix", &dolfin::la::PETScVector::get_options_prefix)
      .def("set_options_prefix", &dolfin::la::PETScVector::set_options_prefix)
      .def("vec", &dolfin::la::PETScVector::vec,
           "Return underlying PETSc Vec object");

  // dolfin::la::PETScOperator
  py::class_<dolfin::la::PETScOperator,
             std::shared_ptr<dolfin::la::PETScOperator>>(m, "PETScOperator")
      .def("mat", &dolfin::la::PETScOperator::mat,
           "Return underlying PETSc Mat object");

  // dolfin::la::PETScMatrix
  py::class_<dolfin::la::PETScMatrix, std::shared_ptr<dolfin::la::PETScMatrix>,
             dolfin::la::PETScOperator>(m, "PETScMatrix", py::dynamic_attr(),
                                        "PETScMatrix object")
      .def(py::init<>());

  // dolfin::la::PETScKrylovSolver
  py::class_<dolfin::la::PETScKrylovSolver,
             std::shared_ptr<dolfin::la::PETScKrylovSolver>>(
      m, "PETScKrylovSolver", "PETScKrylovSolver object")
      .def(py::init([](const MPICommWrapper comm) {
             return std::make_unique<dolfin::la::PETScKrylovSolver>(comm.get());
           }),
           py::arg("comm"))
      .def(py::init<KSP>())
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
      .def("set_reuse_preconditioner",
           &dolfin::la::PETScKrylovSolver::set_reuse_preconditioner)
      .def("set_dm", &dolfin::la::PETScKrylovSolver::set_dm)
      .def("set_dm_active", &dolfin::la::PETScKrylovSolver::set_dm_active)
      .def("ksp", &dolfin::la::PETScKrylovSolver::ksp);

  // dolfin::la::VectorSpaceBasis
  py::class_<dolfin::la::VectorSpaceBasis,
             std::shared_ptr<dolfin::la::VectorSpaceBasis>>(m,
                                                            "VectorSpaceBasis")
      .def(py::init<
           const std::vector<std::shared_ptr<dolfin::la::PETScVector>>>())
      .def("is_orthonormal", &dolfin::la::VectorSpaceBasis::is_orthonormal,
           py::arg("tol") = 1.0e-10)
      .def("is_orthogonal", &dolfin::la::VectorSpaceBasis::is_orthogonal,
           py::arg("tol") = 1.0e-10)
      .def("in_nullspace", &dolfin::la::VectorSpaceBasis::in_nullspace,
           py::arg("A"), py::arg("tol") = 1.0e-10)
      .def("orthogonalize", &dolfin::la::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfin::la::VectorSpaceBasis::orthonormalize,
           py::arg("tol") = 1.0e-10)
      .def("dim", &dolfin::la::VectorSpaceBasis::dim)
      .def("__getitem__", &dolfin::la::VectorSpaceBasis::operator[]);

  // utils
  m.def("create_vector",
        py::overload_cast<const dolfin::common::IndexMap&>(
            &dolfin::la::create_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def("create_matrix",
        [](const MPICommWrapper comm, const dolfin::la::SparsityPattern& p) {
          return dolfin::la::create_matrix(comm.get(), p);
        },
        py::return_value_policy::take_ownership,
        "Create a PETSc Mat from sparsity pattern.");
}
} // namespace dolfin_wrappers
