// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef HAS_PYBIND11_PETSC4PY
#include <petsc4py/petsc4py.h>
#endif

#include "casters.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/VectorSpaceBasis.h>

namespace py = pybind11;

namespace
{
template <typename T>
void check_indices(const py::array_t<T>& x, std::int64_t local_size)
{
  for (std::int64_t i = 0; i < (std::int64_t)x.size(); ++i)
  {
    std::int64_t _x = *(x.data() + i);
    if (_x < 0 or !(_x < local_size))
      throw py::index_error("Vector index out of range");
  }
}

// Linear operator trampoline class
template <typename LinearOperatorBase>
class PyLinearOperator : public LinearOperatorBase
{
  using LinearOperatorBase::LinearOperatorBase;

  // pybdind11 has some issues when passing by reference (due to
  // the return value policy), so the below is non-standard.  See
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

// Linear operator trampoline class (with pure virtual 'mult'
// function)
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
using RowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void la(py::module& m)
{
  // dolfin::la::SparsityPattern
  py::class_<dolfin::la::SparsityPattern,
             std::shared_ptr<dolfin::la::SparsityPattern>>(m, "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             std::array<std::shared_ptr<const dolfin::common::IndexMap>, 2>
                 index_maps) {
            return std::make_unique<dolfin::la::SparsityPattern>(comm.get(),
                                                                 index_maps);
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

  // dolfin::la::Scalar
  py::class_<dolfin::la::Scalar, std::shared_ptr<dolfin::la::Scalar>>(m,
                                                                      "Scalar")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfin::la::Scalar>(comm.get());
      }))
      .def("add", &dolfin::la::Scalar::add)
      .def("apply", &dolfin::la::Scalar::apply)
      .def("mpi_comm",
           [](dolfin::la::Scalar& self) {
             return MPICommWrapper(self.mpi_comm());
           })
      .def("value", &dolfin::la::Scalar::value);

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
      m, "PETScVector", "DOLFIN PETScVector object")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfin::la::PETScVector>(comm.get());
      }))
      .def(py::init<Vec>())
      .def(py::init<const dolfin::la::PETScVector>())
      .def("apply", &dolfin::la::PETScVector::apply)
      .def("norm", &dolfin::la::PETScVector::norm)
      .def("get_options_prefix", &dolfin::la::PETScVector::get_options_prefix)
      .def("set_options_prefix", &dolfin::la::PETScVector::set_options_prefix)
      .def("update_ghost_values", &dolfin::la::PETScVector::update_ghost_values)
      .def("size", (std::size_t(dolfin::la::PETScVector::*)() const)
                       & dolfin::la::PETScVector::size)
      .def("__add__",
           [](const dolfin::la::PETScVector& self,
              const dolfin::la::PETScVector& x) {
             auto y = std::make_shared<dolfin::la::PETScVector>(self);
             *y += x;
             return y;
           },
           py::is_operator())
      .def("__sub__",
           [](dolfin::la::PETScVector& self, const dolfin::la::PETScVector& x) {
             auto y = std::make_shared<dolfin::la::PETScVector>(self);
             *y -= x;
             return y;
           },
           py::is_operator())
      .def("get_local",
           [](const dolfin::la::PETScVector& self) {
             std::vector<double> values;
             self.get_local(values);
             return py::array_t<double>(values.size(), values.data());
           })
      .def(
          "__setitem__",
          [](dolfin::la::PETScVector& self, py::slice slice, double value) {
            std::size_t start, stop, step, slicelength;
            if (!slice.compute(self.size(), &start, &stop, &step, &slicelength))
              throw py::error_already_set();
            if (start != 0 or stop != (std::size_t)self.size() or step != 1)
              throw std::range_error(
                  "Only setting full slices for GenericVector is supported");

            self = value;
          })
      .def(
          "__setitem__",
          [](dolfin::la::PETScVector& self, py::slice slice,
             const py::array_t<double> x) {
            if (x.ndim() != 1)
              throw py::index_error("Values to set must be a 1D array");

            std::size_t start, stop, step, slicelength;
            if (!slice.compute(self.size(), &start, &stop, &step, &slicelength))
              throw py::error_already_set();
            if (start != 0 or stop != (std::size_t)self.size() or step != 1)
              throw std::range_error("Only full slices are supported");

            Eigen::Map<const dolfin::EigenArrayXd> values(x.data(), x.size());
            if (values.size() > 0)
            {
              self.set_local(values);
              self.apply();
            }
          })
      .def("vec", &dolfin::la::PETScVector::vec,
           "Return underlying PETSc Vec object");

  // dolfin::la::PETScBaseMatrix
  py::class_<dolfin::la::PETScBaseMatrix,
             std::shared_ptr<dolfin::la::PETScBaseMatrix>,
             dolfin::common::Variable>(m, "PETScBaseMatrix")
      .def("size",
           (std::int64_t(dolfin::la::PETScBaseMatrix::*)(std::size_t) const)
               & dolfin::la::PETScBaseMatrix::size)
      .def("mat", &dolfin::la::PETScBaseMatrix::mat,
           "Return underlying PETSc Mat object");

  // dolfin::la::PETScMatrix
  py::class_<dolfin::la::PETScMatrix, std::shared_ptr<dolfin::la::PETScMatrix>,
             dolfin::la::PETScBaseMatrix>(m, "PETScMatrix",
                                          "DOLFIN PETScMatrix object")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfin::la::PETScMatrix>(comm.get());
      }))
      .def(py::init<Mat>())
      .def("norm", &dolfin::la::PETScMatrix::norm)
      .def("get_options_prefix", &dolfin::la::PETScMatrix::get_options_prefix)
      .def("set_options_prefix", &dolfin::la::PETScMatrix::set_options_prefix)
      .def("set_nullspace", &dolfin::la::PETScMatrix::set_nullspace)
      .def("set_near_nullspace", &dolfin::la::PETScMatrix::set_near_nullspace);
  /*
  .def("__sub__",
       [](const dolfin::la::PETScMatrix& self, const dolfin::la::PETScMatrix& B)
  { dolfin::la::PETScMatrix C(self); C -= B; return C;
       },
       py::is_operator());
  */

  // dolfin::la::PETScKrylovSolver
  py::class_<dolfin::la::PETScKrylovSolver,
             std::shared_ptr<dolfin::la::PETScKrylovSolver>>
      petsc_ks(m, "PETScKrylovSolver", "DOLFIN PETScKrylovSolver object");

  petsc_ks
      .def(py::init([](const MPICommWrapper comm) {
             return std::unique_ptr<dolfin::la::PETScKrylovSolver>(
                 new dolfin::la::PETScKrylovSolver(comm.get()));
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

#ifdef HAS_SLEPC
  // dolfin::la::SLEPcEigenSolver
  py::class_<dolfin::la::SLEPcEigenSolver,
             std::shared_ptr<dolfin::la::SLEPcEigenSolver>,
             dolfin::common::Variable>(m, "SLEPcEigenSolver")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfin::la::SLEPcEigenSolver>(comm.get());
      }))
      .def("set_options_prefix",
           &dolfin::la::SLEPcEigenSolver::set_options_prefix)
      .def("set_from_options", &dolfin::la::SLEPcEigenSolver::set_from_options)
      .def("set_operators", &dolfin::la::SLEPcEigenSolver::set_operators)
      .def("get_options_prefix",
           &dolfin::la::SLEPcEigenSolver::get_options_prefix)
      .def("get_number_converged",
           &dolfin::la::SLEPcEigenSolver::get_number_converged)
      .def("set_deflation_space",
           &dolfin::la::SLEPcEigenSolver::set_deflation_space)
      .def("set_initial_space",
           &dolfin::la::SLEPcEigenSolver::set_initial_space)
      .def("solve", (void (dolfin::la::SLEPcEigenSolver::*)())
                        & dolfin::la::SLEPcEigenSolver::solve)
      .def("solve", (void (dolfin::la::SLEPcEigenSolver::*)(std::int64_t))
                        & dolfin::la::SLEPcEigenSolver::solve)
      .def("get_eigenvalue",
           [](dolfin::la::SLEPcEigenSolver& self, std::size_t i) {
             double lr, lc;
             self.get_eigenvalue(lr, lc, i);
             return py::make_tuple(lr, lc);
           })
      .def("get_eigenpair",
           [](dolfin::la::SLEPcEigenSolver& self, std::size_t i) {
             double lr, lc;
             dolfin::la::PETScVector r(self.mpi_comm()), c(self.mpi_comm());
             self.get_eigenpair(lr, lc, r, c, i);
             return py::make_tuple(lr, lc, r, c);
           });
#endif

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
      .def("orthogonalize", &dolfin::la::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfin::la::VectorSpaceBasis::orthonormalize,
           py::arg("tol") = 1.0e-10)
      .def("dim", &dolfin::la::VectorSpaceBasis::dim)
      .def("__getitem__", &dolfin::la::VectorSpaceBasis::operator[]);
}
} // namespace dolfin_wrappers
