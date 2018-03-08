// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/common/Variable.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/OptimisationProblem.h>
#include <dolfin/parameter/Parameters.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers {
void nls(py::module &m) {
  // dolfin::NewtonSolver 'trampoline' for overloading virtual
  // functions from Python
  class PyNewtonSolver : public dolfin::nls::NewtonSolver {
    using dolfin::nls::NewtonSolver::NewtonSolver;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    bool converged(const dolfin::la::PETScVector &r,
                   const dolfin::nls::NonlinearProblem &nonlinear_problem,
                   std::size_t iteration) {
      PYBIND11_OVERLOAD_INT(bool, dolfin::nls::NewtonSolver, "converged", &r,
                            &nonlinear_problem, iteration);
      return dolfin::nls::NewtonSolver::converged(r, nonlinear_problem,
                                                  iteration);
    }

    void solver_setup(std::shared_ptr<const dolfin::la::PETScMatrix> A,
                      std::shared_ptr<const dolfin::la::PETScMatrix> P,
                      const dolfin::nls::NonlinearProblem &nonlinear_problem,
                      std::size_t iteration) {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NewtonSolver, "solver_setup", A,
                            P, &nonlinear_problem, iteration);
      return dolfin::nls::NewtonSolver::solver_setup(A, P, nonlinear_problem,
                                                     iteration);
    }

    void update_solution(dolfin::la::PETScVector &x,
                         const dolfin::la::PETScVector &dx,
                         double relaxation_parameter,
                         const dolfin::nls::NonlinearProblem &nonlinear_problem,
                         std::size_t iteration) {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NewtonSolver, "update_solution",
                            &x, &dx, relaxation_parameter, nonlinear_problem,
                            iteration);
      return dolfin::nls::NewtonSolver::update_solution(
          x, dx, relaxation_parameter, nonlinear_problem, iteration);
    }
  };

  // Class used to expose protected dolfin::NewtonSolver members
  // (see https://github.com/pybind/pybind11/issues/991)
  class PyPublicNewtonSolver : public dolfin::nls::NewtonSolver {
  public:
    using dolfin::nls::NewtonSolver::converged;
    using dolfin::nls::NewtonSolver::solver_setup;
    using dolfin::nls::NewtonSolver::update_solution;
  };

  // dolfin::NewtonSolver
  py::class_<dolfin::nls::NewtonSolver,
             std::shared_ptr<dolfin::nls::NewtonSolver>, PyNewtonSolver,
             dolfin::common::Variable>(m, "NewtonSolver")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfin::nls::NewtonSolver>(comm.get());
      }))
      .def("solve", &dolfin::nls::NewtonSolver::solve)
      .def("converged", &PyPublicNewtonSolver::converged)
      .def("solver_setup", &PyPublicNewtonSolver::solver_setup)
      .def("update_solution", &PyPublicNewtonSolver::update_solution);

  // dolfin::NonlinearProblem 'trampoline' for overloading from
  // Python
  class PyNonlinearProblem : public dolfin::nls::NonlinearProblem {
    using dolfin::nls::NonlinearProblem::NonlinearProblem;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    void J(dolfin::la::PETScMatrix &A,
           const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NonlinearProblem, "J", &A, &x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::OptimisationProblem::J");
    }

    void F(dolfin::la::PETScVector &b,
           const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NonlinearProblem, "F", &b, &x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::OptimisationProblem::F");
    }

    void form(dolfin::la::PETScMatrix &A, dolfin::la::PETScMatrix &P,
              dolfin::la::PETScVector &b,
              const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NonlinearProblem, "form", &A, &P,
                            &b, &x);
      return dolfin::nls::NonlinearProblem::form(A, P, b, x);
    }
  };

  // dolfin::NonlinearProblem
  py::class_<dolfin::nls::NonlinearProblem,
             std::shared_ptr<dolfin::nls::NonlinearProblem>,
             PyNonlinearProblem>(m, "NonlinearProblem")
      .def(py::init<>())
      .def("F", &dolfin::nls::NonlinearProblem::F)
      .def("J", &dolfin::nls::NonlinearProblem::J)
      .def("form",
           (void (dolfin::nls::NonlinearProblem::*)(
               dolfin::la::PETScMatrix &, dolfin::la::PETScMatrix &,
               dolfin::la::PETScVector &, const dolfin::la::PETScVector &)) &
               dolfin::nls::NonlinearProblem::form);

  // dolfin::OptimizationProblem 'trampoline' for overloading from
  // Python
  class PyOptimisationProblem : public dolfin::nls::OptimisationProblem {
    using dolfin::nls::OptimisationProblem::OptimisationProblem;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    double f(const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(double, dolfin::nls::OptimisationProblem, "f", &x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::OptimisationProblem::f");
    }

    void F(dolfin::la::PETScVector &b,
           const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::OptimisationProblem, "F", &b,
                            &x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::OptimisationProblem::F");
    }

    void J(dolfin::la::PETScMatrix &A,
           const dolfin::la::PETScVector &x) override {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::OptimisationProblem, "J", &A,
                            &x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::OptimisationProblem::J");
    }
  };

  // dolfin::OptimizationProblem
  py::class_<dolfin::nls::OptimisationProblem,
             std::shared_ptr<dolfin::nls::OptimisationProblem>,
             PyOptimisationProblem>(m, "OptimisationProblem")
      .def(py::init<>())
      .def("f", &dolfin::nls::OptimisationProblem::f)
      .def("F", &dolfin::nls::OptimisationProblem::F)
      .def("J", &dolfin::nls::OptimisationProblem::J);
}
}
