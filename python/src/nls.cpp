// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/Variable.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
void nls(py::module& m)
{
  // dolfin::NewtonSolver 'trampoline' for overloading virtual
  // functions from Python
  class PyNewtonSolver : public dolfin::nls::NewtonSolver
  {
    using dolfin::nls::NewtonSolver::NewtonSolver;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    bool converged(const Vec r,
                   const dolfin::nls::NonlinearProblem& nonlinear_problem,
                   std::size_t iteration)
    {
      PYBIND11_OVERLOAD_INT(bool, dolfin::nls::NewtonSolver, "converged", &r,
                            &nonlinear_problem, iteration);
      return dolfin::nls::NewtonSolver::converged(r, nonlinear_problem,
                                                  iteration);
    }
  };

  // Class used to expose protected dolfin::NewtonSolver members
  // (see https://github.com/pybind/pybind11/issues/991)
  class PyPublicNewtonSolver : public dolfin::nls::NewtonSolver
  {
  public:
    using dolfin::nls::NewtonSolver::converged;
    // using dolfin::nls::NewtonSolver::solver_setup;
    using dolfin::nls::NewtonSolver::update_solution;
  };

  // dolfin::NewtonSolver
  py::class_<dolfin::nls::NewtonSolver,
             std::shared_ptr<dolfin::nls::NewtonSolver>, PyNewtonSolver>(
      m, "NewtonSolver")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<PyNewtonSolver>(comm.get());
      }))
      .def("solve", &dolfin::nls::NewtonSolver::solve)
      .def("converged", &PyPublicNewtonSolver::converged)
      .def("update_solution", &PyPublicNewtonSolver::update_solution)
      .def_readwrite("atol", &dolfin::nls::NewtonSolver::atol)
      .def_readwrite("rtol", &dolfin::nls::NewtonSolver::rtol)
      .def_readwrite("max_it", &dolfin::nls::NewtonSolver::max_it)
      .def_readwrite("convergence_criterion",
                     &dolfin::nls::NewtonSolver::convergence_criterion);

  // dolfin::NonlinearProblem 'trampoline' for overloading from
  // Python
  class PyNonlinearProblem : public dolfin::nls::NonlinearProblem
  {
    using dolfin::nls::NonlinearProblem::NonlinearProblem;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    Mat J(const Vec x) override
    {
      PYBIND11_OVERLOAD_INT(Mat, dolfin::nls::NonlinearProblem, "J", x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::NonlinerProblem::J");
    }

    Vec F(const Vec x) override
    {
      PYBIND11_OVERLOAD_INT(Vec, dolfin::nls::NonlinearProblem, "F", x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfin::NonlinearProblem::F");
    }

    void form(Vec x) override
    {
      PYBIND11_OVERLOAD_INT(void, dolfin::nls::NonlinearProblem, "form", x);
      return dolfin::nls::NonlinearProblem::form(x);
    }
  };

  // dolfin::NonlinearProblem
  py::class_<dolfin::nls::NonlinearProblem,
             std::shared_ptr<dolfin::nls::NonlinearProblem>,
             PyNonlinearProblem>(m, "NonlinearProblem")
      .def(py::init<>())
      .def("F", &dolfin::nls::NonlinearProblem::F)
      .def("J", &dolfin::nls::NonlinearProblem::J)
      .def("P", &dolfin::nls::NonlinearProblem::P)
      .def("form", &dolfin::nls::NonlinearProblem::form);
}
} // namespace dolfin_wrappers
