// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/nls/NewtonSolver.h>
#include <dolfinx/nls/NonlinearProblem.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef DEBUG
// Needed for typeid(_p_Mat) in debug mode
#include <petsc/private/matimpl.h>
// Needed for typeid(_p_Vec) in debug mode
#include <petsc/private/vecimpl.h>
#endif

namespace py = pybind11;

namespace dolfinx_wrappers
{
void nls(py::module& m)
{
  // dolfinx::NewtonSolver 'trampoline' for overloading virtual
  // functions from Python
  class PyNewtonSolver : public dolfinx::nls::NewtonSolver
  {
    using dolfinx::nls::NewtonSolver::NewtonSolver;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    bool converged(const Vec r,
                   const dolfinx::nls::NonlinearProblem& nonlinear_problem,
                   std::size_t iteration)
    {
      PYBIND11_OVERLOAD_INT(bool, dolfinx::nls::NewtonSolver, "converged", &r,
                            &nonlinear_problem, iteration);
      return dolfinx::nls::NewtonSolver::converged(r, nonlinear_problem,
                                                  iteration);
    }

    void update_solution(Vec x, const Vec dx, double relaxation,
                         const dolfinx::nls::NonlinearProblem& nonlinear_problem,
                         std::size_t iteration)
    {
      PYBIND11_OVERLOAD_INT(void, dolfinx::nls::NewtonSolver, "update_solution",
                            x, &dx, relaxation, &nonlinear_problem, iteration);
      return dolfinx::nls::NewtonSolver::update_solution(
          x, dx, relaxation, nonlinear_problem, iteration);
    }
  };

  // Class used to expose protected dolfinx::NewtonSolver members
  // (see https://github.com/pybind/pybind11/issues/991)
  class PyPublicNewtonSolver : public dolfinx::nls::NewtonSolver
  {
  public:
    using dolfinx::nls::NewtonSolver::converged;
    // using dolfinx::nls::NewtonSolver::solver_setup;
    using dolfinx::nls::NewtonSolver::update_solution;
  };

  // dolfinx::NewtonSolver
  py::class_<dolfinx::nls::NewtonSolver,
             std::shared_ptr<dolfinx::nls::NewtonSolver>, PyNewtonSolver>(
      m, "NewtonSolver")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<PyNewtonSolver>(comm.get());
      }))
      .def("solve", &dolfinx::nls::NewtonSolver::solve)
      .def("converged", &PyPublicNewtonSolver::converged)
      .def("update_solution", &PyPublicNewtonSolver::update_solution)
      .def_readwrite("atol", &dolfinx::nls::NewtonSolver::atol)
      .def_readwrite("rtol", &dolfinx::nls::NewtonSolver::rtol)
      .def_readwrite("max_it", &dolfinx::nls::NewtonSolver::max_it)
      .def_readwrite("convergence_criterion",
                     &dolfinx::nls::NewtonSolver::convergence_criterion);

  // dolfinx::NonlinearProblem 'trampoline' for overloading from
  // Python
  class PyNonlinearProblem : public dolfinx::nls::NonlinearProblem
  {
    using dolfinx::nls::NonlinearProblem::NonlinearProblem;

    // pybdind11 has some issues when passing by reference (due to
    // the return value policy), so the below is non-standard.  See
    // https://github.com/pybind/pybind11/issues/250.

    Mat J(const Vec x) override
    {
      PYBIND11_OVERLOAD_INT(Mat, dolfinx::nls::NonlinearProblem, "J", x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfinx::NonlinerProblem::J");
    }

    Vec F(const Vec x) override
    {
      PYBIND11_OVERLOAD_INT(Vec, dolfinx::nls::NonlinearProblem, "F", x);
      py::pybind11_fail(
          "Tried to call pure virtual function dolfinx::NonlinearProblem::F");
    }

    void form(Vec x) override
    {
      PYBIND11_OVERLOAD_INT(void, dolfinx::nls::NonlinearProblem, "form", x);
      return dolfinx::nls::NonlinearProblem::form(x);
    }
  };

  // dolfinx::NonlinearProblem
  py::class_<dolfinx::nls::NonlinearProblem,
             std::shared_ptr<dolfinx::nls::NonlinearProblem>,
             PyNonlinearProblem>(m, "NonlinearProblem")
      .def(py::init<>())
      .def("F", &dolfinx::nls::NonlinearProblem::F)
      .def("J", &dolfinx::nls::NonlinearProblem::J)
      .def("P", &dolfinx::nls::NonlinearProblem::P)
      .def("form", &dolfinx::nls::NonlinearProblem::form);
}
} // namespace dolfinx_wrappers
