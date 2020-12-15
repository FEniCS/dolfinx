// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/nls/NewtonSolver.h>
// #include <dolfinx/nls/NonlinearProblem.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// // Needed for typeid(_p_Mat) in debug mode
// #include <petsc/private/matimpl.h>
// // Needed for typeid(_p_Vec) in debug mode
#include <petsc/private/vecimpl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void nls(py::module& m)
{

  // dolfinx::NewtonSolver
  py::class_<dolfinx::nls::NewtonSolver,
             std::shared_ptr<dolfinx::nls::NewtonSolver>>(m, "NewtonSolver")
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfinx::nls::NewtonSolver>(comm.get());
      }))
      .def("setF", &dolfinx::nls::NewtonSolver::setF)
      .def("setJ", &dolfinx::nls::NewtonSolver::setJ)
      .def("setP", &dolfinx::nls::NewtonSolver::setP)
      .def("set_form", &dolfinx::nls::NewtonSolver::set_form)
      .def("solve", &dolfinx::nls::NewtonSolver::solve)
      // .def("converged", &PyPublicNewtonSolver::converged)
      // .def("update_solution", &PyPublicNewtonSolver::update_solution)
      .def_readwrite("atol", &dolfinx::nls::NewtonSolver::atol)
      .def_readwrite("rtol", &dolfinx::nls::NewtonSolver::rtol)
      .def_readwrite("relaxation_parameter",
                     &dolfinx::nls::NewtonSolver::relaxation_parameter)
      .def_readwrite("max_it", &dolfinx::nls::NewtonSolver::max_it)
      .def_readwrite("convergence_criterion",
                     &dolfinx::nls::NewtonSolver::convergence_criterion);
}
} // namespace dolfinx_wrappers
