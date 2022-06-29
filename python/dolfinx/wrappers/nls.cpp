// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/nls/NewtonSolver.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void nls(py::module& m)
{
  py::module m_petsc
      = m.def_submodule("petsc", "PETSc-specific nonlinear solvers");

  // dolfinx::NewtonSolver
  py::class_<dolfinx::nls::petsc::NewtonSolver,
             std::shared_ptr<dolfinx::nls::petsc::NewtonSolver>>(m_petsc,
                                                                 "NewtonSolver")
      .def(py::init(
               [](const MPICommWrapper comm) {
                 return std::make_unique<dolfinx::nls::petsc::NewtonSolver>(
                     comm.get());
               }),
           py::arg("comm"))
      .def_property_readonly("krylov_solver",
                             [](const dolfinx::nls::petsc::NewtonSolver& self)
                             {
                               const dolfinx::la::petsc::KrylovSolver& solver
                                   = self.get_krylov_solver();
                               return solver.ksp();
                             })
      .def("setF", &dolfinx::nls::petsc::NewtonSolver::setF, py::arg("F"),
           py::arg("b"))
      .def("setJ", &dolfinx::nls::petsc::NewtonSolver::setJ, py::arg("J"),
           py::arg("Jmat"))
      .def("setP", &dolfinx::nls::petsc::NewtonSolver::setP, py::arg("P"),
           py::arg("Pmat"))
      .def("set_update", &dolfinx::nls::petsc::NewtonSolver::set_update,
           py::arg("update"))
      .def("set_form", &dolfinx::nls::petsc::NewtonSolver::set_form,
           py::arg("form"))
      .def("solve", &dolfinx::nls::petsc::NewtonSolver::solve, py::arg("x"))
      .def_readwrite("atol", &dolfinx::nls::petsc::NewtonSolver::atol,
                     "Absolute tolerance")
      .def_readwrite("rtol", &dolfinx::nls::petsc::NewtonSolver::rtol,
                     "Relative tolerance")
      .def_readwrite(
          "error_on_nonconvergence",
          &dolfinx::nls::petsc::NewtonSolver::error_on_nonconvergence)
      .def_readwrite("report", &dolfinx::nls::petsc::NewtonSolver::report)
      .def_readwrite("relaxation_parameter",
                     &dolfinx::nls::petsc::NewtonSolver::relaxation_parameter,
                     "Relaxation parameter")
      .def_readwrite("max_it", &dolfinx::nls::petsc::NewtonSolver::max_it,
                     "Maximum number of iterations")
      .def_readwrite("convergence_criterion",
                     &dolfinx::nls::petsc::NewtonSolver::convergence_criterion,
                     "Convergence criterion, either 'residual' (default) or "
                     "'incremental'");
}
} // namespace dolfinx_wrappers
