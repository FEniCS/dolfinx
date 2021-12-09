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

  // dolfinx::NewtonSolver
  py::class_<dolfinx::nls::NewtonSolver,
             std::shared_ptr<dolfinx::nls::NewtonSolver>>(m, "NewtonSolver")
      .def(py::init(
          [](const MPICommWrapper comm)
          { return std::make_unique<dolfinx::nls::NewtonSolver>(comm.get()); }))
      .def_property_readonly(
          "krylov_solver",
          [](const dolfinx::nls::NewtonSolver& self)
          {
            const dolfinx::la::PETScKrylovSolver& krylov_solver
                = self.get_krylov_solver();
            KSP ksp = krylov_solver.ksp();
            return ksp;
          })
      .def("setF", &dolfinx::nls::NewtonSolver::setF)
      .def("setJ", &dolfinx::nls::NewtonSolver::setJ)
      .def("setP", &dolfinx::nls::NewtonSolver::setP)
      .def("set_update", &dolfinx::nls::NewtonSolver::set_update)
      .def("set_form", &dolfinx::nls::NewtonSolver::set_form)
      .def("solve", &dolfinx::nls::NewtonSolver::solve)
      .def_readwrite("atol", &dolfinx::nls::NewtonSolver::atol,
                     "Absolute tolerance")
      .def_readwrite("rtol", &dolfinx::nls::NewtonSolver::rtol,
                     "Relative tolerance")
      .def_readwrite("error_on_nonconvergence",
                     &dolfinx::nls::NewtonSolver::error_on_nonconvergence)
      .def_readwrite("report", &dolfinx::nls::NewtonSolver::report)
      .def_readwrite("relaxation_parameter",
                     &dolfinx::nls::NewtonSolver::relaxation_parameter,
                     "Relaxation parameter")
      .def_readwrite("max_it", &dolfinx::nls::NewtonSolver::max_it,
                     "Maximum number of iterations")
      .def_readwrite("convergence_criterion",
                     &dolfinx::nls::NewtonSolver::convergence_criterion,
                     "Convergence criterion, either 'residual' (default) or "
                     "'incremental'");
}
} // namespace dolfinx_wrappers
