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
// #include <nanobind/functional.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void nls(nb::module_& m)
{
  nb::module_ m_petsc
      = m.def_submodule("petsc", "PETSc-specific nonlinear solvers");

  // dolfinx::NewtonSolver
  nb::class_<dolfinx::nls::petsc::NewtonSolver>(m_petsc, "NewtonSolver")
      .def(
          "__init__",
          [](dolfinx::nls::petsc::NewtonSolver* ns, const MPICommWrapper comm)
          { new (ns) dolfinx::nls::petsc::NewtonSolver(comm.get()); },
          nb::arg("comm"))
      .def_prop_ro("krylov_solver",
                             [](const dolfinx::nls::petsc::NewtonSolver& self)
                             {
                               const dolfinx::la::petsc::KrylovSolver& solver
                                   = self.get_krylov_solver();
                               return solver.ksp();
                             })
      .def("setF", &dolfinx::nls::petsc::NewtonSolver::setF, nb::arg("F"),
           nb::arg("b"))
      .def("setJ", &dolfinx::nls::petsc::NewtonSolver::setJ, nb::arg("J"),
           nb::arg("Jmat"))
      .def("setP", &dolfinx::nls::petsc::NewtonSolver::setP, nb::arg("P"),
           nb::arg("Pmat"))
      .def("set_update", &dolfinx::nls::petsc::NewtonSolver::set_update,
           nb::arg("update"))
      .def("set_form", &dolfinx::nls::petsc::NewtonSolver::set_form,
           nb::arg("form"))
      .def("solve", &dolfinx::nls::petsc::NewtonSolver::solve, nb::arg("x"))
      .def_rw("atol", &dolfinx::nls::petsc::NewtonSolver::atol,
                     "Absolute tolerance")
      .def_rw("rtol", &dolfinx::nls::petsc::NewtonSolver::rtol,
                     "Relative tolerance")
      .def_rw(
          "error_on_nonconvergence",
          &dolfinx::nls::petsc::NewtonSolver::error_on_nonconvergence)
      .def_rw("report", &dolfinx::nls::petsc::NewtonSolver::report)
      .def_rw("relaxation_parameter",
                     &dolfinx::nls::petsc::NewtonSolver::relaxation_parameter,
                     "Relaxation parameter")
      .def_rw("max_it", &dolfinx::nls::petsc::NewtonSolver::max_it,
                     "Maximum number of iterations")
      .def_rw("convergence_criterion",
                     &dolfinx::nls::petsc::NewtonSolver::convergence_criterion,
                     "Convergence criterion, either 'residual' (default) or "
                     "'incremental'");
}
} // namespace dolfinx_wrappers
