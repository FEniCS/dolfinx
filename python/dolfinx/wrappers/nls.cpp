// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
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
      .def(py::init([](const MPICommWrapper comm) {
        return std::make_unique<dolfinx::nls::NewtonSolver>(comm.get());
      }))
      .def("setF", &dolfinx::nls::NewtonSolver::setF,
           R"mydelimiter(
        Set the function for computing the residual and the vector to the assemble the residual into
        Parameters
        -----------
        F
          Function to compute the residual vector b (x, b)
        b
          The vector to assemble to residual into)mydelimiter")
      .def("setJ", &dolfinx::nls::NewtonSolver::setJ,
           R"mydelimiter(
        Set the function for computing the Jacobian (dF/dx) and the matrix to assemble the residual into
        Parameters
        -----------
        J
          Function to compute the Jacobian matrix b (x, A)
        Jmat
          The matrix to assemble the Jacobian into)mydelimiter")
      .def("setP", &dolfinx::nls::NewtonSolver::setP,
           R"mydelimiter(
        Set the function for computing the preconditioner matrix (optional)
        Parameters
        -----------
        P
          Function to compute the preconditioner matrix b (x, P)
        Pmat
          The matrix to assemble the preconditioner into)mydelimiter")
      .def("set_form", &dolfinx::nls::NewtonSolver::set_form,
           R"mydelimiter(
        Set the function that is called before the residual or Jacobian are computed. It is commonly used to update ghost values.
        Parameters
        -----------
        form
          The function to call. It takes the latest solution
        vector @p x as an argument.)mydelimiter")
      .def("solve", &dolfinx::nls::NewtonSolver::solve,
           R"mydelimiter(
        Solve the nonlinear problem \f$`F(x) = 0\f$ for given \f$F\f$ and Jacobian \f$\dfrac{\partial F}{\partial x}\f$.
        Returns the number of Newton iterations and whether iteration converged)
        Parameters
        -----------
        x
          The vector)mydelimiter")
      .def_readwrite("atol", &dolfinx::nls::NewtonSolver::atol,
                     "Absolute tolerance")
      .def_readwrite("rtol", &dolfinx::nls::NewtonSolver::rtol,
                     "Relative tolerance")
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
