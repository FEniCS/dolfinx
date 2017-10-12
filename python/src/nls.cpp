// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/PETScSNESSolver.h>
#include <dolfin/nls/PETScTAOSolver.h>
#include <dolfin/nls/TAOLinearBoundSolver.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/OptimisationProblem.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{
  void nls(py::module& m)
  {
    // dolfin::NewtonSolver 'trampoline' for overloading virtual
    // functions from Python
    class PyNewtonSolver : public dolfin::NewtonSolver
    {
      using dolfin::NewtonSolver::NewtonSolver;

      // pybdind11 has some issues when passing by reference (due to
      // the return value policy), so the below is non-standard.  See
      // https://github.com/pybind/pybind11/issues/250.

      bool converged(const dolfin::GenericVector& r,
                     const dolfin::NonlinearProblem& nonlinear_problem,
                     std::size_t iteration)
      {
        PYBIND11_OVERLOAD_INT(bool, dolfin::NewtonSolver, "converged", &r, &nonlinear_problem, iteration);
        return dolfin::NewtonSolver::converged(r, nonlinear_problem, iteration);
      }

      void solver_setup(std::shared_ptr<const dolfin::GenericMatrix> A,
                        std::shared_ptr<const dolfin::GenericMatrix> P,
                        const dolfin::NonlinearProblem& nonlinear_problem,
                        std::size_t iteration)
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::NewtonSolver, "solver_setup", A, P,
                          &nonlinear_problem, iteration);
        return dolfin::NewtonSolver::solver_setup(A, P, nonlinear_problem, iteration);
      }

      void update_solution(dolfin::GenericVector& x, const dolfin::GenericVector& dx,
                           double relaxation_parameter,
                           const dolfin::NonlinearProblem& nonlinear_problem,
                           std::size_t iteration)
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::NewtonSolver, "update_solution",
                              &x, &dx, relaxation_parameter, nonlinear_problem,
                              iteration);
        return dolfin::NewtonSolver::update_solution(x, dx, relaxation_parameter,
                                                     nonlinear_problem, iteration);
      }
    };

    // Class used to expose protected dolfin::NewtonSolver members
    // (see https://github.com/pybind/pybind11/issues/991)
    class PyPublicNewtonSolver : public dolfin::NewtonSolver
    {
    public:
      using NewtonSolver::converged;
      using NewtonSolver::solver_setup;
      using NewtonSolver::update_solution;
    };

    // dolfin::NewtonSolver
    py::class_<dolfin::NewtonSolver, std::shared_ptr<dolfin::NewtonSolver>, PyNewtonSolver,
               dolfin::Variable>(m, "NewtonSolver")
      .def(py::init<>())
      .def(py::init([](const MPICommWrapper comm)
          { return std::unique_ptr<dolfin::NewtonSolver>(new dolfin::NewtonSolver(comm.get())); }))
      .def("solve", &dolfin::NewtonSolver::solve)
      .def("converged", &PyPublicNewtonSolver::converged)
      .def("solver_setup", &PyPublicNewtonSolver::solver_setup)
      .def("update_solution", &PyPublicNewtonSolver::update_solution);

#ifdef HAS_PETSC
    // dolfin::PETScSNESSolver
    py::class_<dolfin::PETScSNESSolver, std::shared_ptr<dolfin::PETScSNESSolver>,
               dolfin::PETScObject>(m, "PETScSNESSolver")
      .def(py::init<std::string>(), py::arg("nls_type")="default")
      .def(py::init([](const MPICommWrapper comm, std::string nls_type="default")
          { return std::unique_ptr<dolfin::PETScSNESSolver>(
              new dolfin::PETScSNESSolver(comm.get(), nls_type)); }),
          py::arg("comm"), py::arg("nls_type") = "default")
      .def_readwrite("parameters", &dolfin::PETScSNESSolver::parameters)
      .def("solve", (std::pair<std::size_t, bool> (dolfin::PETScSNESSolver::*)(dolfin::NonlinearProblem&,
                                                                               dolfin::GenericVector&))
           &dolfin::PETScSNESSolver::solve);

    // dolfin::TAOLinearBoundSolver
    py::class_<dolfin::TAOLinearBoundSolver>(m, "TAOLinearBoundSolver")
      .def(py::init([](const MPICommWrapper comm)
          { return std::unique_ptr<dolfin::TAOLinearBoundSolver>(new dolfin::TAOLinearBoundSolver(comm.get())); }))
      .def(py::init<std::string, std::string, std::string>(), py::arg("method")="default",
           py::arg("ksp_type")="default", py::arg("pc_type")="default")
      .def("solve", (std::size_t (dolfin::TAOLinearBoundSolver::*)
                     (const dolfin::GenericMatrix&, dolfin::GenericVector&,
                      const dolfin::GenericVector&, const dolfin::GenericVector&,
                      const dolfin::GenericVector&))
           &dolfin::TAOLinearBoundSolver::solve);

    // dolfin::PETScTAOSolver
    py::class_<dolfin::PETScTAOSolver, std::shared_ptr<dolfin::PETScTAOSolver>, dolfin::PETScObject>(m, "PETScTAOSolver")
      .def(py::init<>())

      .def(py::init([](const MPICommWrapper comm, std::string tao_type="default",
                       std::string ksp_type="default", std::string pc_type="default")
          { return std::unique_ptr<dolfin::PETScTAOSolver>(
              new dolfin::PETScTAOSolver(comm.get(), tao_type, ksp_type, pc_type)); }),
           py::arg("comm"), py::arg("tao_type")="default",
           py::arg("ksp_type")="default", py::arg("pc_type")="default")
      .def_readwrite("parameters", &dolfin::PETScTAOSolver::parameters)
      .def("solve", (std::pair<std::size_t, bool> (dolfin::PETScTAOSolver::*)(dolfin::OptimisationProblem&, dolfin::GenericVector&))
           &dolfin::PETScTAOSolver::solve)
      .def("solve", (std::pair<std::size_t, bool> (dolfin::PETScTAOSolver::*)(dolfin::OptimisationProblem&, dolfin::GenericVector&,
                                                                              const dolfin::GenericVector&, const dolfin::GenericVector&))
           &dolfin::PETScTAOSolver::solve);
#endif

    // dolfin::NonlinearProblem 'trampoline' for overloading from
    // Python
    class PyNonlinearProblem : public dolfin::NonlinearProblem
    {
      using dolfin::NonlinearProblem::NonlinearProblem;

      // pybdind11 has some issues when passing by reference (due to
      // the return value policy), so the below is non-standard.  See
      // https://github.com/pybind/pybind11/issues/250.

      void J(dolfin::GenericMatrix& A, const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::NonlinearProblem, "J", &A, &x);
        py::pybind11_fail("Tried to call pure virtual function dolfin::OptimisationProblem::J");
      }

      void F(dolfin::GenericVector& b, const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::NonlinearProblem, "F", &b, &x);
        py::pybind11_fail("Tried to call pure virtual function dolfin::OptimisationProblem::F");
      }

      void form(dolfin::GenericMatrix& A, dolfin::GenericMatrix& P,
                dolfin::GenericVector& b, const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::NonlinearProblem, "form", &A, &P, &b, &x);
        return dolfin::NonlinearProblem::form(A, P, b, x);
      }

    };

    // dolfin::NonlinearProblem
    py::class_<dolfin::NonlinearProblem, std::shared_ptr<dolfin::NonlinearProblem>, PyNonlinearProblem>(m, "NonlinearProblem")
      .def(py::init<>())
      .def("F", &dolfin::NonlinearProblem::F)
      .def("J", &dolfin::NonlinearProblem::J)
      .def("form", (void (dolfin::NonlinearProblem::*)(dolfin::GenericMatrix&, dolfin::GenericMatrix&,
                                                       dolfin::GenericVector&, const dolfin::GenericVector&))
                    &dolfin::NonlinearProblem::form);

    // dolfin::OptimizationProblem 'trampoline' for overloading from
    // Python
    class PyOptimisationProblem : public dolfin::OptimisationProblem
    {
      using dolfin::OptimisationProblem::OptimisationProblem;

      // pybdind11 has some issues when passing by reference (due to
      // the return value policy), so the below is non-standard.  See
      // https://github.com/pybind/pybind11/issues/250.

      double f(const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(double, dolfin::OptimisationProblem, "f", &x);
        py::pybind11_fail("Tried to call pure virtual function dolfin::OptimisationProblem::f");
      }

      void F(dolfin::GenericVector& b, const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::OptimisationProblem, "F", &b, &x);
        py::pybind11_fail("Tried to call pure virtual function dolfin::OptimisationProblem::F");
      }

      void J(dolfin::GenericMatrix& A, const dolfin::GenericVector& x) override
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::OptimisationProblem, "J", &A, &x);
        py::pybind11_fail("Tried to call pure virtual function dolfin::OptimisationProblem::J");
      }
    };

    // dolfin::OptimizationProblem
    py::class_<dolfin::OptimisationProblem, std::shared_ptr<dolfin::OptimisationProblem>,
               PyOptimisationProblem>(m, "OptimisationProblem")
      .def(py::init<>())
      .def("f", &dolfin::OptimisationProblem::f)
      .def("F", &dolfin::OptimisationProblem::F)
      .def("J", &dolfin::OptimisationProblem::J);
  }
}
