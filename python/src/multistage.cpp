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
#include <pybind11/stl.h>

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/multistage/MultiStageScheme.h>
#include <dolfin/multistage/PointIntegralSolver.h>
#include <dolfin/multistage/RKSolver.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
  void multistage(py::module& m)
  {
    // dolfin::MultiStageScheme
    py::class_<dolfin::MultiStageScheme, std::shared_ptr<dolfin::MultiStageScheme>>
      (m, "MultiStageScheme")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const dolfin::Form>>>,
           std::shared_ptr<const dolfin::Form>,
           std::vector<std::shared_ptr<dolfin::Function>>,
           std::shared_ptr<dolfin::Function>,
           std::shared_ptr<dolfin::Constant>,
           std::shared_ptr<dolfin::Constant>,
           std::vector<double>,
           std::vector<int>,
           unsigned int,
           const std::string,
           const std::string,
           std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def("order", &dolfin::MultiStageScheme::order);

    // dolfin::RKSolver
    py::class_<dolfin::RKSolver, std::shared_ptr<dolfin::RKSolver>>
      (m, "RKSolver")
      .def(py::init<std::shared_ptr<dolfin::MultiStageScheme>>())
      .def("step_interval", &dolfin::RKSolver::step_interval);

    // dolfin::PointIntegralSolver
    py::class_<dolfin::PointIntegralSolver, std::shared_ptr<dolfin::PointIntegralSolver>>
      (m, "PointIntegralSolver")
      .def(py::init<std::shared_ptr<dolfin::MultiStageScheme>>())
      .def("reset_newton_solver", &dolfin::PointIntegralSolver::reset_newton_solver)
      .def("reset_stage_solutions", &dolfin::PointIntegralSolver::reset_stage_solutions)
      .def("step", &dolfin::PointIntegralSolver::step)
      .def("step_interval", &dolfin::PointIntegralSolver::step_interval);
  }
}
