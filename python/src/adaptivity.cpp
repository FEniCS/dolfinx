// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
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
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dolfin/adaptivity/AdaptiveLinearVariationalSolver.h>
#include <dolfin/adaptivity/AdaptiveNonlinearVariationalSolver.h>
#include <dolfin/adaptivity/ErrorControl.h>
#include <dolfin/adaptivity/GenericAdaptiveVariationalSolver.h>
#include <dolfin/adaptivity/GoalFunctional.h>
#include <dolfin/adaptivity/TimeSeries.h>
#include <dolfin/common/Variable.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{
  // Interface for dolfin/adaptivity
  void adaptivity(py::module& m)
  {
#ifdef HAS_HDF5
    // dolfin::TimesSeries
    py::class_<dolfin::TimeSeries, std::shared_ptr<dolfin::TimeSeries>, dolfin::Variable>(m, "TimeSeries")
      .def(py::init<std::string>())
      .def(py::init([](const MPICommWrapper comm, const std::string &arg)
                    { return std::unique_ptr<dolfin::TimeSeries>(new dolfin::TimeSeries(comm.get(), arg)); }))
      .def("store", (void (dolfin::TimeSeries::*)(const dolfin::GenericVector&, double)) &dolfin::TimeSeries::store)
      .def("store", (void (dolfin::TimeSeries::*)(const dolfin::Mesh&, double)) &dolfin::TimeSeries::store)
      .def("retrieve", (void (dolfin::TimeSeries::*)(dolfin::GenericVector&, double, bool) const) &dolfin::TimeSeries::retrieve,
           py::arg("vector"), py::arg("t"), py::arg("interpolate")=true)
      .def("retrieve", (void (dolfin::TimeSeries::*)(dolfin::Mesh&, double) const) &dolfin::TimeSeries::retrieve)
      .def("vector_times", &dolfin::TimeSeries::vector_times)
      .def("mesh_times", &dolfin::TimeSeries::mesh_times);
#endif

    // dolfin::ErrorControl
    py::class_<dolfin::ErrorControl, std::shared_ptr<dolfin::ErrorControl>,
               dolfin::Variable>
      (m, "ErrorControl", "Error control")
      .def(py::init<std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::Form>,
           bool>())
      .def("estimate_error", &dolfin::ErrorControl::estimate_error)
      .def("estimate_error", [](dolfin::ErrorControl& self, py::object u,
                                const std::vector<std::shared_ptr<const dolfin::DirichletBC>> bcs)
           {
             auto _u = u.attr("_cpp_object").cast<dolfin::Function*>();
             return self.estimate_error(*_u, bcs);
           });

    // dolfin::GoalFunctional
    py::class_<dolfin::GoalFunctional, std::shared_ptr<dolfin::GoalFunctional>,
               dolfin::Form>
      (m, "GoalFunctional", "Goal functional", py::multiple_inheritance());

    py::class_<dolfin::GenericAdaptiveVariationalSolver,
               std::shared_ptr<dolfin::GenericAdaptiveVariationalSolver>,
               dolfin::Variable>
      (m, "GenericAdaptiveVariationalSolver", "Generic adaptive variational solver")
      .def("solve", &dolfin::GenericAdaptiveVariationalSolver::solve)
      .def("summary", &dolfin::GenericAdaptiveVariationalSolver::summary);

    // dolfin::AdaptiveLinearVariationalSolver
    py::class_<dolfin::AdaptiveLinearVariationalSolver,
               std::shared_ptr<dolfin::AdaptiveLinearVariationalSolver>,
               dolfin::GenericAdaptiveVariationalSolver>
      (m, "AdaptiveLinearVariationalSolver", "Adaptive linear variational solver")
      .def(py::init<std::shared_ptr<dolfin::LinearVariationalProblem>,
           std::shared_ptr<dolfin::GoalFunctional>>())
      .def(py::init<std::shared_ptr<dolfin::LinearVariationalProblem>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::ErrorControl>>());

    // dolfin::AdaptiveNonlinearVariationalSolver
    py::class_<dolfin::AdaptiveNonlinearVariationalSolver,
               std::shared_ptr<dolfin::AdaptiveNonlinearVariationalSolver>,
               dolfin::GenericAdaptiveVariationalSolver>
      (m, "AdaptiveNonlinearVariationalSolver", "Adaptive nonlinear variational solver")
      .def(py::init<std::shared_ptr<dolfin::NonlinearVariationalProblem>,
           std::shared_ptr<dolfin::GoalFunctional>>())
      .def(py::init<std::shared_ptr<dolfin::NonlinearVariationalProblem>,
           std::shared_ptr<dolfin::Form>,
           std::shared_ptr<dolfin::ErrorControl>>());
  }
}
