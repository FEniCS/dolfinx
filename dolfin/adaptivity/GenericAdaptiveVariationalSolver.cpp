// Copyright (C) 2010 Marie E. Rognes
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
//
// Modified by Anders Logg 2010-2013
//
// First added:  2010-08-19
// Last changed: 2015-06-16

#include <sstream>
#include <stdio.h>

#include <dolfin/common/utils.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/Table.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/plot/plot.h>

#include "GenericAdaptiveVariationalSolver.h"
#include "GoalFunctional.h"
#include "ErrorControl.h"
#include "adapt.h"
#include "ErrorControl.h"
#include "marking.h"
#include "TimeSeries.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericAdaptiveVariationalSolver::~GenericAdaptiveVariationalSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericAdaptiveVariationalSolver::solve(const double tol)
{
  log(INFO, "Solving variational problem adaptively");

  // Check that goal functional has at least one coefficient (last one
  // is the solution variable u)
  if (goal->num_coefficients() == 0)
  {
    dolfin_error("AdaptiveLinearVariationalSolver.cpp",
                 "adaptively solve variational problem",
                 "The goal functional must have at least one coefficient. "
                 "Check that it has been defined in terms of a Function, not a TrialFunction");
  }

  // Initialize storage of meshes and indicators
  std::string label = parameters["data_label"];
#ifdef HAS_HDF5
  TimeSeries series(goal->mesh()->mpi_comm(), label);
#endif

  // Iterate over a series of meshes
  Timer timer("Adaptive solve");
  const std::size_t max_iterations = parameters["max_iterations"];
  for (std::size_t i = 0; i < max_iterations; i++)
  {
    log(INFO, "Adaptive iteration %d", i );

    // Check that num_dofs is not greater than than max dimension (and
    // that that parameter is modified)
    const std::size_t max_dimension = parameters["max_dimension"];
    if (parameters["max_dimension"].change_count() > 0
        && num_dofs_primal() > max_dimension)
    {
      info("Maximal number of dofs reached, finishing");
      return;
    }

    // Initialize adaptive data
    std::shared_ptr<Parameters> datum(new Parameters("adaptive_data"));
    _adaptive_data.push_back(datum);
    const int refinement_level = i;
    datum->add("refinement_level", refinement_level);
    datum->add("tolerance", tol);
    if (parameters["reference"].change_count() > 0)
    {
      const double reference = parameters["reference"];
      datum->add("reference", reference);
    }

    // Deal with goal and error control on current mesh
    Form& M = goal->leaf_node();
    ErrorControl& ec = control->leaf_node();
    ec.parameters.update(parameters("error_control"));

    //--- Stage 0: Solve primal problem
    begin(PROGRESS, "Stage %d.0: Solving primal problem...", i);
    timer.start();
    std::shared_ptr<const Function> u = solve_primal();
    datum->add("time_solve_primal", timer.stop());

    // Extract views to primal trial space and mesh
    dolfin_assert(u->function_space());
    const FunctionSpace& V = *u->function_space();
    dolfin_assert(V.mesh());
    std::shared_ptr<const Mesh> mesh = V.mesh();
    dolfin_assert(mesh);

    // Evaluate goal functional
    log(PROGRESS, "Evaluating goal functional.");
    const double functional_value = evaluate_goal(M, u);
    info("Value of goal functional is %g.", functional_value);
    end();

    //--- Stage 1: Estimate error
    begin(PROGRESS, "Stage %d.1: Computing error estimate...", i);
    timer.start();
    const double error_estimate = ec.estimate_error(*u, extract_bcs());
    datum->add("time_estimate_error", timer.stop());
    log(PROGRESS, "Error estimate is %g (tol = %g).", error_estimate, tol);
    end();

    const int num_cells = mesh->num_cells();
    const int num_dofs = V.dim();
    datum->add("num_cells", num_cells);
    datum->add("num_dofs", num_dofs);
    datum->add("error_estimate", error_estimate);
    datum->add("functional_value", functional_value);

    // Stop if error estimate is less than tolerance
    if (std::abs(error_estimate) < tol)
    {
      info("Error estimate (%g) is less than tolerance (%g), returning.",
           error_estimate, tol);
      return;
    }

    //--- Stage 2: Compute error indicators
    begin(PROGRESS, "Stage %d.2: Computing error indicators...", i);
    timer.start();
    MeshFunction<double> indicators(mesh, mesh->topology().dim());
    dolfin_assert(u);
    ec.compute_indicators(indicators, *u);
    datum->add("time_compute_indicators", timer.stop());
    if (parameters["save_data"])
    {
#ifdef HAS_HDF5
      //series.store(indicators, i); // No TimeSeries storage of MeshFunction
      series.store(*mesh, i);
#else
      warning("HDF5 not available, unable to save refinement data as time series.");
#endif
    }
    end();

    //--- Stage 3: Mark mesh for refinement ---
    begin(PROGRESS, "Stage %d.3: Marking mesh for refinement...", i);
    MeshFunction<bool> markers(mesh, mesh->topology().dim());
    const std::string strategy = parameters["marking_strategy"];
    const double fraction = parameters["marking_fraction"];
    timer.start();
    mark(markers, indicators, strategy, fraction);
    datum->add("time_mark_mesh", timer.stop());
    end();

    //--- Stage 4: Refine mesh ---
    begin(PROGRESS, "Stage %d.4: Refining mesh...", i);
    timer.start();
    adapt(*mesh, markers);
    datum->add("time_adapt_mesh", timer.stop());
    if (parameters["plot_mesh"])
      plot(mesh->child(), "Refined mesh");
    end();

    //--- Stage 5: Update forms ---
    begin(PROGRESS, "Stage %d.5: Updating forms...", i);
    timer.start();
    adapt_problem(mesh->leaf_node_shared_ptr());
    adapt(M, mesh->leaf_node_shared_ptr());
    adapt(ec, mesh->leaf_node_shared_ptr(), false);
    datum->add("time_adapt_forms", timer.stop());
    end();
  }

  warning("Maximal number of iterations (%d) exceeded! Returning anyhow.",
          max_iterations);
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<Parameters>>
GenericAdaptiveVariationalSolver::adaptive_data() const
{
  return _adaptive_data;
}
//-----------------------------------------------------------------------------
void GenericAdaptiveVariationalSolver::summary()
{
  // Show parameters used
  info("");
  info("Parameters used for adaptive solve:");
  info("");
  info(parameters, false);

  Table table("Level");
  Table time_table("Level");

  for (std::size_t i = 0; i < _adaptive_data.size(); i++)
  {
    std::stringstream s;
    s << i;
    const Parameters& datum = *_adaptive_data[i];

    if (datum.has_key("reference"))
      table(s.str(), "reference") = datum["reference"].value_str();
    table(s.str(), "functional_value") = datum["functional_value"].value_str();
    table(s.str(), "error_estimate") = datum["error_estimate"].value_str();
    table(s.str(), "tolerance") = datum["tolerance"].value_str();
    table(s.str(), "num_cells") = datum["num_cells"].value_str();
    table(s.str(), "num_dofs") = datum["num_dofs"].value_str();

    time_table(s.str(), "solve_primal")
      = datum["time_solve_primal"].value_str();
    time_table(s.str(), "estimate_error")
      = datum["time_estimate_error"].value_str();
    if (datum.has_key("time_compute_indicators"))
      time_table(s.str(), "compute_indicators")
        = datum["time_compute_indicators"].value_str();
    else
      time_table(s.str(), "compute_indicators") = 0.0;
    if (datum.has_key("time_mark_mesh"))
      time_table(s.str(), "mark_mesh") = datum["time_mark_mesh"].value_str();
    else
      time_table(s.str(), "mark_mesh") = 0.0;
    if (datum.has_key("time_adapt_mesh"))
      time_table(s.str(), "adapt_mesh") = datum["time_adapt_mesh"].value_str();
    else
      time_table(s.str(), "adapt_mesh") = 0.0;
    if (datum.has_key("time_adapt_forms"))
      time_table(s.str(), "update") = datum["time_adapt_forms"].value_str();
    else
      time_table(s.str(), "update") = 0.0;
  }

  // Show summary for all iterations
  info("");
  info("Summary of adaptive data:");
  info("");
  info(indent(table.str(true)));
  info("");
  info("Time spent for adaptive solve (in seconds):");
  info("");
  info(indent(time_table.str(true)));
  info("");
}
//-----------------------------------------------------------------------------
