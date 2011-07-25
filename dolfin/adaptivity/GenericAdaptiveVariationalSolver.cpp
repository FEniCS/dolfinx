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
// Modified by Anders Logg 2010-2011
//
// First added:  2010-08-19
// Last changed: 2011-06-22

#include <dolfin/common/utils.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/plot/plot.h>

#include "GenericAdaptiveVariationalSolver.h"
#include "GoalFunctional.h"
#include "ErrorControl.h"
#include "TimeSeries.h"
#include "adapt.h"
#include "marking.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GenericAdaptiveVariationalSolver::solve(const double tol,
                                             Form& goal,
                                             ErrorControl& control)
{
  // Clear adaptive data
  _adaptive_data.clear();
  std::string label = parameters["data_label"];
  TimeSeries series(label);

  // Start adaptive loop
  const uint max_iterations = parameters["max_iterations"];

  // Iterate over a series of meshes
  for (uint i = 0; i < max_iterations; i++)
  {
    // Deal with goal and error control on current mesh
    Form& M = goal.fine();
    ErrorControl& ec = control.fine();

    //--- Stage 0: Solve primal problem
    begin("Stage %d.0: Solving primal problem...", i);
    boost::shared_ptr<const Function> u = solve_primal();
    assert(u);

    // Evaluate goal functional
    info("Evaluating goal functional.");
    const double functional_value = evaluate_goal(M, u);
    info("Value of goal functional is %g.", functional_value);
    end();

    //--- Stage 1: Estimate error
    begin("Stage %d.1: Computing error estimate...", i);
    const double error_estimate = ec.estimate_error(*u, extract_bcs());
    info("Error estimate is %g (tol = %g).", error_estimate, tol);
    end();

    // Initialize adaptive data
    assert(u);
    const FunctionSpace& V = u->function_space();
    const Mesh& mesh = V.mesh();
    boost::shared_ptr<AdaptiveDatum> datum(new AdaptiveDatum(i, V.dim(),
                                                             mesh.num_cells(),
                                                             error_estimate,
                                                             tol,
                                                             functional_value));
    if (parameters["reference"].change_count() > 0)
      datum->set_reference_value(parameters["reference"]);
    _adaptive_data.push_back(datum);
    summary(*datum);

    // Check stopping criterion
    if (stop(V, error_estimate, tol))
    {
      end();
      summary();
      return;
    }

    //--- Stage 2: Compute error indicators
    begin("Stage %d.2: Computing error indicators...", i);
    Vector indicators(mesh.num_cells());
    assert(u);
    ec.compute_indicators(indicators, *u);
    if (parameters["save_data"])
    {
      series.store(indicators, i);
      series.store(mesh, i);
    }
    end();

    //--- Stage 3: Mark mesh for refinement ---
    begin("Stage %d.3: Marking mesh for refinement...", i);
    MeshFunction<bool> markers(mesh, mesh.topology().dim());
    const std::string strategy = parameters["marking_strategy"];
    const double fraction = parameters["marking_fraction"];
    mark(markers, indicators, strategy, fraction);
    end();

    //--- Stage 4: Refine mesh ---
    begin("Stage %d.4: Refining mesh...", i);
    adapt(mesh, markers);
    if (parameters["plot_mesh"])
      plot(mesh.child(), "Refined mesh");
    end();

    //--- Stage 5: Update forms ---
    begin("Stage %d.5: Updating forms...", i);
    adapt_problem(mesh.fine_shared_ptr());
    adapt(M, mesh.fine_shared_ptr());
    adapt(ec, mesh.fine_shared_ptr());
    end();
  }

  summary();
  warning("Maximal number of iterations (%d) exceeded! Returning anyhow.",
          max_iterations);
}
//-----------------------------------------------------------------------------
std::vector<boost::shared_ptr<AdaptiveDatum> >
GenericAdaptiveVariationalSolver::adaptive_data() const
{
  return _adaptive_data;
}
//-----------------------------------------------------------------------------
bool GenericAdaptiveVariationalSolver::stop(const FunctionSpace& V,
                                            const double error_estimate,
                                            const double tolerance)
{
  // Done if error is less than tolerance
  if (std::abs(error_estimate) < tolerance)
    return true;

  // Or done if dimension is larger than max dimension (and that
  // parameter is set).
  const uint max_dimension = parameters["max_dimension"];
  if (parameters["max_dimension"].change_count() > 0
      && V.dim() > max_dimension)
    return true;

  return false;
}
//-----------------------------------------------------------------------------
void GenericAdaptiveVariationalSolver::summary()
{
  // Show parameters used
  info("");
  info("Parameters used for adaptive solve:");
  info("");
  info(parameters, true);

  // Show summary for all iterations
  info("");
  info("Summary of adaptive solve:");
  info("");
  Table table("Level");
  for (uint i = 0; i < _adaptive_data.size(); i++)
    _adaptive_data[i]->store(table);
  info(indent(table.str(true)));
  info("");
}
//-----------------------------------------------------------------------------
void GenericAdaptiveVariationalSolver::summary(const AdaptiveDatum& datum)
{
  // Show summary for given adaptive datum
  info("");
  info("Current adaptive data");
  info("");
  Table table("Level");
  datum.store(table);
  info(indent(table.str(true)));
  info("");
}
//-----------------------------------------------------------------------------
