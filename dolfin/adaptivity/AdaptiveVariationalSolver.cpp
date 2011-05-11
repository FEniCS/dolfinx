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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-08-19
// Last changed: 2011-03-15

#include <dolfin/common/utils.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/Table.h>
#include <dolfin/fem/VariationalProblem.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/plot/plot.h>
#include "adapt.h"
#include "ErrorControl.h"
#include "GoalFunctional.h"
#include "marking.h"
#include "AdaptiveVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AdaptiveVariationalSolver::solve(Function& u,
                                      const VariationalProblem& problem,
                                      const double tol,
                                      GoalFunctional& M,
                                      const Parameters& parameters)
{

  // Extract error control view from goal functional
  boost::shared_ptr<const Form> a = problem.bilinear_form();
  assert(a);
  boost::shared_ptr<const Form> L = problem.linear_form();
  assert(L);

  M.update_ec(*a, *L);
  ErrorControl& ec(*(M._ec));

  AdaptiveVariationalSolver::solve(u, problem, tol, M, ec, parameters);
}
//-----------------------------------------------------------------------------
void AdaptiveVariationalSolver::solve(Function& w,
                                      const VariationalProblem& pde,
                                      const double tol,
                                      Form& goal,
                                      ErrorControl& control,
                                      const Parameters& parameters)
{
  // Set tolerance parameter if not set
  //if (parameters["tolerance"].change_count() == 0)
  //  parameters["tolerance"] = tol;

  // A list of adaptive data
  std::vector<AdaptiveDatum> data;

  // Start adaptive loop
  const uint maxiter = parameters["max_iterations"];

  for (uint i = 0; i < maxiter; i++)
  {
    // Update references to current
    const VariationalProblem& problem = pde.fine();
    Function& u = w.fine();
    Form& M = goal.fine();
    ErrorControl& ec = control.fine();

    //--- Stage 0: Solve primal problem ---
    info("");
    begin("Stage %d.0: Solving primal problem...", i);
    problem.solve(u);

    // Extract function space and mesh
    const FunctionSpace& V(u.function_space());
    const Mesh& mesh(V.mesh());
    end();

    //--- Stage 1: Estimate error ---
    begin("Stage %d.1: Computing error estimate...", i);
    const double error_estimate = ec.estimate_error(u, problem.bcs());

    // Evaluate functional value
    if (!problem.is_nonlinear())
    {
      boost::shared_ptr<const GenericFunction> _u(&u, NoDeleter());
      M.set_coefficient(M.num_coefficients() - 1, _u);
    }
    const double functional_value = assemble(M);

    // Initialize adaptive data
    AdaptiveDatum datum(i, V.dim(), mesh.num_cells(), error_estimate,
                        tol, functional_value);
    if (parameters["reference"].change_count() > 0)
      datum.set_reference_value(parameters["reference"]);
    data.push_back(datum);

    // Check stopping criterion
    if (stop(V, error_estimate, tol, parameters))
    {
      end();
      summary(data, parameters);
      return;
    }
    info("Estimated error (%0.5g) does not satisfy tolerance (%0.5g).",
         error_estimate, tol);
    end();
    summary(datum);


    //--- Stage 2: Compute error indicators ---
    begin("Stage %d.2: Computing error indicators...", i);
    Vector indicators(mesh.num_cells());
    ec.compute_indicators(indicators, u);
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
    adapt(problem, mesh.fine_shared_ptr());
    adapt(u, mesh.fine_shared_ptr());
    adapt(M, mesh.fine_shared_ptr());
    adapt(ec, mesh.fine_shared_ptr());
    end();
  }

  summary(data, parameters);
  warning("Maximal number of iterations (%d) exceeded! Returning anyhow.", maxiter);
}
//-----------------------------------------------------------------------------
bool AdaptiveVariationalSolver::stop(const FunctionSpace& V,
                                     const double error_estimate,
                                     const double tolerance,
                                     const Parameters& parameters)
{
  // Done if error is less than tolerance
  if (std::abs(error_estimate) < tolerance)
    return true;

  // Or done if dimension is larger than max dimension (and that
  // parameter is set).
  const uint max_dimension = parameters["max_dimension"];
  if (parameters["max_dimension"].change_count() > 0
      && V.dim() > max_dimension)
  {
    return true;
  }
  else
    return false;
}
//-----------------------------------------------------------------------------
void AdaptiveVariationalSolver::summary(const std::vector<AdaptiveDatum>& data,
                                        const Parameters& parameters)
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
  for (uint i = 0; i < data.size(); i++)
    data[i].store(table);
  info(indent(table.str(true)));
  info("");
}
//-----------------------------------------------------------------------------
void AdaptiveVariationalSolver::summary(const AdaptiveDatum& datum)
{
  // Show summary for all iterations
  info("");
  info("Current adaptive data");
  info("");
  Table table("Level");
  datum.store(table);
  info(indent(table.str(true)));
  info("");
}
