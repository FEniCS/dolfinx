// Copyright (C) 2010 Marie E. Rognes.
// Licensed under the GNU LGPL Version 3.0 or any later version.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-08-19
// Last changed: 2011-01-14

#include <dolfin/common/utils.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/Table.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/refine.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/plot/plot.h>
#include <dolfin/fem/VariationalProblem.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/assemble.h>
#include "marking.h"
#include "GoalFunctional.h"
#include "ErrorControl.h"
#include "AdaptiveSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AdaptiveSolver::solve(Function& u,
                           VariationalProblem& problem,
                           double tol,
                           GoalFunctional& M,
                           Parameters& parameters)
{

  // Extract error control view from goal functional
  M.update_ec(problem.a, problem.L);
  ErrorControl& ec(*(M._ec));

  AdaptiveSolver::solve(u, problem, tol, M, ec, parameters);
}
//-----------------------------------------------------------------------------
void AdaptiveSolver::solve(Function& u,
                           VariationalProblem& problem,
                           double tol,
                           Form& M,
                           ErrorControl& ec,
                           Parameters& parameters)
{
  // Set tolerance parameter if not set
  if (parameters["tolerance"].change_count() == 0)
    parameters["tolerance"] = tol;

  // A list of adaptive data
  std::vector<AdaptiveDatum> data;

  // Start adaptive loop
  const uint N = parameters["max_iterations"];
  for (uint i = 0; i < N; i++)
  {
    //--- Stage 0: Solve primal problem ---
    info("");
    begin("Stage %d.0: Solving primal problem...", i);
    problem.solve(u);

    // Extract function space and mesh
    const FunctionSpace& V(u.function_space());
    const Mesh& mesh(V.mesh());

    // FIXME: Init mesh (should only initialize required stuff.)
    mesh.init();
    end();

    //--- Stage 1: Estimate error ---
    begin("Stage %d.1: Estimating error...", i);
    const double error_estimate = ec.estimate_error(u, problem.bcs());

    // Evaluate functional value
    if (!problem.is_nonlinear())
      M.set_coefficient(M.num_coefficients()-1, u);
    M.set_mesh(u.function_space().mesh()); // FIXME
    const double functional_value = assemble(M);

    // Initialize adaptive data
    AdaptiveDatum datum(i, V.dim(), mesh.num_cells(), error_estimate,
                        parameters["tolerance"],
                        functional_value);
    if (parameters["reference"].change_count() > 0)
      datum.set_reference_value(parameters["reference"]);
    data.push_back(datum);

    // Stop if stopping criterion is satisfied
    if (stop(V, error_estimate, parameters))
    {
      end();
      summary(data, parameters);
      return;
    }
    info(INFO, "Estimated error (%0.5g) does not satisfy tolerance (%0.5g).",
         error_estimate, tol);
    end();

    //--- Stage 2: Compute error indicators ---
    begin("Stage %d.2: Computing indicators...", i);

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

    Mesh new_mesh = refine(mesh, markers);
    if (parameters["plot_mesh"])
      plot(new_mesh, "Refined mesh");

    end();

    //--- Stage 5: Update forms ---
    begin("Stage %d.5: Updating forms...", i);

    info("Updating forms to new mesh. Dc. Logg will fix...");

    end();
  }

  summary(data, parameters);
  info(WARNING,
       "Maximal number of iterations (%d) exceeded! Returning anyhow.", N);
}
//-----------------------------------------------------------------------------
bool AdaptiveSolver::stop(const FunctionSpace& V,
                          const double error_estimate,
                          const Parameters& parameters)
{
  // Done if error is less than tolerance
  const double tolerance = parameters["tolerance"];
  if (std::abs(error_estimate) < tolerance)
    return true;

  // Or done if dimension is larger than max dimension (and that
  // parameter is set).
  const uint max_dimension = parameters["max_dimension"];
  if (parameters["max_dimension"].change_count() > 0
      && V.dim() > max_dimension)
    return true;

  // Otherwise, not done.
  return false;
}
//-----------------------------------------------------------------------------
void AdaptiveSolver::summary(const std::vector<AdaptiveDatum>& data,
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
