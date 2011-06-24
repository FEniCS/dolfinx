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
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-08-19
// Last changed: 2011-06-22

// #include <dolfin/common/utils.h>
// #include <dolfin/common/Variable.h>
// #include <dolfin/log/Table.h>
// #include <dolfin/fem/Form.h>
// #include <dolfin/fem/assemble.h>
// #include <dolfin/function/FunctionSpace.h>
// #include <dolfin/function/Function.h>
// #include <dolfin/la/Vector.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/mesh/MeshEntity.h>
// #include <dolfin/mesh/MeshFunction.h>
// #include <dolfin/plot/plot.h>
// #include "adapt.h"
// #include "marking.h"

#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalSolver.h>

#include "AdaptiveLinearVariationalSolver.h"
#include "GoalFunctional.h"
#include "ErrorControl.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveLinearVariationalSolver::
AdaptiveLinearVariationalSolver(LinearVariationalProblem& problem)
  : problem(reference_to_no_delete_pointer(problem))
{
  // Set parameters
}
//-----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::solve(const double tol, GoalFunctional& M)
{
  // Initialize goal functional
  boost::shared_ptr<const Form> a = problem->bilinear_form();
  boost::shared_ptr<const Form> L = problem->linear_form();
  M.update_ec(*a, *L);

  // Extract error control from goal functional
  ErrorControl& ec(*(M._ec));

  // Call solve with given error control
  AdaptiveLinearVariationalSolver::solve(tol, M, ec);
}
//-----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::
solve(const double tol, GoalFunctional& M, ErrorControl& ec)
{

  // A list of adaptive data
  std::vector<AdaptiveDatum> data;

  // Start adaptive loop
  const uint max_iterations = 10;//parameters["max_iterations"];

  // Iterate over a series of meshes
  for (uint i = 0; i < max_iterations; i++)
  {
    // Deal with problem, goal and error control on current mesh
    //const VariationalProblem& this_problem = problem.fine();
    //Form& this_goal = M.fine();
    //ErrorControl& this_ec = ec.fine();

    // Solve primal problem ---
    begin("Stage %d.0: Solving primal problem ...", i);
    solve_primal();
    end();

    //--- Stage 1: Estimate error ---
    begin("Stage %d.1: Computing error estimate...", i);
    const double error_estimate = ec.estimate_error(*(problem->solution()),
                                                    problem->bcs());
    std::cout << "eta_h = " << error_estimate << std::endl;
    end();

  }
}
//-----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::solve_primal()
{
  //LinearVariationalSolver solver(*this_problem);
  LinearVariationalSolver solver(*problem);
  solver.solve();
}
//-----------------------------------------------------------------------------

// //-----------------------------------------------------------------------------
// void AdaptiveLinearVariationalSolver::solve(const LinearVariationalProblem& pde,
//                                       const double tol,
//                                       Form& goal,
//                                       ErrorControl& control,
//                                       const Parameters& parameters)
// {
//   error("Adaptive solver needs to be updated after change to variational problem interface.");

//   /*

//   // Set tolerance parameter if not set
//   //if (parameters["tolerance"].change_count() == 0)
//   //  parameters["tolerance"] = tol;

//   // A list of adaptive data
//   std::vector<AdaptiveDatum> data;

//   // Start adaptive loop
//   const uint maxiter = parameters["max_iterations"];

//   for (uint i = 0; i < maxiter; i++)
//   {
//     // Update references to current
//     const VariationalProblem& problem = pde.fine();
//     Function& u = w.fine();
//     Form& M = goal.fine();
//     ErrorControl& ec = control.fine();

//     //--- Stage 0: Solve primal problem ---
//     info("");
//     begin("Stage %d.0: Solving primal problem...", i);
//     problem.solve(u);

//     // Extract function space and mesh
//     const FunctionSpace& V(u.function_space());
//     const Mesh& mesh(V.mesh());
//     end();

//     //--- Stage 1: Estimate error ---
//     begin("Stage %d.1: Computing error estimate...", i);
//     const double error_estimate = ec.estimate_error(u, problem.bcs());

//     // Evaluate functional value
//     if (!problem.is_nonlinear())
//     {
//       boost::shared_ptr<const GenericFunction> _u(&u, NoDeleter());
//       M.set_coefficient(M.num_coefficients() - 1, _u);
//     }
//     const double functional_value = assemble(M);

//     // Initialize adaptive data
//     AdaptiveDatum datum(i, V.dim(), mesh.num_cells(), error_estimate,
//                         tol, functional_value);
//     if (parameters["reference"].change_count() > 0)
//       datum.set_reference_value(parameters["reference"]);
//     data.push_back(datum);

//     // Check stopping criterion
//     if (stop(V, error_estimate, tol, parameters))
//     {
//       end();
//       summary(data, parameters);
//       return;
//     }
//     info("Estimated error (%0.5g) does not satisfy tolerance (%0.5g).",
//          error_estimate, tol);
//     end();
//     summary(datum);


//     //--- Stage 2: Compute error indicators ---
//     begin("Stage %d.2: Computing error indicators...", i);
//     Vector indicators(mesh.num_cells());
//     ec.compute_indicators(indicators, u);
//     end();

//     //--- Stage 3: Mark mesh for refinement ---
//     begin("Stage %d.3: Marking mesh for refinement...", i);
//     MeshFunction<bool> markers(mesh, mesh.topology().dim());
//     const std::string strategy = parameters["marking_strategy"];
//     const double fraction = parameters["marking_fraction"];
//     mark(markers, indicators, strategy, fraction);
//     end();

//     //--- Stage 4: Refine mesh ---
//     begin("Stage %d.4: Refining mesh...", i);
//     adapt(mesh, markers);
//     if (parameters["plot_mesh"])
//       plot(mesh.child(), "Refined mesh");
//     end();

//     //--- Stage 5: Update forms ---
//     begin("Stage %d.5: Updating forms...", i);
//     adapt(problem, mesh.fine_shared_ptr());
//     adapt(u, mesh.fine_shared_ptr());
//     adapt(M, mesh.fine_shared_ptr());
//     adapt(ec, mesh.fine_shared_ptr());
//     end();
//   }

//   summary(data, parameters);
//   warning("Maximal number of iterations (%d) exceeded! Returning anyhow.", maxiter);
//   */
// }
// //-----------------------------------------------------------------------------
// void AdaptiveLinearVariationalSolver::solve(const NonlinearVariationalProblem& pde,
//                                       const double tol,
//                                       Form& goal,
//                                       ErrorControl& control,
//                                       const Parameters& parameters)
// {
//   // FIXME: Is there a way to implement this without needing to repeat
//   // FIXME: the implementation for linear problems?
// }
// //-----------------------------------------------------------------------------
// bool AdaptiveLinearVariationalSolver::stop(const FunctionSpace& V,
//                                      const double error_estimate,
//                                      const double tolerance,
//                                      const Parameters& parameters)
// {
//   // Done if error is less than tolerance
//   if (std::abs(error_estimate) < tolerance)
//     return true;

//   // Or done if dimension is larger than max dimension (and that
//   // parameter is set).
//   const uint max_dimension = parameters["max_dimension"];
//   if (parameters["max_dimension"].change_count() > 0
//       && V.dim() > max_dimension)
//   {
//     return true;
//   }
//   else
//     return false;
// }
// //-----------------------------------------------------------------------------
// void AdaptiveLinearVariationalSolver::summary(const std::vector<AdaptiveDatum>& data,
//                                         const Parameters& parameters)
// {
//   // Show parameters used
//   info("");
//   info("Parameters used for adaptive solve:");
//   info("");
//   info(parameters, true);

//   // Show summary for all iterations
//   info("");
//   info("Summary of adaptive solve:");
//   info("");
//   Table table("Level");
//   for (uint i = 0; i < data.size(); i++)
//     data[i].store(table);
//   info(indent(table.str(true)));
//   info("");
// }
// //-----------------------------------------------------------------------------
// void AdaptiveLinearVariationalSolver::summary(const AdaptiveDatum& datum)
// {
//   // Show summary for all iterations
//   info("");
//   info("Current adaptive data");
//   info("");
//   Table table("Level");
//   datum.store(table);
//   info(indent(table.str(true)));
//   info("");
// }
// //-----------------------------------------------------------------------------
