// Copyright (C) 2007-2010 Garth N. Wells
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
// Modified by Ola Skavhaug 2008
// Modified by Anders Logg 2008-2012
//
// First added:  2007-07-03
// Last changed: 2013-11-25

#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/log/LogStream.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "DefaultFactory.h"
#include "LinearSolver.h"
#include "VectorSpaceBasis.h"
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters KrylovSolver::default_parameters()
{
  Parameters p("krylov_solver");

  p.add("relative_tolerance",      1.0e-6);
  p.add("absolute_tolerance",      1.0e-15);
  p.add("divergence_limit",        1.0e4);
  p.add("maximum_iterations",      10000);
  p.add("report",                  true);
  p.add("monitor_convergence",     false);
  p.add("error_on_nonconvergence", true);
  p.add("nonzero_initial_guess",   false);

  // General preconditioner options
  Parameters p_pc("preconditioner");
  p_pc.add("shift_nonzero", 0.0);

  // Re-use options
  std::set<std::string> structure_options;
  structure_options.insert("same");
  structure_options.insert("same_nonzero_pattern");
  structure_options.insert("different_nonzero_pattern");
  p_pc.add("structure", "different_nonzero_pattern", structure_options);

  p_pc.add("report", false);

  // ILU preconditioner options
  Parameters p_pc_ilu("ilu");
  p_pc_ilu.add("fill_level", 0);

  // Schwartz preconditioner options
  Parameters p_pc_schwarz("schwarz");
  p_pc_schwarz.add("overlap", 1);

  // Add sub-preconditioner options
  p_pc.add(p_pc_ilu);
  p_pc.add(p_pc_schwarz);

  // Add preconditioner options
  p.add(p_pc);

  return p;
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(std::string method, std::string preconditioner)
{
  // Initialize solver
  init(method, preconditioner);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(std::shared_ptr<const GenericLinearOperator> A,
                           std::string method, std::string preconditioner)
{
  // Initialize solver
  init(method, preconditioner);

  // Set operator
  set_operator(A);
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void
KrylovSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  dolfin_assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void
KrylovSolver::set_operators(std::shared_ptr<const GenericLinearOperator> A,
                            std::shared_ptr<const GenericLinearOperator> P)
{
  dolfin_assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
std::size_t KrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(solver);
  //check_dimensions(solver->get_operator(), x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
std::size_t KrylovSolver::solve(const GenericLinearOperator& A,
                                GenericVector& x,
                                const GenericVector& b)
{
  dolfin_assert(solver);
  //check_dimensions(A, x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
void KrylovSolver::init(std::string method, std::string preconditioner)
{
  // Get default linear algebra factory
  DefaultFactory factory;

  // Get list of available methods and preconditioners
  const std::map<std::string, std::string>
    methods = factory.krylov_solver_methods();
  const std::map<std::string, std::string>
    preconditioners = factory.krylov_solver_preconditioners();

  // Check that method is available
  if (!LinearSolver::in_list(method, methods))
  {
    dolfin_error("KrylovSolver.cpp",
                 "solve linear system using Krylov iteration",
                 "Unknown Krylov method \"%s\". "
                 "Use list_krylov_solver_methods() to list available Krylov methods",
                 method.c_str());
  }

  // Check that preconditioner is available
  if (!LinearSolver::in_list(preconditioner, preconditioners))
  {
    dolfin_error("KrylovSolver.cpp",
                 "solve linear system using Krylov iteration",
                 "Unknown preconditioner \"%s\". "
                 "Use list_krylov_solver_preconditioners() to list available preconditioners()",
                 preconditioner.c_str());
  }

  // Set default parameters
  parameters = dolfin::parameters("krylov_solver");

  // Initialize solver
  solver = factory.create_krylov_solver(method, preconditioner);
  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
