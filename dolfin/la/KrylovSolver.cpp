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
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2011-03-24

#include <dolfin/common/Timer.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "DefaultFactory.h"
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
  p.add("report",                  true); /* deprecate? */
  p.add("monitor_convergence",     false);
  p.add("error_on_nonconvergence", true);
  p.add("nonzero_initial_guess",   false);

  // GMRES options
  Parameters p_gmres("gmres");
  p_gmres.add("restart", 30);
  p.add(p_gmres);

  // General preconditioner options
  Parameters p_pc("preconditioner");
  p_pc.add("shift_nonzero",        0.0);
  p_pc.add("reuse",                false);
  p_pc.add("same_nonzero_pattern", false);
  p_pc.add("report",               false);

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
KrylovSolver::KrylovSolver(std::string solver_type, std::string pc_type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset(factory.create_krylov_solver(solver_type, pc_type));
  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void KrylovSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void KrylovSolver::set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                 const boost::shared_ptr<const GenericMatrix> P)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  //check_dimensions(solver->get_operator(), x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  assert(solver);
  //check_dimensions(A, x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
