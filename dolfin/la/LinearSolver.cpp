// Copyright (C) 2008 Anders Logg
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
// Modified by Johan Hake, 2010.
// Modified by Garth N. Wells, 2010.
//
// First added:  2008-05-10
// Last changed: 2011-03-24

#include "KrylovSolver.h"
#include "LUSolver.h"
#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(std::string solver_type, std::string pc_type)
{
  // Choose solver and set parameters
  if (solver_type == "lu" || solver_type == "cholesky")
    solver.reset(new LUSolver(solver_type));
  else
    solver.reset(new KrylovSolver(solver_type, pc_type));

  // Get parameters
  parameters = solver->parameters;

  // Rename the parameters
  parameters.rename("linear_solver");
}
//-----------------------------------------------------------------------------
LinearSolver::~LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operator(const GenericMatrix& A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operators(const GenericMatrix& A, const GenericMatrix& P)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
const GenericMatrix& LinearSolver::get_operator() const
{
  assert(solver);
  return solver->get_operator();
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  assert(solver);
  check_dimensions(A, x, b);

  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  check_dimensions(get_operator(), x, b);

  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
