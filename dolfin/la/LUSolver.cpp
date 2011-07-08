// Copyright (C) 2010 Garth N. Wells
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
// First added:  2010-07-11
// Last changed:

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include "DefaultFactory.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LUSolver::LUSolver(std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset(factory.create_lu_solver());

  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(boost::shared_ptr<const GenericMatrix> A, std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset(factory.create_lu_solver());
  solver->set_operator(A);

  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
LUSolver::~LUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LUSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(const GenericMatrix& A, GenericVector& x,
                             const GenericVector& b)
{
  assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
