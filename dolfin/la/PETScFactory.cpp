// Copyright (C) 2005-2006 Ola Skavhaug
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
// Modified by Anders Logg 2009-2011
//
// First added:  2007-12-06
// Last changed: 2011-10-19

#ifdef HAS_PETSC

#include "SparsityPattern.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScFactory.h"

using namespace dolfin;

// Singleton instance
PETScFactory PETScFactory::factory;

//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> PETScFactory::create_matrix() const
{
  boost::shared_ptr<GenericMatrix> A(new PETScMatrix);
  return A;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> PETScFactory:: create_vector() const
{
  boost::shared_ptr<GenericVector> x(new PETScVector("global"));
  return x;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> PETScFactory:: create_local_vector() const
{
  boost::shared_ptr<GenericVector> x(new PETScVector("local"));
  return x;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericSparsityPattern> PETScFactory::create_pattern() const
{
  boost::shared_ptr<GenericSparsityPattern> pattern(new SparsityPattern(0, true));
  return pattern;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericLUSolver> PETScFactory::create_lu_solver(std::string method) const
{
  boost::shared_ptr<GenericLUSolver> solver(new PETScLUSolver(method));
  return solver;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericLinearSolver>
PETScFactory::create_krylov_solver(std::string method,
                                   std::string preconditioner) const
{
  boost::shared_ptr<GenericLinearSolver>
    solver(new PETScKrylovSolver(method, preconditioner));
  return solver;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScFactory::lu_solver_methods() const
{
  return PETScLUSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScFactory::krylov_solver_methods() const
{
  return PETScKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScFactory::krylov_solver_preconditioners() const
{
  return PETScKrylovSolver::preconditioners();
}
//-----------------------------------------------------------------------------

#endif
