// Copyright (C) 2011 Fredrik Valdmanis
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
// First added:  2011-09-13 
// Last changed: 2012-02-29

#ifdef HAS_PETSC_CUSP

#include "SparsityPattern.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScCuspFactory.h"

using namespace dolfin;

// Singleton instance
PETScCuspFactory PETScCuspFactory::factory;

//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> PETScCuspFactory::create_matrix() const
{
  boost::shared_ptr<GenericMatrix> A(new PETScMatrix(true));
  return A;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> PETScCuspFactory:: create_vector() const
{
  boost::shared_ptr<GenericVector> x(new PETScVector("global", true));
  return x;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> PETScCuspFactory:: create_local_vector() const
{
  boost::shared_ptr<GenericVector> x(new PETScVector("local", true));
  return x;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<TensorLayout> PETScCuspFactory::create_layout(uint rank) const
{
  bool sparsity = false;
  if (rank > 1)
    sparsity = true;
  boost::shared_ptr<TensorLayout> pattern(new TensorLayout(0, sparsity));
  return pattern;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericLUSolver> PETScCuspFactory::create_lu_solver(std::string method) const
{
  boost::shared_ptr<GenericLUSolver> solver(new PETScLUSolver(method));
  return solver;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericLinearSolver>
PETScCuspFactory::create_krylov_solver(std::string method,
                                       std::string preconditioner) const
{
  boost::shared_ptr<GenericLinearSolver>
    solver(new PETScKrylovSolver(method, preconditioner));
  return solver;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScCuspFactory::lu_solver_methods() const
{
  return PETScLUSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScCuspFactory::krylov_solver_methods() const
{
  return PETScKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
  PETScCuspFactory::krylov_solver_preconditioners() const
{
  return PETScKrylovSolver::preconditioners();
}
//-----------------------------------------------------------------------------

#endif
