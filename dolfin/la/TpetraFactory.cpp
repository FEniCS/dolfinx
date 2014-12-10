// Copyright (C) 2014
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
// First added:  2014-12-06

#ifdef HAS_TRILINOS

#include "SparsityPattern.h"
#include "TpetraVector.h"
#include "TpetraMatrix.h"
#include "TpetraFactory.h"

using namespace dolfin;

// Singleton instance
TpetraFactory TpetraFactory::factory;

//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> TpetraFactory::create_matrix() const
{
  std::shared_ptr<GenericMatrix> A(new TpetraMatrix);
  return A;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> TpetraFactory::create_vector() const
{
  std::shared_ptr<GenericVector> x(new TpetraVector);
  return x;
}
//-----------------------------------------------------------------------------
std::shared_ptr<TensorLayout>
TpetraFactory::create_layout(std::size_t rank) const
{
  bool sparsity = false;
  if (rank > 1)
    sparsity = true;
  std::shared_ptr<TensorLayout> pattern(new TensorLayout(0, sparsity));
  return pattern;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearOperator>
TpetraFactory::create_linear_operator() const
{
  std::shared_ptr<GenericLinearOperator> A; //(new TpetraLinearOperator);
  return A;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLUSolver>
TpetraFactory::create_lu_solver(std::string method) const
{
  std::shared_ptr<GenericLUSolver> solver; //(new TpetraLUSolver(method));
  return solver;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
TpetraFactory::create_krylov_solver(std::string method,
                                   std::string preconditioner) const
{
  std::shared_ptr<GenericLinearSolver>
    solver; //(new TpetraKrylovSolver(method, preconditioner));
  return solver;
}
//-----------------------------------------------------------------------------

#endif
