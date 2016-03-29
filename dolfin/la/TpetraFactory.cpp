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

#include "BelosKrylovSolver.h"
#include "SparsityPattern.h"
#include "TpetraMatrix.h"
#include "TpetraVector.h"
#include "TpetraFactory.h"
#include "Amesos2LUSolver.h"

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
std::shared_ptr<GenericVector> TpetraFactory::create_vector(MPI_Comm comm) const
{
  return std::make_shared<TpetraVector>(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<TensorLayout>
TpetraFactory::create_layout(std::size_t rank) const
{
  TensorLayout::Sparsity sparsity = TensorLayout::Sparsity::DENSE;
  if (rank > 1)
    sparsity = TensorLayout::Sparsity::SPARSE;
  std::shared_ptr<TensorLayout> pattern(new TensorLayout(0, sparsity));
  return pattern;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearOperator>
TpetraFactory::create_linear_operator() const
{
  std::shared_ptr<GenericLinearOperator> A; //(new TpetraLinearOperator);
  dolfin_not_implemented();

  return A;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLUSolver>
TpetraFactory::create_lu_solver(std::string method) const
{
  std::shared_ptr<GenericLUSolver> solver(new Amesos2LUSolver(method));
  return solver;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
TpetraFactory::create_krylov_solver(std::string method,
                                    std::string preconditioner) const
{
  std::shared_ptr<GenericLinearSolver>
    solver(new BelosKrylovSolver(method, preconditioner));
  return solver;
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> TpetraFactory::lu_solver_methods() const
{
  return Amesos2LUSolver::methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> TpetraFactory::krylov_solver_methods() const
{
  return BelosKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string>
TpetraFactory::krylov_solver_preconditioners() const
{
  return BelosKrylovSolver::preconditioners();
}
//-----------------------------------------------------------------------------

#endif
