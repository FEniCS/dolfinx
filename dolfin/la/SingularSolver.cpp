// Copyright (C) 2008-2009 Anders Logg
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
// Modified by Garth N. Wells, 2010.
//
// First added:  2008-05-15
// Last changed: 2011-03-17

#include <dolfin/common/Array.h>
#include <dolfin/common/MPI.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "SparsityPattern.h"
#include "SingularSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SingularSolver::SingularSolver(std::string solver_type,
                               std::string pc_type)
  : linear_solver(solver_type, pc_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SingularSolver::~SingularSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint SingularSolver::solve(const GenericMatrix& A,
                                   GenericVector& x,
                                   const GenericVector& b)
{
  log(TRACE, "Solving singular system...");

  // Propagate parameters
  linear_solver.parameters.update(parameters("linear_solver"));

  // Initialize data structures for extended system
  init(A);

  // Create extended system
  create(A, b, 0);

  // Solve extended system
  const uint num_iterations = linear_solver.solve(*B, *y, *c);

  // Extract solution
  x.resize(y->size() - 1);
  Array<double> vals(y->size());
  y->get_local(vals);
  x.set_local(vals);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint SingularSolver::solve(const GenericMatrix& A,
                                   GenericVector& x,
                                   const GenericVector& b,
                                   const GenericMatrix& M)
{
  log(TRACE, "Solving singular system...");

  // Propagate parameters
  linear_solver.parameters.update(parameters("linear_solver"));

  // Initialize data structures for extended system
  init(A);

  // Create extended system
  create(A, b, &M);

  // Solve extended system
  const uint num_iterations = linear_solver.solve(*B, *y, *c);

  // Extract solution
  x.resize(y->size() - 1);
  Array<double> vals(y->size());
  y->get_local(vals);
  x.set_local(vals);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void SingularSolver::init(const GenericMatrix& A)
{
  // Check size of system
  if (A.size(0) != A.size(1))
    error("Matrix must be square.");
  if (A.size(0) == 0)
    error("Matrix size must be non-zero.");

  // Get dimension
  const uint N = A.size(0);

  // Check if we have already initialized system
  if (B && B->size(0) == N + 1 && B->size(1) == N + 1)
    return;

  // Create sparsity pattern for B
  SparsityPattern s;
  std::vector<uint> dims(2);
  std::vector<std::pair<uint, uint> > local_range(2);
  std::vector<const boost::unordered_map<uint, uint>* > off_process_owner(2);
  const boost::unordered_map<uint, uint> empty_off_process_owner;
  for (uint i = 0; i < 1; ++i)
  {
    dims[i] = N + 1;
    local_range[i] = MPI::local_range(dims[i]);
    off_process_owner[i] = &empty_off_process_owner;
  }
  s.init(dims, local_range, off_process_owner);

  // Copy sparsity pattern for A and last column
  std::vector<uint> columns;
  std::vector<double> dummy;
  std::vector<const std::vector<uint>* > _rows(2);
  std::vector<std::vector<uint> > rows(2);
  rows[0].resize(1);
  for (uint i = 0; i < N; i++)
  {
    // FIXME: Add function to get row sparsity pattern
    // Get row
    A.getrow(i, columns, dummy);

    // Copy columns to vector
    const uint num_cols = columns.size() + 1;
    rows[1].resize(num_cols);
    std::copy(columns.begin(), columns.end(), rows[1].begin());

    // Add last entry
    rows[1][num_cols - 1] = N;

    // Set row index
    rows[0][0] = i;

    // Insert into sparsity pattern
    _rows[0] = &rows[0];
    _rows[1] = &rows[1];
    s.insert(_rows);
  }

  // Add last row
  const uint num_cols = N;
  rows[1].resize(num_cols);
  std::copy(columns.begin(), columns.end(), rows[1].begin());
  rows[0][0] = N;
  _rows[0] = &rows[0];
  _rows[1] = &rows[1];
  s.insert(_rows);

  // Create matrix and vector
  B.reset(A.factory().create_matrix());
  y.reset(A.factory().create_vector());
  c.reset(A.factory().create_vector());
  B->init(s);
  y->resize(N + 1);
  c->resize(N + 1);

  // FIXME: Do these need to be zeroed?
  y->zero();
  c->zero();
}
//-----------------------------------------------------------------------------
void SingularSolver::create(const GenericMatrix& A, const GenericVector& b,
                            const GenericMatrix* M)
{
  assert(B);
  assert(c);

  log(TRACE, "Creating extended hopefully non-singular system...");

  // Reset matrix
  B->zero();

  // Copy rows from A into B
  const uint N = A.size(0);
  std::vector<uint> columns;
  std::vector<double> values;
  for (uint i = 0; i < N; i++)
  {
    A.getrow(i, columns, values);
    B->setrow(i, columns, values);
  }

  // Compute lumped mass matrix
  columns.resize(N);
  values.resize(N);
  if (M)
  {
    boost::scoped_ptr<GenericVector> ones(A.factory().create_vector());
    boost::scoped_ptr<GenericVector> z(A.factory().create_vector());
    ones->resize(N);
    *ones = 1.0;
    z->resize(N);
    // FIXME: Do we need to zero z?
    z->zero();
    M->mult(*ones, *z);
    for (uint i = 0; i < N; i++)
    {
      columns[i] = i;
      values[i] = (*z)[i];
    }
  }
  else
  {
    for (uint i = 0; i < N; i++)
    {
      columns[i] = i;
      values[i] = 1.0;
    }
  }

  // Add last row
  B->setrow(N, columns, values);

  // Add last column
  for (uint i = 0; i < N; i++)
    B->set(&values[i], 1, &i, 1, &N);

  // Copy values from b into c
  Array<double> vals(N + 1);
  b.get_local(vals);
  vals[N] = 0.0;
  c->set_local(vals);

  // Apply changes
  B->apply("insert");
  c->apply("insert");
}
//-----------------------------------------------------------------------------
