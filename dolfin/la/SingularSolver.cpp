// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-15
// Last changed: 2008-05-15

#include <dolfin/common/Array.h>
#include "LinearAlgebraFactory.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "SparsityPattern.h"
#include "SingularSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SingularSolver::SingularSolver(SolverType solver_type,
                               PreconditionerType pc_type)
  : Parametrized(), linear_solver(solver_type, pc_type), B(0), y(0), c(0)
{
  // Set parameters for linear solver
  linear_solver.set("parent", *this);
}
//-----------------------------------------------------------------------------
SingularSolver::~SingularSolver()
{
  delete B;
  delete y;
  delete c;
}
//-----------------------------------------------------------------------------
dolfin::uint SingularSolver::solve(const GenericMatrix& A,
                                   GenericVector& x, const GenericVector& b)
{
  message("Solving singular system...");

  // Initialize data structures for extended system
  init(A);

  // Create extended system
  create(A, b, 0);
  
  // Solve extended system
  const uint num_iterations = linear_solver.solve(*B, *y, *c);

  // Extract solution
  x.init(y->size() - 1);
  real* vals = new real[y->size()];
  y->get(vals);
  x.set(vals);
  delete [] vals;

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint SingularSolver::solve(const GenericMatrix& A,
                                   GenericVector& x, const GenericVector& b,
                                   const GenericMatrix& M)
{
  message("Solving singular system...");

  // Initialize data structures for extended system
  init(A);

  // Create extended system
  create(A, b, &M);

  // Solve extended system
  const uint num_iterations = linear_solver.solve(*B, *y, *c);

  // Extract solution
  x.init(y->size() - 1);
  real* vals = new real[y->size()];
  y->get(vals);
  x.set(vals);
  delete [] vals;

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

  cout << "Initializing" << endl;
  
  // Delete any old data
  delete B;
  delete y;
  delete c;

  // Create sparsity pattern for B
  SparsityPattern s(N + 1, N + 1);

  // Copy sparsity pattern for A and last column
  Array<uint> columns;
  Array<real> dummy;
  for (uint i = 0; i < N; i++)
  {
    // Get row
    A.getrow(i, columns, dummy);

    // Copy columns to array
    const uint num_cols = columns.size() + 1;
    uint* cols = new uint[num_cols];
    for (uint j = 0; j < columns.size(); j++)
      cols[j] = columns[j];

    // Add last entry
    cols[num_cols - 1] = N;

    // Insert into sparsity pattern
    s.insert(1, &i, num_cols, cols);

    // Delete temporary array
    delete [] cols;
  }

  // Add last row
  const uint num_cols = N;
  uint* cols = new uint[num_cols];
  for (uint j = 0; j < num_cols; j++)
    cols[j] = j;
  const uint row = N;
  s.insert(1, &row, num_cols, cols);
  delete [] cols;

  // Create matrix and vector
  B = A.factory().createMatrix();
  y = A.factory().createVector();
  c = A.factory().createVector();
  B->init(s);
  y->init(N + 1);
  c->init(N + 1);
}
//-----------------------------------------------------------------------------
void SingularSolver::create(const GenericMatrix& A, const GenericVector& b,
                            const GenericMatrix* M)
{
  dolfin_assert(B);
  dolfin_assert(c);

  // Reset matrix
  B->zero();

  // Copy rows from A into B
  const uint N = A.size(0);
  Array<uint> columns;
  Array<real> values;
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
    GenericVector* ones = A.factory().createVector();
    GenericVector* z = A.factory().createVector();
    ones->init(N);
    z->init(N);
    *ones = 1.0;
    A.mult(*ones, *z);
    for (uint i = 0; i < N; i++)
    {
      columns[i] = i;
      values[i] = (*z)[i];
    }
    delete ones;
    delete z;
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
  real* vals = new real[N + 1];
  b.get(vals);
  vals[N] = 0.0;
  c->set(vals);
  delete [] vals;

  // Apply changes
  B->apply();
  c->apply();
}
//-----------------------------------------------------------------------------
