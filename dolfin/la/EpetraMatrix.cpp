// Copyright (C) 2008 Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-24
// Last changed: 2008-01-24

#ifdef HAS_TRILINOS

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "EpetraVector.h"
#include "EpetraMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "EpetraFactory.h"
//#include <dolfin/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix()
  : GenericMatrix(), 
    Variable("A", "a sparse matrix"),
    A(0), _copy(false)
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(uint M, uint N)
  : GenericMatrix(),
    Variable("A", "a sparse matrix"),
    A(0), _copy(false)
{
  // TODO: call Epetra_Init or something?
  // Create Epetra matrix
  init(M, N);
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(Epetra_FECrsMatrix* A)
  : GenericMatrix(),
    Variable("A", "a sparse matrix"),
    A(A), _copy(true)
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const Epetra_CrsGraph& graph)
  : GenericMatrix(),
    Variable("A", "a sparse matrix"),
    A(0), _copy(false)
{
  // TODO: call Epetra_Init or something?
  A = new Epetra_FECrsMatrix(Copy, graph);
}
//-----------------------------------------------------------------------------
EpetraMatrix::~EpetraMatrix()
{
  // Free memory of matrix
  if (!_copy) delete A;
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if (A) delete A;

  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraMatrix::create() const
{
  return new EpetraMatrix();
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraMatrix::copy() const
{
  EpetraMatrix* mcopy = create();

  //MatDuplicate(A, MAT_COPY_VALUES, &(mcopy->A));
  // Not yet implemented
  error("Not yet implemented.");

  return mcopy;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraMatrix::size(uint dim) const
{
  int M = A->NumGlobalRows();
  int N = A->NumGlobalCols();
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::get(real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols) const
{
  // for each row in rows
  //A->ExtractGlobalRowCopy(...)

  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::set(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  A->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
			 n, reinterpret_cast<const int*>(cols), 
			 block);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::add(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  A->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows),
			 n, reinterpret_cast<const int*>(cols),
			 block);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero()
{
  A->PutScalar(0.0);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::apply()
{
  A->GlobalAssemble();
  //A->OptimizeStorage(); // TODO
}
//-----------------------------------------------------------------------------
void EpetraMatrix::disp(uint precision) const
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::ident(uint m, const uint* rows)
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero(uint m, const uint* rows)
{
  // Not yet implemented
  error("Not yet implemented.");
}

//-----------------------------------------------------------------------------
void EpetraMatrix::mult(const GenericVector& x_, GenericVector& Ax_, bool transposed) const
{
  const EpetraVector* x = dynamic_cast<const EpetraVector*>(x_.instance());  
  if (!x) error("The vector x should be of type EpetraVector");  

  EpetraVector* Ax = dynamic_cast<EpetraVector*>(Ax_.instance());  
  if (!Ax) error("The vector Ax should be of type EpetraVector");  

  A->Multiply(transposed, x->vec(), Ax->vec());
}

//-----------------------------------------------------------------------------
void EpetraMatrix::getRow(uint i, int& ncols, Array<int>& columns, 
                         Array<real>& values) const
{
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraMatrix::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------




#endif
