// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-04-21
// Last changed: 2008-04-28

#ifdef HAS_TRILINOS

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "EpetraVector.h"
#include "EpetraMatrix.h"
#include "GenericSparsityPattern.h"
#include "EpetraSparsityPattern.h"
#include "EpetraFactory.h"
//#include <dolfin/MPI.h>

#include <Epetra_CrsGraph.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix():
    Variable("A", "Epetra matrix"),
    A(0), is_view(false)
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(uint M, uint N):
    Variable("A", "Epetra matrix"),
    A(0), is_view(false)
{
  // TODO: call Epetra_Init or something?
  // Create Epetra matrix
  init(M, N);
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const EpetraMatrix& A):
  Variable("A", "Epetra matrix"),
  A(0), is_view(true)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(Epetra_FECrsMatrix* A):
    Variable("A", "a sparse matrix"),
    A(A), is_view(true)
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const Epetra_CrsGraph& graph):
    Variable("A", "a sparse matrix"),
    A(0), is_view(false)
{
  // TODO: call Epetra_Init or something?
  A = new Epetra_FECrsMatrix(Copy, graph);
}
//-----------------------------------------------------------------------------
EpetraMatrix::~EpetraMatrix()
{
  // Free memory of matrix
  if (!is_view) delete A;
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if (A) delete A;

  // Not yet implemented
  error("EpetraMatrix::init(uint, unit) not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  const EpetraSparsityPattern& epetra_pattern = dynamic_cast<const EpetraSparsityPattern&>(sparsity_pattern);
  A = new Epetra_FECrsMatrix(Copy, epetra_pattern.pattern());
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraMatrix::copy() const
{
  EpetraMatrix* mcopy = EpetraFactory::instance().createMatrix();

  //MatDuplicate(A, MAT_COPY_VALUES, &(mcopy->A));
  // Not yet implemented
  error("EpetraMatrix::copy not yet implemented.");

  return mcopy;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraMatrix::size(uint dim) const
{
  dolfin_assert(A); 
  int M = A->NumGlobalRows();
  int N = A->NumGlobalCols();
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::get(real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols) const
{
  dolfin_assert(A); 
  // for each row in rows
  //A->ExtractGlobalRowCopy(...)

  // Not yet implemented
  error("EpetraMatrix::get not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::set(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  dolfin_assert(A); 
  A->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
			 n, reinterpret_cast<const int*>(cols), 
			 block);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::add(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  dolfin_assert(A); 
  /*
  int err = A->InsertGlobalValues(m, reinterpret_cast<const int*>(rows), 
                                   n, reinterpret_cast<const int*>(cols), block);

  */


  int err = A->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), 
                                   n, reinterpret_cast<const int*>(cols), block);


  if (err!= 0) error("Did not manage to put the values into the matrix"); 

}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero()
{
  std::cout <<"in zero "<<std::endl; 
  dolfin_assert(A); 
  A->PutScalar(0.0);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::apply()
{
  dolfin_assert(A); 
  A->GlobalAssemble();
  //A->OptimizeStorage(); // TODO
}
//-----------------------------------------------------------------------------
void EpetraMatrix::disp(uint precision) const
{
  dolfin_assert(A); 
  A->Print(std::cout); 
}
//-----------------------------------------------------------------------------
void EpetraMatrix::ident(uint m, const uint* rows)
{
  dolfin_assert(A); 
  double* values;
  int* indices; 
  int row_size; 
  int r;

  for (uint i=0; i<m; i++){
    r = rows[i]; 
    A->ExtractMyRowView(r, row_size, values, indices); 
    memset(values, 0,  row_size*sizeof(double)); 
    for (uint j=0; j<m; j++) {
      if (r == indices[j]) {
        values[j] = 1.0; 
        break; 
      }
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero(uint m, const uint* rows)
{
  dolfin_assert(A); 
  double* values; 
  int* indices; 
  int row_size; 
  int r;

  for (uint i=0; i<m; i++){
    r = rows[i]; 
    A->ExtractMyRowView(r, row_size, values, indices); 
    memset(values, 0,  row_size*sizeof(double)); 
  }
}

//-----------------------------------------------------------------------------
void EpetraMatrix::mult(const GenericVector& x_, GenericVector& Ax_, bool transposed) const
{
  dolfin_assert(A); 

  const EpetraVector* x = dynamic_cast<const EpetraVector*>(x_.instance());  
  if (!x) error("The vector x should be of type EpetraVector");  

  EpetraVector* Ax = dynamic_cast<EpetraVector*>(Ax_.instance());  
  if (!Ax) error("The vector Ax should be of type EpetraVector");  

  A->Multiply(transposed, x->vec(), Ax->vec());
}

//-----------------------------------------------------------------------------
void EpetraMatrix::getrow(uint row, Array<uint>& columns, Array<real>& values) const
{
  dolfin_assert(A); 

  // temporal variables
  int *indices; 
  double* vals; 
  int* num_entries = new int; 

  // extract data from Epetra matrix 
  int err = A->ExtractMyRowView(row, *num_entries, vals, indices); 
  if (err!= 0) error("Did not manage to get a copy of the row."); 

  // put data in columns and values
  columns.clear();
  values.clear(); 
  for (int i=0; i< *num_entries; i++)
  {
    columns.push_back(indices[i]);
    values.push_back(vals[i]);
  }

  delete num_entries; 
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraMatrix::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
Epetra_FECrsMatrix& EpetraMatrix::mat() const
{
  dolfin_assert(A); 
  return *A;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator*= (real a)
{
  dolfin_assert(A);
  A->Scale(a);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator/= (real a)
{
  dolfin_assert(A);
  A->Scale(1.0 / a);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const Epetra_FECrsMatrix& A)
{
  error("operator << EpetraMatrix not implemented yet"); 
  return stream;
}
//-----------------------------------------------------------------------------

#endif
