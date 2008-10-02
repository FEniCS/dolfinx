// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-04-21
// Last changed: 2008-05-15

#ifdef HAS_TRILINOS

#include <cstring>
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
#include <dolfin/common/Timer.h>

#include <Epetra_CrsGraph.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <EpetraExt_MatrixMatrix.h>

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
  resize(M, N);
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const EpetraMatrix& A):
  Variable("A", "Epetra matrix"),
  A(0), is_view(false)
{
  if (&A.mat())
    this->A = new Epetra_FECrsMatrix(A.mat());
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
  if (!is_view) 
    delete A;
}
//-----------------------------------------------------------------------------
void EpetraMatrix::resize(uint M, uint N)
{
  // Not yet implemented
  error("EpetraMatrix::resize(uint, unit) not yet implemented.");
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
  //Epetra_FECrsMatrix* copy = new Epetra_FECrsMatrix(*A); 
  EpetraMatrix* mcopy = new EpetraMatrix(*this);
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
  
  int max_num_indices = A->MaxNumEntries();
  int num_entities = 0;
  int * indices  = new int[max_num_indices];
  double * values  = new double[max_num_indices];
  
  // For each row in rows
  for(uint i = 0; i < m; ++i)
  {
    // Extract the values and indices from row: rows[i]
    if (A->IndicesAreLocal())
      A->ExtractMyRowView(rows[i], num_entities, values, indices);
    else
      A->ExtractGlobalRowView(rows[i], num_entities, values, indices);
    int k = 0;
    // Step the indices to the start of cols
    while (indices[k] < static_cast<int>(cols[0]))
      k++;
    // Fill the collumns in the block 
    for (uint j = 0; j < n; j++)
    {
      if (k < num_entities and indices[k] == static_cast<int>(cols[j]))
      {
	block[i*n + j] = values[k];
	k++;
      }
      else
	block[i*n + j] = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::set(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  dolfin_assert(A); 
  int err = A->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
			 n, reinterpret_cast<const int*>(cols), 
			 block);
  if (err!= 0) error("Did not manage to set the values into the matrix"); 
}
//-----------------------------------------------------------------------------
void EpetraMatrix::add(const real* block,
		       uint m, const uint* rows,
		       uint n, const uint* cols)
{
  Timer t0("Matrix add"); 
  dolfin_assert(A); 

  int err = A->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), 
                                   n, reinterpret_cast<const int*>(cols), block, 
                                   Epetra_FECrsMatrix::ROW_MAJOR
                                   );

  if (err!= 0) error("Did not manage to put the values into the matrix"); 
}
//-----------------------------------------------------------------------------
void EpetraMatrix::axpy(real a, const GenericMatrix& A)
{
  const EpetraMatrix* AA = &A.down_cast<EpetraMatrix>();
  dolfin_assert(AA->mat().NumGlobalNonzeros() == this->A->NumGlobalNonzeros() and
		AA->mat().NumGlobalCols() == this->A->NumGlobalCols() and
		AA->mat().NumGlobalRows() == this->A->NumGlobalRows() );
  EpetraExt::MatrixMatrix::Add(AA->mat(),false,a,*(this->A),1.0);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero()
{
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
  int* row_size = new int; 
  int r;

  for (uint i=0; i<m; i++){
    r = rows[i]; 
    int err = A->ExtractMyRowView(r, *row_size, values, indices); 
    if (err!= 0) error("Trouble with ExtractMyRowView in EpetraMatrix::ident."); 
    memset(values, 0,  (*row_size)*sizeof(double)); 
    for (uint j=0; j<m; j++) {
      if (r == indices[j]) {
        values[j] = 1.0; 
        break; 
      }
    }
  }
  delete row_size; 
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero(uint m, const uint* rows)
{
  dolfin_assert(A); 
  double* values; 
  int* indices; 
  int* row_size = new int; 
  int r;

  for (uint i=0; i<m; i++){
    r = rows[i]; 
    int err = A->ExtractMyRowView(r, *row_size, values, indices); 
    if (err!= 0) error("Trouble with ExtractMyRowView in EpetraMatrix::ident."); 
    memset(values, 0, (*row_size)*sizeof(double)); 
  }
  delete row_size; 
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
void EpetraMatrix::setrow(uint row, const Array<uint>& columns, const Array<real>& values)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraMatrix::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
Epetra_FECrsMatrix& EpetraMatrix::mat() const
{
  // FIXME: Shouldn't this function just return A, not a dereferenced version?
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
const GenericMatrix& EpetraMatrix::operator= (const GenericMatrix& A)
{
  *this = A.down_cast<EpetraMatrix>();
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator= (const EpetraMatrix& A)
{
  dolfin_assert(&A.mat());
  *(this->A) = A.mat();
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const EpetraMatrix& A)
{
  stream << "[ Epetra matrix of size " << A.size(0) << " x " << A.size(1) << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
#endif
