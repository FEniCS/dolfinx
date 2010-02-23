// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2008, 2009.
//
// First added:  2008-04-21
// Last changed: 2009-09-08

#ifdef HAS_TRILINOS

#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Epetra_CrsGraph.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <EpetraExt_MatrixMatrix.h>

#include <ml_epetra_utils.h>

#include <dolfin/common/Timer.h>
#include <dolfin/log/dolfin_log.h>
#include "EpetraVector.h"
#include "GenericSparsityPattern.h"
#include "EpetraSparsityPattern.h"
#include "EpetraFactory.h"
#include "EpetraMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix()
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(uint M, uint N)
{
  // TODO: call Epetra_Init or something?
  // Create Epetra matrix
  resize(M, N);
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const EpetraMatrix& A)
{
  if (A.mat())
    this->A.reset(new Epetra_FECrsMatrix(*A.mat()));
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(boost::shared_ptr<Epetra_FECrsMatrix> A) : A(A)
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const Epetra_CrsGraph& graph) :
    A(new Epetra_FECrsMatrix(Copy, graph))
{
  // TODO: call Epetra_Init or something?
}
//-----------------------------------------------------------------------------
EpetraMatrix::~EpetraMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraMatrix::resize(uint M, uint N)
{
  error("EpetraMatrix::resize(uint, unit) not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  if (A && !A.unique())
    error("Cannot initialise EpetraMatrix. More than one object points to the underlying Epetra object.");

  const EpetraSparsityPattern& epetra_pattern = dynamic_cast<const EpetraSparsityPattern&>(sparsity_pattern);
  A.reset(new Epetra_FECrsMatrix(Copy, epetra_pattern.pattern()));
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraMatrix::copy() const
{
  return new EpetraMatrix(*this);
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraMatrix::size(uint dim) const
{
  assert(A);
  int M = A->NumGlobalRows();
  int N = A->NumGlobalCols();
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::get(double* block, uint m, const uint* rows,
		                   uint n, const uint* cols) const
{
  error("EpetraMatrix::get needs to be fixed.");

  assert(A);

  int num_entities = 0;
  int * indices;
  double * values;

  // For each row in rows
  for(uint i = 0; i < m; ++i)
  {
    // Extract the values and indices from row: rows[i]
    if (A->IndicesAreLocal())
    {
      int err = A->ExtractMyRowView(rows[i], num_entities, values, indices);
      if (err!= 0) {
        error("EpetraMatrix::get: Did not manage to perform Epetra_CrsMatrix::ExtractMyRowView.");
      }
    }
    else
    {
      int err = A->ExtractGlobalRowView(rows[i], num_entities, values, indices);
      if (err!= 0)
        error("EpetraMatrix::get: Did not manage to perform Epetra_CRSMatrix::ExtractGlobalRowView.");
    }

    // Step the indices to the start of cols
    int k = 0;
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
void EpetraMatrix::set(const double* block,
                       uint m, const uint* rows,
                       uint n, const uint* cols)
{
  assert(A);
  int err = A->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
                                   n, reinterpret_cast<const int*>(cols), block);
  if (err!= 0)
    error("EpetraMatrix::set: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::add(const double* block,
                       uint m, const uint* rows,
                       uint n, const uint* cols)
{
  assert(A);
  Timer t0("Matrix add");
  int err = A->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows),
                                   n, reinterpret_cast<const int*>(cols), block,
                                   Epetra_FECrsMatrix::ROW_MAJOR);

  if (err != 0)
    error("EpetraMatrix::add: Did not manage to perform Epetra_CrsMatrix::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern)
{
  const EpetraMatrix* AA = &A.down_cast<EpetraMatrix>();

  if (!AA->mat()->Filled())
    error("Epetramatrix is not in the correct state for addition.");

  int err = EpetraExt::MatrixMatrix::Add(*(AA->mat()), false, a, *(this->A), 1.0);
  if (err != 0)
    error("EpetraMatrDid::axpy: Did not manage to perform EpetraExt::MatrixMatrix::Add. If the matrix has been assembled, the nonzero patterns must match.");
}
//-----------------------------------------------------------------------------
double EpetraMatrix::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return A->NormOne();
  else if (norm_type == "linf")
    return A->NormInf();
  else if (norm_type == "frobenius")
    return A->NormFrobenius();
  else
  {
    error("Unknown norm type in EpetraMatrix.");
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero()
{
  assert(A);
  int err = A->PutScalar(0.0);
  if (err!= 0)
    error("EpetraMatrix::zero: Did not manage to perform Epetra_CrsMatrix::PutScalar.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::apply()
{
  assert(A);
  int err = A->GlobalAssemble();
  if (err!= 0)
    error("EpetraMatrix::apply: Did not manage to perform Epetra_CrsMatrix::GlobalAssemble.");

  // TODO
  //A->OptimizeStorage();
}
//-----------------------------------------------------------------------------
std::string EpetraMatrix::str(bool verbose) const
{
  assert(A);

  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for EpetraMatrix not implemented, calling Epetra Print directly.");
    A->Print(std::cout);
  }
  else
    s << "<EpetraMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void EpetraMatrix::ident(uint m, const uint* rows)
{
  assert(A);

  // FIXME: Is this the best way to do this?

  // FIXME: This can be made more efficient by eliminating creation of some
  //        obejcts inside the loop

  const Epetra_CrsGraph& graph = A->Graph();
  for (uint i = 0; i < m; ++i)
  {
    int row = rows[i];
    int num_nz = graph.NumGlobalIndices(row);
    std::vector<int> indices(num_nz);

    int out_num = 0;
    graph.ExtractGlobalRowCopy(row, num_nz, out_num, &indices[0]);

    //cout << "Testing graph " << row << "   " << out_num << endl;
    //for (int j = 0; j < num_nz1; ++j)
    //  cout << "  "  << j << "  " << indices[j] << endl;

    // Zero row
    std::vector<double> block(num_nz);
    int err = A->ReplaceGlobalValues(row, num_nz, &block[0], &indices[0]);
    if (err!= 0)
      error("EpetraMatrix::ident: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");

    // Place one on the diagonal
    double one = 1.0;
    A->ReplaceGlobalValues(row, 1, &one, &row);
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero(uint m, const uint* rows)
{
  // FIXME: This can be made more efficient by eliminating creation of some
  //        obejcts inside the loop

  assert(A);

  const Epetra_CrsGraph& graph = A->Graph();
  for (uint i = 0; i < m; ++i)
  {
    int row = rows[i];
    int num_nz = graph.NumGlobalIndices(row);
    std::vector<int> indices(num_nz);

    int out_num = 0;
    graph.ExtractGlobalRowCopy(row, num_nz, out_num, &indices[0]);

    std::vector<double> block(num_nz);
    int err = A->ReplaceGlobalValues(row, num_nz, &block[0], &indices[0]);
    if (err!= 0)
      error("EpetraMatrix::zero: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::mult(const GenericVector& x_, GenericVector& Ax_) const
{
  assert(A);

  const EpetraVector* x = dynamic_cast<const EpetraVector*>(x_.instance());
  if (!x)
    error("EpetraMatrix::mult: The vector x should be of type EpetraVector.");

  EpetraVector* Ax = dynamic_cast<EpetraVector*>(Ax_.instance());
  if (!Ax)
    error("EpetraMatrix::mult: The vector Ax should be of type EpetraVector.");

  if (size(1) != x->size())
    error("EpetraMatrix::mult: Matrix and vector dimensions don't match for matrix-vector product.");
  Ax->resize(size(0));

  int err = A->Multiply(false, *(x->vec()), *(Ax->vec()));
  if (err!= 0)
    error("EpetraMatrix::mult: Did not manage to perform Epetra_CRSMatrix::Multiply.");

}
//-----------------------------------------------------------------------------
void EpetraMatrix::transpmult(const GenericVector& x_, GenericVector& Ax_) const
{
  assert(A);

  const EpetraVector* x = dynamic_cast<const EpetraVector*>(x_.instance());
  if (!x)
    error("EpetraMatrix::transpmult: The vector x should be of type EpetraVector.");

  EpetraVector* Ax = dynamic_cast<EpetraVector*>(Ax_.instance());
  if (!Ax)
    error("EpetraMatrix::transpmult: The vector Ax should be of type EpetraVector.");

  if (size(0) != x->size())
    error("EpetraMatrix::transpmult: Matrix and vector dimensions don't match for (transposed) matrix-vector product.");
  Ax->resize(size(1));

  int err = A->Multiply(true, *(x->vec()), *(Ax->vec()));
  if (err!= 0)
    error("EpetraMatrix::transpmult: Did not manage to perform Epetra_CRSMatrix::Multiply.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::getrow(uint row, std::vector<uint>& columns,
                          std::vector<double>& values) const
{
  error("EpetraMatrix::getrow needs to be fixed.");

  assert(A);

  // Temporary variables
  int *indices;
  double* vals;
  int* num_entries = new int;

  // Extract data from Epetra matrix
  int err = A->ExtractMyRowView(row, *num_entries, vals, indices);
  if (err!= 0)
    error("EpetraMatrix::getrow: Did not manage to perform Epetra_CrsMatrix::ExtractMyRowView.");

  // Put data in columns and values
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
void EpetraMatrix::setrow(uint row, const std::vector<uint>& columns,
                          const std::vector<double>& values)
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraMatrix::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FECrsMatrix> EpetraMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator*= (double a)
{
  assert(A);
  int err = A->Scale(a);
  if (err!=0)
    error("EpetraMatrix::operator*=: Did not manage to perform Epetra_CrsMatrix::Scale.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator/= (double a)
{
  assert(A);
  int err = A->Scale(1.0/a);
  if (err!=0)
    error("EpetraMatrix::operator/=: Did not manage to perform Epetra_CrsMatrix::Scale.");
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
  assert(A);
  *(this->A) = *A.mat();
  return *this;
}
//-----------------------------------------------------------------------------

#endif
