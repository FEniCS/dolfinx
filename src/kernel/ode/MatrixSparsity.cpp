// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>
#include <dolfin/MatrixSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MatrixSparsity::MatrixSparsity(int N, const Matrix& A_) : 
  GenericSparsity(N), A(A_)
{
  // Check that the matrix is full
  if ( A.type() != Matrix::sparse )
    dolfin_error("You need to use a sparse matrix to specify sparsity.");
}
//-----------------------------------------------------------------------------
MatrixSparsity::~MatrixSparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericSparsity::Type MatrixSparsity::type() const
{
  return matrix;
}
//-----------------------------------------------------------------------------
MatrixSparsity::Iterator* MatrixSparsity::createIterator(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
MatrixSparsity::Iterator::Iterator(int i, const MatrixSparsity& sparsity) 
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;
}
//-----------------------------------------------------------------------------
MatrixSparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MatrixSparsity::Iterator& MatrixSparsity::Iterator::operator++()
{
  pos++;
  
  return *this;
}
//-----------------------------------------------------------------------------
int MatrixSparsity::Iterator::operator*() const
{
  // FIXME: This will be better when the iterator in Matrix is fixed
  if ( s.A.endrow(i, pos) )
    dolfin_error("Reached end of row.");

  int j;
  s.A(i, j, pos);
  
  return j;
}
//-----------------------------------------------------------------------------
bool MatrixSparsity::Iterator::end() const
{
  return s.A.endrow(i, pos);
}
//-----------------------------------------------------------------------------
