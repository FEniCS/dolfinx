// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25
// Last changed: 2009-08-11
//
// Modified by Anders Logg, 2008.

#include <iostream>
#include <stdexcept>

#include "dolfin/common/utils.h"
#include "DefaultFactory.h"
#include "GenericVector.h"

#include "BlockVector.h"
#include "BlockMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockMatrix::BlockMatrix(uint n_, uint m_, bool owner_)
    : owner(owner_), n(n_), m(m_)
{
  matrices = new Matrix*[n*m];
  if (owner)
  {
    for (uint i = 0; i < n; i++)
      for (uint j = 0; j<m; j++)
        matrices[i*n + j] = new Matrix();
  }
}
//-----------------------------------------------------------------------------
BlockMatrix::~BlockMatrix()
{
  if (owner)
  {
    for (uint i = 0; i < n; i++)
      for (uint j = 0; j<m; j++)
        delete matrices[i*n + j];
  }
  delete [] matrices;
}
//-----------------------------------------------------------------------------
void BlockMatrix::set(uint i, uint j, Matrix& m)
{
//  matrices[i*n+j] = m.copy(); //FIXME. not obvious that copy is the right thing
  matrices[i*n+j] = &m;         //FIXME. not obvious that copy is the right thing
}
//-----------------------------------------------------------------------------
const Matrix& BlockMatrix::get(uint i, uint j) const
{
  return *(matrices[i*n+j]);
}
//-----------------------------------------------------------------------------
Matrix& BlockMatrix::get(uint i, uint j)
{
  return *(matrices[i*n+j]);
}
//-----------------------------------------------------------------------------
dolfin::uint BlockMatrix::size(uint dim) const
{
  if (dim == 0) return n;
  if (dim == 1) return m;
  error("BlockMatrix has rank 2!"); return 0;
}
//-----------------------------------------------------------------------------
void BlockMatrix::zero()
{
  for(uint i = 0; i < n; i++)
    for(uint j = 0; j < n; j++)
      this->get(i,j).zero();
}
//-----------------------------------------------------------------------------
void BlockMatrix::apply(std::string mode)
{
  for(uint i = 0; i < n; i++)
    for(uint j = 0; j < n; j++)
      this->get(i,j).apply(mode);
}
//-----------------------------------------------------------------------------
std::string BlockMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (uint i = 0; i < n; i++)
    {
      for (uint j = 0; i < m; j++)
      {
        s << "  BlockMatrix (" << i << ", " << j << ")" << std::endl << std::endl;
        s << indent(indent(get(i, j).str(true))) << std::endl;
      }
    }
  }
  else
    s << "<BlockMatrix containing " << n << " x " << m << " blocks>";

  return s.str();
}
//-----------------------------------------------------------------------------
void BlockMatrix::mult(const BlockVector& x, BlockVector& y,
                       bool transposed) const
{
  if (transposed)
    error("BlockMatrix::mult: transposed not implemented");
  DefaultFactory factory;
  GenericVector* vec = factory.create_vector();
  for(uint i = 0; i < n; i++)
  {
    y.get(i).resize(this->get(i, 0).size(0));
    vec->resize(y.get(i).size());
    // FIXME: Do we need to zero the vector?
    y.get(i).zero();
    vec->zero();
    for(uint j = 0; j < n; j++)
    {
      this->get(i, j).mult(x.get(j), *vec);
      y.get(i) += *vec;
    }
  }
  delete vec;
}
//-----------------------------------------------------------------------------
SubMatrix BlockMatrix::operator()(uint i, uint j)
{
  SubMatrix sm(i,j,*this);
  return sm;
}
//-----------------------------------------------------------------------------
//FIXME there are numerous functions that should be added

//-----------------------------------------------------------------------------
// SubMatrix
//-----------------------------------------------------------------------------
SubMatrix::SubMatrix(uint col, uint row, BlockMatrix& bm)
  : row(row), col(col), bm(bm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubMatrix::~SubMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const SubMatrix& SubMatrix::operator=(Matrix& m)
{
  bm.set(row, col, m);
  return *this;
}
/*
//-----------------------------------------------------------------------------
Matrix& SubMatrix::operator()
{
  return bm.get(row, col);
}
*/
//-----------------------------------------------------------------------------
