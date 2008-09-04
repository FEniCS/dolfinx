// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25

#include <stdexcept>
#include "BlockVector.h"
#include "BlockMatrix.h"
#include "DefaultFactory.h"
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockMatrix::BlockMatrix(uint n_, uint m_, bool owner_): owner(owner_), n(n_), m(m_)
{
  matrices = new Matrix*[n*m]; 
  if (owner) 
  {
    for (uint i=0; i<n; i++) 
    {
      for (uint j=0; j<m; j++) 
      {
        matrices[i*n + j] = new Matrix(); 
      }
    }
  }
} 
//-----------------------------------------------------------------------------
BlockMatrix::~BlockMatrix() 
{
  if (owner) 
  {
    for (uint i=0; i<n; i++) 
    {
      for (uint j=0; j<m; j++) 
      {
        delete matrices[i*n + j];
      }
    }
  }
  delete [] matrices;  
}
//-----------------------------------------------------------------------------
/*
const Matrix& BlockMatrix::mat(uint i, uint j) const 
{
  if (i >= n || j >= m) {  
    error("The index is out of range!");
  }
  return *(matrices[i*n+j]);
}
*/
//-----------------------------------------------------------------------------
void BlockMatrix::set(uint i, uint j, Matrix& m)
{
//  matrices[i*n+j] = m.copy(); //FIXME. not obvious that copy is the right thing
  matrices[i*n+j] = &m;         //FIXME. not obvious that copy is the right thing
}
//-----------------------------------------------------------------------------
const Matrix& BlockMatrix::getc(uint i, uint j) const
{
  return *(matrices[i*n+j]); 
}
//-----------------------------------------------------------------------------
Matrix& BlockMatrix::get(uint i, uint j)
{
  return *(matrices[i*n+j]); 
}

//-----------------------------------------------------------------------------
/*
Matrix& BlockMatrix::mat(uint i, uint j) 
{
  // FIXME this function does not work because operator= is not implemented in the
  // various Matrix classes, should it be ? 
  if (i >= n || j >= m) {  
    error("The index is out of range!");
  }
  return *(matrices[i*n+j]);
}
*/
//-----------------------------------------------------------------------------
dolfin::uint BlockMatrix::size(uint dim) const {
  if (dim==0) return n; 
  if (dim==1) return m; 
  error("BlockMatrix has rank 2!"); return 0; 
}
//-----------------------------------------------------------------------------
void BlockMatrix::zero()  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      this->get(i,j).zero(); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::apply()  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      this->get(i,j).apply(); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::disp(uint precision) const  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      std::cout <<"BlockMatrix("<<i<<","<<j<<"):"<<std::endl;  
      this->getc(i,j).disp(precision); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::mult(const BlockVector& x, BlockVector& y, bool transposed) const  
{
  if (transposed) error("BlockMatrix::mult: transposed not implemented");  
  DefaultFactory factory; 
  GenericVector* vec; 
  vec = factory.createVector();
  for(uint i=0; i<n; i++) 
  {
    y.get(i).init(this->getc(i,0).size(0));
    vec->init(y.getc(i).size()); 
    for(uint j=0; j<n; j++) 
    {
      this->getc(i,j).mult(x.getc(j), *vec);   
      y.get(i) += *vec; 
    }
  }
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
SubMatrix::SubMatrix(uint col_, uint row_, BlockMatrix& bm_) 
  : row(row_), col(col_), bm(bm_)
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






