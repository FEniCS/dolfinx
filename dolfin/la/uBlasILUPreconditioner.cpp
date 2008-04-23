// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-06-23
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include "uBlasVector.h"
#include "uBlasSparseMatrix.h"
#include "uBlasILUPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasILUPreconditioner::uBlasILUPreconditioner() : uBlasPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasILUPreconditioner::uBlasILUPreconditioner(const uBlasMatrix<ublas_sparse_matrix>& A)
  : uBlasPreconditioner()
{
  // Initialize preconditioner
  init(A);
}
//-----------------------------------------------------------------------------
uBlasILUPreconditioner::~uBlasILUPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasILUPreconditioner::solve(uBlasVector& x, const uBlasVector& b) const
{
  // Get uderlying uBLAS matrices and vectors
  ublas_vector& _x = x.vec(); 
  const ublas_vector& _b = b.vec(); 
  const ublas_sparse_matrix & _M = M.mat();

  dolfin_assert( _x.size() == _M.size1() );
  dolfin_assert( _x.size() == _b.size());

  // Solve in-place
  _x.assign(_b);

  // Perform substutions for compressed row storage. This is the fastest.
  const uint size = _M.size1();
  for(uint i =0; i < size; ++i)
  {
    uint k;
    for(k = _M.index1_data () [i]; k < diagonal[i]; ++k)
      _x(i) -= ( _M.value_data () [k] )*x[ _M.index2_data () [k] ];
  } 
  for(int i =size-1; i >= 0; --i)
  {
    uint k;
    for(k = _M.index1_data () [i+1]-1; k > diagonal[i]; --k)
      _x(i) -= ( _M.value_data () [k] )*x[ _M.index2_data () [k] ];
    _x(i) /= ( _M.value_data () [k] );  
  }
}
//-----------------------------------------------------------------------------
void uBlasILUPreconditioner::init(const uBlasMatrix<ublas_sparse_matrix>& A)
{
  ublas_sparse_matrix & _M = M.mat();

  const uint size = A.size(0); 
  _M.resize(size, size, false);
  _M.assign(A.mat()); 

  // Add term to diagonal to avoid negative pivots
  const real zero_shift = get("Krylov shift nonzero");
  if(zero_shift > 0.0)
    _M.plus_assign( zero_shift*ublas::identity_matrix<double>(size) );  
  
  /*
  // Straightforward and very slow implementation. This is used for verification
  tic();
  M.assign(A);  
  for(uint i=1; i < size; ++i)
  {
    for(uint k=0; k < i; ++k)
    { 
     if( fabs( M(i,k) ) > 0.0 )
      {
        M(i,k) = M(i,k)/M(k,k);
        for(uint j=k+1; j < size; ++j)
        {
          if( fabs( M(i,j) ) > DOLFIN_EPS )
              M(i,j) = M(i,j) - M(i,k)*M(k,j);
        }
      }    
    }  
  }
  */
  
  // This approach using uBlas iterators is simple and quite fast.
  // Is it possible to remove the M(.,.) calls? This would speed it up a lot.

  /*
  // Sparse matrix iterators
  typedef uBlasMatrix<ublas_sparse_matrix>::iterator1 it1;
  typedef uBlasMatrix<ublas_sparse_matrix>::iterator2 it2;
  it2 ij;
  for (it1 i1 = M.begin1(); i1 != M.end1(); ++i1)  // iterate over rows  i=0 -> size
  {
    for (it2 ik = i1.begin(); ik.index2() <  ik.index1(); ++ik) // move along row k=0 -> i
    {
      *ik /= M(ik.index2(),ik.index2());    // M(i,k) = M(i,k)/M(k,k)
      ij=ik;
      ++ij;
      for( ; ij != i1.end(); ++ij)                  // over j=k+1 -> size
        *ij -=  (*ik)*M(ik.index2(), ij.index2());  // M(i,j) = M(i,j) - M(i,k)*M(k, j)
    }
  }
  */

  // The below algorithm is based on that in the book
  // Y. Saad, "Iterative Methods for Sparse Linear Systems", p.276-278.
  // It is specific to compressed row storage

  // Initialise some data
  diagonal.resize(size);
  diagonal[0] = 0;

  Array<uint> iw(size);
  for(uint i=0; i< size; ++i)
    iw[i] = 0;

  uint j0=0, j1=0, j=0, jrow=0, jw=0;
  real t1;   

  for (uint k = 0; k < size ; ++k)        // i (rows)
  {
    j0 = _M.index1_data () [k];    // ia 
    j1 = _M.index1_data () [k+1]-1;

    // Initialise working array iw
    for (uint i = j0;  i <= j1; ++i)       
      iw[ _M.index2_data () [i] ] = i;  // ja     

    // Move along row looking for diagonal
    j=j0;
    while(j <= j1)
    {
      jrow = _M.index2_data () [j];  // ja  

      if( jrow >= k ) // passed or found diagonal, therefore break
        break;

      t1 = (_M.value_data() [j])/(_M.value_data() [ diagonal[jrow] ]);  //M(k,j) = M(k,j)/M(j,j) 
      _M.value_data() [j] = t1;
      for(uint jj = diagonal[jrow]+1; jj <= _M.index1_data () [jrow+1]-1; ++jj)
      {
        jw = iw[ _M.index2_data () [jj] ];
        if(jw != 0)
          _M.value_data() [jw] = _M.value_data() [jw] -t1*(_M.value_data() [jj]);
      } // jj
      ++j;
    }
    diagonal[k] = j;

    if( jrow != k || fabs( _M.value_data() [j] ) < DOLFIN_EPS )
      error("Zero pivot detected in uBlas ILU preconditioner in row %u.", k);

    for(uint i=j0; i <= j1; ++i)
      iw[ _M.index2_data () [i] ] = 0;        
  } // k
}
//-----------------------------------------------------------------------------
