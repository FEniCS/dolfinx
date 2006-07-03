// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-23
// Last changed: 

#include <dolfin/timing.h>
#include <dolfin/uBlasPreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasPreconditioner::uBlasPreconditioner() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasPreconditioner::uBlasPreconditioner(const uBlasSparseMatrix& A)
{
  // Do nothingc
}
//-----------------------------------------------------------------------------
uBlasPreconditioner::~uBlasPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasPreconditioner::init(const uBlasSparseMatrix& A)
{
  // Create M and other data
  create(A);
}
//-----------------------------------------------------------------------------
void uBlasPreconditioner::solve(ublas_vector& x) const
{
  // Do nothing if not initialized
  if ( M.size(0) == 0 )
    return;

  dolfin_assert( x.size() == M.size2() );

  // Let uBlas solve the systems. This is however very slow as uBlas
  // is not optimised for sparse storage, especially when using using
  // row-major matrices.
  //ublas::lu_substitute(M, x);

  // Perform substutions using uBlas sparse matrix iterators. This is 
  // relatively fast, but not that great.

/*
  uBlasSparseMatrix::const_iterator1 row;
  uBlasSparseMatrix::const_iterator1 row_end = M.end1();
  for (row = M.begin1(); row != row_end; ++ row) 
  {
    uint n = row.index1();
    uBlasSparseMatrix::const_iterator2 col = row.begin();
    uBlasSparseMatrix::const_iterator2 col_end = row.end();
    while ((col != col_end) && (col.index2() < n) ) 
    {
      x (n) -= *col * x (col.index2());
      ++ col;
    }
  }
  uBlasSparseMatrix::const_reverse_iterator1 row_r;
  uBlasSparseMatrix::const_reverse_iterator1 row_end_r = M.rend1();
  for (row_r = M.rbegin1(); row_r != row_end_r; ++ row_r) 
  {
    uint n = row_r.index1();
    uBlasSparseMatrix::const_reverse_iterator2 col = row_r.rbegin();
    uBlasSparseMatrix::const_reverse_iterator2 col_end = row_r.rend();
    while ((col != col_end) && (col.index2() > n) ) 
    {
      x (n) -= *col * x (col.index2());
      ++ col;
    }
    x (n) /= (*col);
  }
*/

  // Perform substutions for compressed row storage. This is the fastest.
  const uint size = M.size1();
  for(uint i =0; i < size; ++i)
  {
    uint k;
    for(k = M.index1_data () [i]; k < diagonal[i]; ++k)
      x(i) -= ( M.value_data () [k] )*x( M.index2_data () [k] );
  } 
  for(int i =size-1; i >= 0; --i)
  {
    uint k;
    for(k = M.index1_data () [i+1]-1; k > diagonal[i]; --k)
      x(i) -= ( M.value_data () [k] )*x( M.index2_data () [k] );
    x(i) /= ( M.value_data () [k] );  
  } 

}
//-----------------------------------------------------------------------------
void uBlasPreconditioner::solve(const ublas_vector& x, ublas_vector& y) const
{
  // Do nothing if not initialized
  if ( M.size(0) == 0 )
    return;

  // For efficiency, it is assumed here that y has been allocated 
  dolfin_assert( x.size() == y.size());
  y.assign(x);
  solve(y);
}
//-----------------------------------------------------------------------------
void uBlasPreconditioner::create(const uBlasSparseMatrix& A)
{
  const uint size = A.size(0); 
  M.resize(size, size, false);
  M.assign(A); 

  // Add term to diagonal to avoid negative pivots
  const real zero_shift = get("Krylov shift nonzero");
  if(zero_shift > 0.0)
    M.plus_assign( zero_shift*ublas::identity_matrix<double>(size) );  

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
  typedef uBlasSparseMatrix::iterator1 it1;
  typedef uBlasSparseMatrix::iterator2 it2;
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
    j0 = M.index1_data () [k];    // ia 
    j1 = M.index1_data () [k+1]-1;

    // Initialise working array iw
    for (uint i = j0;  i <= j1; ++i)       
      iw[ M.index2_data () [i] ] = i;  // ja     

    // Move along row looking for diagonal
    j=j0;
    while(j <= j1)
    {
      jrow = M.index2_data () [j];  // ja  

      if( jrow >= k ) // passed or found diagonal, therefore break
        break;

//      t1 = (M.value_data() [j])*(M.value_data() [ diagonal[jrow] ]);
      t1 = (M.value_data() [j])/(M.value_data() [ diagonal[jrow] ]);  //M(k,j) = M(k,j)/M(j,j) 
      M.value_data() [j] = t1;
      for(uint jj = diagonal[jrow]+1; jj <= M.index1_data () [jrow+1]-1; ++jj)
      {
        jw = iw[ M.index2_data () [jj] ];
        if(jw != 0)
          M.value_data() [jw] = M.value_data() [jw] -t1*(M.value_data() [jj]);
      } // jj
      ++j;
    }
    diagonal[k] = j;

    if( jrow != k || fabs( M.value_data() [j] ) < DOLFIN_EPS )
      dolfin_error1("Zero pivot detected in uBlas ILU preconditioner in row %u.", k);

//    M.value_data() [j] = 1.0/( M.value_data() [j] );
    for(uint i=j0; i <= j1; ++i)
      iw[ M.index2_data () [i] ] = 0;        
  } // k

  // Restore diagonal
//  for (uint k = 0; k < size ; ++k)
//    M.value_data() [ uptr[k] ] = 1.0/M.value_data() [ uptr[k] ];

}
//-----------------------------------------------------------------------------
