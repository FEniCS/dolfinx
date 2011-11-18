// Copyright (C) 2006-2009 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2006-2010.
//
// First added:  2006-06-23
// Last changed: 2010-08-31

#include <dolfin/common/constants.h>
#include "uBLASVector.h"
#include "uBLASSparseMatrix.h"
#include "uBLASILUPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASILUPreconditioner::uBLASILUPreconditioner(const Parameters& krylov_parameters)
  : krylov_parameters(krylov_parameters)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASILUPreconditioner::~uBLASILUPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASILUPreconditioner::init(const uBLASMatrix<ublas_sparse_matrix>& P)
{
  ublas_sparse_matrix& _M = M.mat();

  const uint size = P.size(0);
  _M.resize(size, size, false);
  _M.assign(P.mat());

  // Add term to diagonal to avoid negative pivots
  const double zero_shift = krylov_parameters("preconditioner")["shift_nonzero"];
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

  // This approach using uBLAS iterators is simple and quite fast.
  // Is it possible to remove the M(.,.) calls? This would speed it up a lot.

  /*
  // Sparse matrix iterators
  typedef uBLASMatrix<ublas_sparse_matrix>::iterator1 it1;
  typedef uBLASMatrix<ublas_sparse_matrix>::iterator2 it2;
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

  std::vector<uint> iw(size);
  for(uint i=0; i< size; ++i)
    iw[i] = 0;

  uint j0=0, j1=0, j=0, jrow=0, jw=0;
  double t1;

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

    if (jrow != k || fabs( _M.value_data() [j] ) < DOLFIN_EPS)
    {
      dolfin_error("uBLASILUPreconditioner.cpp",
                   "initialize uBLAS ILU preconditioner",
                   "Zero pivot detected in row %u", k);
    }

    for(uint i=j0; i <= j1; ++i)
      iw[ _M.index2_data () [i] ] = 0;
  } // k
}
//-----------------------------------------------------------------------------
void uBLASILUPreconditioner::solve(uBLASVector& x, const uBLASVector& b) const
{
  // Get uderlying uBLAS matrices and vectors
  ublas_vector& _x = x.vec();
  const ublas_vector& _b = b.vec();
  const ublas_sparse_matrix & _M = M.mat();

  dolfin_assert(_M.size1() > 0 && _M.size2() > 0);
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

