// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25


#ifndef __BLOCKMATRIX_H
#define __BLOCKMATRIX_H

#include <map>
#include "GenericMatrix.h"

namespace dolfin
{
  class BlockMatrix 
  {
    private:
      uint n, m; 
      //    std::map<std::pair<int,int>, GenericMatrix*> matrices; 
      GenericMatrix* matrices; 

    public:

      BlockMatrix(uint n_=0, uint m_=0); 

      /// Return GenericMatrix reference number (i,j) 
      /* FIXME these functions should probably be inline
       * and all the LA function should rely on these */
      const GenericMatrix& mat(uint i, uint j) const; 
      GenericMatrix& mat(uint i, uint j); 

      /// Return size of given dimension
      uint size(uint dim) const;

      /// Set all entries to zero and keep any sparse structure
      void zero();

      /// Finalize assembly of tensor
      void apply();

      /// Display tensor
      void disp(uint precision=2) const;

      /// Matrix-vector product, y = Ax
      void mult(const BlockVector& x, BlockVector& y, bool transposed=false) const;
  }; 
}

#endif 


