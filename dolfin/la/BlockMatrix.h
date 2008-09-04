// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25


#ifndef __BLOCKMATRIX_H
#define __BLOCKMATRIX_H

#include <map>
#include "Matrix.h"

namespace dolfin
{
  // Forward declaration
  class SubMatrix; 


  class BlockMatrix 
  {
    private:
      bool owner; 
      uint n, m; 
      //    std::map<std::pair<int,int>, Matrix*> matrices; 
      Matrix** matrices; 

    public:

      // Constructor
      BlockMatrix(uint n=0, uint m=0, bool owner=true); 

      // Destructor
      ~BlockMatrix(); 

      /// Return SubMatrix reference number (i,j) 
      SubMatrix operator() (uint i, uint j); 

      void set(uint i, uint j, Matrix& m); 
      const Matrix& getc(uint i, uint j) const; 
            Matrix& get(uint i, uint j); 

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


  // SubMatrix 
  // Rip off of the design in Table and TableEntry for giving nice operators
  // A(0,0) = A00 also in the case with external storage. 

  class SubMatrix
  {
  public:
    SubMatrix(uint row, uint col, BlockMatrix& bm); 
    ~SubMatrix();

    /// Assign Matrix to SubMatrix 
    const SubMatrix& operator= (Matrix& m); 
//          Matrix& operator() const; 

  private:
    uint row, col; 
    BlockMatrix& bm; 
  }; 

}

#endif 


