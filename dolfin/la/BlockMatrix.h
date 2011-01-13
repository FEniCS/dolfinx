// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-08-25
// Last changed: 2011-01-13

#ifndef __BLOCKMATRIX_H
#define __BLOCKMATRIX_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{
  // Forward declarations
  class GenericMatrix;
  class SubMatrix;

  class BlockMatrix
  {
  public:

    // Constructor
    BlockMatrix(uint m = 0, uint n = 0);

    // Destructor
    ~BlockMatrix();

    /// Set block
    void set_block(uint i, uint j, boost::shared_ptr<GenericMatrix> m);

    /// Get block (const version)
    const GenericMatrix& get_block(uint i, uint j) const;

    /// Get block
    GenericMatrix& get_block(uint i, uint j);

    /// Return SubMatrix reference number (i,j)
    SubMatrix operator() (uint i, uint j);

    /// Return size of given dimension
    uint size(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Matrix-vector product, y = Ax
    void mult(const BlockVector& x, BlockVector& y, bool transposed=false) const;

  private:

    std::vector<std::vector<boost::shared_ptr<GenericMatrix> > > matrices;

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
    const SubMatrix& operator= (boost::shared_ptr<GenericMatrix> m);

  private:

    uint row, col;
    BlockMatrix& bm;

  };

}

#endif
