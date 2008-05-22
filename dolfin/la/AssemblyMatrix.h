// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-05-19

#ifndef __ASSEMBLY_MATRIX_H
#define __ASSEMBLY_MATRIX_H

//#include <vector>
//#include <map>
//#include <sstream>
//#include <iomanip>

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Variable.h>
#include "GenericMatrix.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"
#include "AssemblyFactory.h"

namespace dolfin
{

  /// Simple implementation of a GenericMatrix for experimenting
  /// with new assembly. Not sure this will be used later but it
  /// might be useful.

  class uBlasVector;
  class AssemblyFactory;

  class AssemblyMatrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    AssemblyMatrix() : dims(0)
    { dims = new uint[2]; }

    /// Create M x N matrix
    AssemblyMatrix(uint M, uint N) : dims(0)
    { dims = new uint[2]; dims[0] = M; dims[1] = N;}
  
    /// Copy constructor
    explicit AssemblyMatrix(const AssemblyMatrix& A)
    { 
      //dims = new uint[2]; 
      //dims[0] = A.size(0);
      //dims[1] = A.size(1);
      error("Not implemented.");
    }

    /// Destructor
    virtual ~AssemblyMatrix()
    { delete [] dims; }

    ///--- Implementation of the GenericTenson interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern)
      //    { init(sparsity_pattern.size(0), sparsity_pattern.size(1); }
    {
      uint M = sparsity_pattern.size(0);
      /// The comment directly below also makes the following 
      /// two lines unnecessary:
      /*
      uint* nzrow = new uint[M];
      sparsity_pattern.numNonZeroPerRow(nzrow);
      */
      A.resize(M);
      /// Map does not support reserve. Also, allocating space for a std::map
      /// is not needed as it does not need to copy an element because others 
      /// are inserted.
      /*
      for (uint i = 0; i < M; ++i)
        A[i].reserve(nzrow[i]);
      delete [] nzrow;
      */
    }

    /// Return copy of tensor
    virtual AssemblyMatrix* copy() const;
    //    { 
    //      AssemblyMatrix* mcopy = AssemblyFactory::instance().createMatrix();
    //      error("Not implemented.");
    //      return mcopy;
    //    }

    /// Return size of given dimension
    virtual uint size(uint dim) const
    { return dims[dim]; }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      for (uint i = 0; i < A.size(); i++)
        for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
          it->second = 0.0;
    } 
    
    /// Finalize assembly of tensor
    virtual void apply()
    { error("Not implemented."); }

    /// Display tensor
    virtual void disp(uint precision = 2) const
    {
      for (uint i = 0; i < dims[0]; i++)
      {
        std::stringstream line;
        line << std::setiosflags(std::ios::scientific);
        line << std::setprecision(precision);
        
        line << "|";
        for (std::map<uint, real>::const_iterator it = A[i].begin(); it != A[i].end(); it++)
          line << " (" << i << ", " << it->first << ", " << it->second << ")";
        line << " |";
        
        dolfin::cout << line.str().c_str() << dolfin::endl;
      }
    }

    //--- Implementation of the GenericMatrix interface ---
    
    /// Initialize M x N matrix
    virtual void init(uint M, uint N)
    {
      // Set number of rows
      A.resize(M);
      
      /*
      // Initialize with zeros
      for (uint i = 0; i < M; i++)
        for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
          it->second = 0.0;
      */
    }

    /// Get block of values
    virtual void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set block of values
    virtual void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Add block of values
    virtual void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Get non-zero values of given row
    virtual void getrow(uint row, Array<uint>& columns, Array<real>& values) const
    { error("Not implemented."); }

    /// Set values for given row
    virtual void setrow(uint row, const Array<uint>& columns, const Array<real>& values)
    { error("Not implemented."); }

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows)
    { error("Not implemented."); }

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows)
    { error("Not implemented."); }

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const
    { error("Not implemented."); }

    /// Multiply matrix by given number
    virtual const GenericMatrix& operator*= (real a);
    //{ error("Not implemented."); }

    /// Divide matrix by given number
    virtual const GenericMatrix& operator/= (real a);
    //{ error("Not implemented."); }

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);
    //{ error("Not implemented."); } 

    ///--- Specialized matrix functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, const uint* dims, bool reset = true)
    {
      // Check that the rank is 2
      if ( rank != 2 )
        error("Illegal tensor rank (%d) for matrix. Rank must be 2.", rank);

      // Initialize matrix
      init(dims[0], dims[1]);

      // Save dimensions
      this->dims[0] = dims[0];
      this->dims[1] = dims[1];
    }

    /// Add entries to matrix
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    {
      message("Test");
      uint pos = 0;
      for (uint i = 0; i < m; i++)
      {
        std::map<uint, real>& row = A[rows[i]];
        for (uint j = 0; j < n; j++)
        {
          const uint col = cols[j];
          const std::map<uint, real>::iterator it = row.find(col);
          if ( it == row.end() )
            row.insert(it, std::map<uint, real>::value_type(col, block[pos++]));
          else
            it->second += block[pos++];
        }
      }
    }
    
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const
    { error("Not implemented"); }

    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { error("Not implemented"); }

  private:

    // The matrix representation
    std::vector<std::map<uint, real> > A;

    // The size of the matrix
    uint* dims;

  };

}

#endif
