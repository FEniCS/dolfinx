// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

#ifndef __ASSEMBLY_MATRIX_H
#define __ASSEMBLY_MATRIX_H

#include <vector>
#include <map>
#include <sstream>
#include <iomanip>

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  /// Simple implementation of a GenericTensor for experimenting
  /// with new assembly. Not sure this will be used later but it
  /// might be useful.

  class AssemblyMatrix : public GenericTensor
  {
  public:

    /// Constructor
    AssemblyMatrix() : GenericTensor(), dims(0)
    {
      dims = new uint[2];
    }

    /// Destructor
    ~AssemblyMatrix()
    {
      delete [] dims;
    }

    ///--- Functions overloaded from GenericTensor ---

    /// Initialize zero tensor of given rank and dimensions
    void init(uint rank, uint* dims)
    {
      // Check that the rank is 2
      if ( rank != 2 )
        dolfin_error1("Illegal tensor rank (%d) for matrix. Rank must be 2.", rank);

      // Initialize matrix
      init(dims[0], dims[1]);

      // Save dimensions
      this->dims[0] = dims[0];
      this->dims[1] = dims[1];
    }
    
    /// Return size of given dimension
    virtual uint size(const uint dim) const
    {
      return dims[dim];
    }

    /// Add block of values
    void add(real* block, uint* num_rows, uint** rows)
    {
      add(block, num_rows[0], rows[0], num_rows[1], rows[1]);
    }

    ///--- Specialized matrix functions ---

    /// Initialize M x N matrix
    void init(uint M, uint N)
    {
      // Set number of rows
      A.resize(M);
      
      // Initialize with zeros
      for (uint i = 0; i < M; i++)
        for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
          it->second = 0.0;
    }

    /// Add entries to matrix
    void add(real* block, uint m, uint* rows, uint n, uint* cols)
    {
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

    /// Finalise assembly
    void apply() {}

    /// Display matrix
    void disp(uint precision = 2)
    {
      for (uint i = 0; i < dims[0]; i++)
      {
        std::stringstream line;
        line << std::setiosflags(std::ios::scientific);
        line << std::setprecision(precision);
    
        line << "|";
        for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
          line << " (" << i << ", " << it->first << ", " << it->second << ")";
        line << " |";
        
        dolfin::cout << line.str().c_str() << dolfin::endl;
      }
    }

  private:

    // The matrix representation
    std::vector<std::map<uint, real> > A;

    // The size of the matrix
    uint* dims;

  };

}

#endif
