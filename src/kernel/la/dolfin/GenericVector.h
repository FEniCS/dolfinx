// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006-2007.
//
// First added:  2006-04-25
// Last changed: 2007-03-15

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <dolfin/constants.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  /// This class defines a common interface for sparse and dense vectors.

  class GenericVector : public GenericTensor
  {
  public:
 
    /// Constructor
    GenericVector() {}

    /// Destructor
    virtual ~GenericVector() {}

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, uint* dims)
    { init(dims[0]); }

    /// Return size of given dimension
    virtual uint size(const uint dim) const
    { return size(); }

    /// Add block of values
    virtual void add(real* block, uint* num_rows, uint** rows)
    { 
      // FIXME: Change order of arguments to add() function
      // FIXME: Change from int to uint
      add(block,
          reinterpret_cast<const int*>(rows[0]),
          static_cast<int>(num_rows[0]));
    }

    ///--- Vector functions ---
    
    /// Initialize vector of size N
    virtual void init(const uint N) = 0;

    /// Return size
    virtual uint size() const = 0;

    /// Access element value
    virtual real get(const uint i) const = 0;

    /// Set element value
    virtual void set(const uint i, const real value) = 0;

    /// Set block of values
    virtual void set(const real block[], const int pos[], const int n) = 0;

    /// Add block of values
    virtual void add(const real block[], const int pos[], const int n) = 0;

    /// Apply changes to vector (only needed for PETSc vectors)
    virtual void apply() = 0;

    /// Set all entries to zero
    virtual void zero() = 0;
    
    /// Compute sum of vector
    virtual real sum() const = 0;

  };  

}
#endif
