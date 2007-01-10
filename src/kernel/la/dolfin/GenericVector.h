// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-25
// Last changed: 2006-12-12

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// This class defines a common interface for sparse and dense vectors.

  class GenericVector
  {
  public:
 
    /// Constructor
    GenericVector() {}

    /// Destructor
    virtual ~GenericVector() {}
    
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
