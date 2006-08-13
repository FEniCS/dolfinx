// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-25
// Last changed: 2006-05-30

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
    virtual void init(uint N) = 0;

    /// Return size
    virtual uint size() const = 0;

    /// Access value of given entry
    virtual real operator() (uint i) const = 0;

    /// Set block of values
    virtual void set(const real block[], const int pos[], int n) = 0;

    /// Add block of values
    virtual void add(const real block[], const int pos[], int n) = 0;

    /// Get block of values
    virtual void get(real block[], const int pos[], int n) const = 0;

    /// Apply changes to vector (only needed for PETSc vectors)
    virtual void apply() = 0;

    /// Set all entries to zero
    virtual void zero() = 0;

    /// Element access
    virtual real get(int i) = 0;
    virtual void set(int i, real value) = 0;
    
  };  

}
#endif
