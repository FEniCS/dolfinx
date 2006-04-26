// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-25
// Last changed: 

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <dolfin/constants.h>
//#include <dolfin/Vector.h>

namespace dolfin
{
  /// This template provdies a uniform interface to both dense and sparse 
  /// matrices. It provides member functions that are required by functions 
  /// that operate with both dense and sparse matrices. 

  template < class T >
  class GenericVector 
  {
  public:
 
    /// Constructor
    GenericVector(){}

    /// Destructor
    ~GenericVector(){}

    /// Return the "leaf" object
    T& vector() 
      { return static_cast<T&>(*this); }

    /// Return the "leaf" object (constant version)
    const T& vector() const
      { return static_cast<const T&>(*this); }

    /// Initialize vector of length N
    void init(uint N)
      { vector().init(N); }

    /// Return size
    uint size() 
      { return vector().size(); }

    /// Set all entries to a single scalar value
    const GenericVector& operator= (real a)
      { return vector() = a; }

    /// Set all entries to zero
    void clear()
      { vector().clear(); }

    /// Add block of values
    void add(const real block[], const int pos[], int n)
      { vector().add(block, pos, n); }

    /// Add block of values
    void insert(const real block[], const int pos[], int n)
      { vector().insert(block, pos, n); }

    /// Apply changes to matrix (only needed for sparse matrices)
    void apply()
      { vector().apply(); }


  private:

    
  };  
}
#endif
