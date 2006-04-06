// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-06
// Last changed: 

#ifndef __NEW_MATRIX_H
#define __NEW_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// This template provdies a uniform interface to both dense and sparse 
  /// matrices.

  template < class T >
  class NewMatrix 
  {
  public:
 
    /// Constructor
    NewMatrix();

    /// Constructor
    NewMatrix(uint i, uint j);

    /// Destructor
    ~NewMatrix();

    /// Return address of an entry (inline is important here for speed)
    inline real& operator() (uint i, uint j);

  private:
    
    /// Pointer to matrix
    T* matrix;

  };  

  //---------------------------------------------------------------------------
  // Implementation of NewMatrix
  //---------------------------------------------------------------------------
  template <class T> NewMatrix<T>::NewMatrix()
  {
    matrix = new T;  
  }
  //---------------------------------------------------------------------------
  template <class T> NewMatrix<T>::NewMatrix(uint i, uint j)
  {
    matrix = new T(i,j);  
  }
  //---------------------------------------------------------------------------
  template <class T> NewMatrix<T>::~NewMatrix()
  {
    delete matrix;  
  }
  //---------------------------------------------------------------------------
  template <class T> real& NewMatrix<T>::operator() (uint i, uint j)
  {
    return (*matrix)(i,j);  
  }
  //---------------------------------------------------------------------------
}

#endif
