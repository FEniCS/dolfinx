// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005

#ifndef __TENSOR_H
#define __TENSOR_H

namespace dolfin
{

  /// A Tensor is a datastructure containing values accessed by a number of
  /// indices. The number of such indices is the order of the tensor. If the
  /// order is 1, then the Tensor behaves as a vector, and if the order is 2,
  /// then the Thensor behaves as a matrix, but also higher orders (more
  /// indices) can be used.
  ///
  /// The number of possible values for each index is the dimension of the
  /// tensor.
  ///
  /// The tensor is stored as an array where element (i_0, i_1, ... , i_m-1) is
  /// is given by element
  ///
  ///     i_0*n^(m-1) + i_1*n^(m-2) + ... + i_m-1,
  ///
  /// where m is the order and n is the dimension.

  template <class T> class Tensor
  {
  public:

    /// Create empty tensor
    Tensor();
    
    /// Create tensor of given order and dimension
    Tensor(int order, int dim);

    /// Copy constructor
    Tensor(const Tensor& t);

    /// Destructor
    ~Tensor();

    /// Initialize to given order and dimension
    void init(int order, int dim);

    /// Clear tesor
    void clear();
    
    /// Evaluation for tensors of order = 1
    T& operator() (int i0);
    
    /// Evaluation for tensors of order = 2
    T& operator() (int i0, int i1);
    
    /// Evaluation for tensors of order = 3
    T& operator() (int i0, int i1, int i2);
    
    /// Evaluation for tensors of general order
    T& operator() (int* i);

    /// Assignment
    void operator=(const Tensor<T>& t);
    
  private:
    
    int order;  // Order of tensor (number of indices)
    int dim;    // Dimension
    int size;   // Total number of elements (dim^order)
    int *index; // Powers of dim
    
    T *data;
    
  };

  //---------------------------------------------------------------------------
  // Implementation of Tensor
  //---------------------------------------------------------------------------
  template <class T> Tensor<T>::Tensor() 
  {
    order = 0;
    dim = 0;
    size = 0;
    index = 0;
    data = 0;
  }
  //---------------------------------------------------------------------------
  template <class T> Tensor<T>::Tensor(int order, int dim)
  {
    dolfin_assert(order > 0);
    dolfin_assert(dim > 0);

    this->order = 0;
    this->dim = 0;
    size = 0;
    index = 0;
    data = 0;
    
    init(order, dim);
  }
  //---------------------------------------------------------------------------
  template <class T> Tensor<T>::Tensor(const Tensor& t)
  {
    order = 0;
    dim = 0;
    size = 0;
    index = 0;
    data = 0;

    *this = t;
  }
  //---------------------------------------------------------------------------
  template <class T> Tensor<T>::~Tensor()
  {
    clear();
  }
  //---------------------------------------------------------------------------
  template <class T> void Tensor<T>::init(int order, int dim)
  {
    dolfin_assert(order > 0);
    dolfin_assert(dim > 0);

    clear();
    
    this->order = order;
    this->dim = dim;
    
    // Initialise the powers of dim: index = [ dim^(order-1) ... 1 ]
    index = new int[order];
    for (int i = order - 1; i >= 0; i--) {
      if ( i == (order - 1) )
	index[i] = 1;
      else
	index[i] = index[i+1] * dim;
    }
    
    // Initialise the tensor
    size = index[0] * dim;
    data = new T[size];
    for (int i = 0; i < size; i++)
      data[i] = 0;
    
  }
  //---------------------------------------------------------------------------
  template <class T> void Tensor<T>::clear()
  {
    order = 0;
    dim = 0;
    size = 0;
    
    if ( index )
      delete [] index;
    index = 0;
    
    if ( data )
      delete [] data;
    data = 0;
  }
  //---------------------------------------------------------------------------    
  template <class T> T& Tensor<T>::operator() (int i0)
  {
    return data[i0];
  }
  //---------------------------------------------------------------------------    
  template <class T> T& Tensor<T>::operator() (int i0, int i1)
  {
    return data[i0*index[0] + i1];
  }
  //---------------------------------------------------------------------------    
  template <class T> T& Tensor<T>::operator() (int i0, int i1, int i2)
  {
    return data[i0*index[0] + i1*index[1] + i2];
  }
  //---------------------------------------------------------------------------    
  template <class T>  T& Tensor<T>::operator() (int* i)
  {
    int ii = 0;
    for (int j = 0; j < order; j++)
      ii += i[j] * index[j];
    return data[ii];
  }
  //---------------------------------------------------------------------------    
  template <class T> void Tensor<T>::operator= (const Tensor<T>& t)
  {
    clear();
    init(t.order, t.dim);
    
    for (int i = 0; i < size; i++)
      data[i] = t.data[i];
  }
  //---------------------------------------------------------------------------    
  
}

#endif
