// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// The tensor is stored as an array where element (i_0, i_1, ... , i_m-1) is
// is given by element
//
//     i_0*n^(m-1) + i_1*n^(m-2) + ... + i_m-1.
//
// Here m is the order and n is the dimension.

#ifndef __TENSOR_H
#define __TENSOR_H

namespace dolfin {
  
  template <class T> class Tensor {
  public:

	 Tensor(int order, int dim) {

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
	 
	 //--- Evaluation, user is responsible for invoking the correct version!

	 // Evaluation for tensors of order = 1
	 T& operator() (int i0)
	 {
		return data[i0];
	 }
	 
	 // Evaluation for tensors of order = 2
	 T& operator() (int i0, int i1)
	 {
		return data[i0*index[0] + i1];
	 }

	 // Evaluation for tensors of order = 3
 	 T& operator() (int i0, int i1, int i2)
	 {
		return data[i0*index[0] + i1*index[1] + i2];
	 }

	 // Evaluation for tensors of general order
	 T& operator() (int *i)
	 {
		int ii = 0;
		for (int j = 0; j < order; j++)
		  ii += i[j] * index[j];
		return data[ii];
	 }

	 // Copy values from another tensor (without resizing)
	 void copy(const Tensor<T> &t) {
		for (int i = 0; i < size & i < t.size; i++)
		  data[i] = t.data[i];
	 }
	 
  private:

	 int order;  // Order of tensor (number of indices)
	 int dim;    // Dimension
	 int size;   // Total number of elements (dim^order)
	 int *index; // Powers of dim
	 
	 T *data;

  };

}

#endif
