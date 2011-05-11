/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-04-27
// Last changed: 2009-09-13

class Indices
{
public:

  // Constructor
  Indices():_index_size(0),_indices(0),_range(0){}

  // Constructor
  Indices(unsigned int size):_index_size(size),_indices(0),_range(0){}

  // Destructor
  virtual ~Indices(){    
    clear();
  }
  
  // Clear any created array
  void clear(){
    if (_indices) 
      delete[] _indices;
    if (_range) 
      delete[] _range;
  }

  // Returns an array of indices
  unsigned int* indices(){
    // Construct the array if not excisting
    if (!_indices) {
      _indices = new unsigned int[size()];
      for ( unsigned int i = 0; i < size(); i++)
	_indices[i] = index(i);
    }
    return _indices;
  }

  unsigned int* range(){
    // Construct the array if not excisting
    if (!_range) {
      _range = new unsigned int[size()];
      for ( unsigned int i = 0; i < size(); i++)
	_range[i] = i;
    }
    return _range;
  }
  
  // Returns the ith index, raises RuntimeError if i is out of range
  virtual unsigned int index( unsigned int i ) = 0;

  // Return the size of the indices
  unsigned int size(){ return _index_size; }

  // Check bounds of index and change from any negative to positive index
  static unsigned int check_index(int index, unsigned int vector_size){
    // Check bounds
    if ( index >= static_cast<int>(vector_size) or 
         index < -static_cast<int>(vector_size) )
      throw std::runtime_error("index out of range");
    
    // If a negative index is provided swap it
    if (index < 0)
      index += vector_size;
    
  return index;
  }
  
protected:

  unsigned int _index_size;
  unsigned int* _indices;
  unsigned int* _range;

};

class SliceIndices : public Indices
  /// SliceIndices provides a c++ wrapper class for a Python slice
{
public:

  // Constructor
  SliceIndices( PyObject* op, unsigned int vector_size ):
    Indices(),_start(0), _step(0)
  {
    if ( op == Py_None or !PySlice_Check(op) )
      throw std::runtime_error("expected slice");
    Py_ssize_t stop, start, step, index_size;
    
    if ( PySlice_GetIndicesEx((PySliceObject*)op, vector_size, &start, &stop, &step, &index_size) < 0 )
      throw std::runtime_error("invalid slice");
    _step  = step;
    _start = start;
    _index_size = index_size;
  }

  // Destructor
  virtual ~SliceIndices(){
  }
  
  // Returns the ith index, raises RuntimeError if i is out of range
  virtual unsigned int index( unsigned int i ) {
    if ( i >= size() )
      throw std::runtime_error("index out of range");
    return _start + i*_step;
  }
  
private:
  unsigned int _start, _step;
};

class ListIndices : public Indices
  /// ListIndices provides a c++ wrapper class for a Python List of integer, 
  /// which is ment to hold indices to a Vector
{
public:

  // Constructor
  ListIndices( PyObject* op, unsigned int vector_size )
    :Indices(), _list(NULL), _vector_size(vector_size)
  {
    if ( op == Py_None or !PyList_Check(op) )
      // FIXME: Is it OK, to throw exception in constructor?
      throw std::runtime_error("expected list");
    
    // Set protected member
    _index_size = PyList_Size(op);
    if ( _index_size > vector_size)
      throw std::runtime_error("index list too large");
    
    // Set members
    _vector_size = vector_size;
    _list = op;
    
    // Increase reference to list 
    Py_INCREF(_list);
  }
  
  // Destructor
  virtual ~ListIndices(){
    Py_DECREF(_list);
  }
  
  // Returns the ith index, raises RuntimeError if i is out of range
  virtual unsigned int index( unsigned int i ) {
    PyObject* op = NULL;

    // Check size of passed index
    if ( i >= size() )
      throw std::runtime_error("index out of range");
    
    // Get the index
    if (!(op=PyList_GetItem(_list, i)))
      throw std::runtime_error("invalid index");
    
    // Check for int
    if (!PyInteger_Check(op))
      throw std::runtime_error("invalid index, must be int");
    
    // Return checked index
    return check_index(PyArray_PyIntAsInt(op));
  }
  
  // Check bounds of index by calling static function in base class
  unsigned int check_index(int index){
    return Indices::check_index(index, _vector_size);
  }

private:
  PyObject* _list;
  unsigned int _vector_size;
};


class IntArrayIndices : public Indices
  /// IntArrayIndices provides a c++ wrapper class for a NumPy array of integer,
  /// which is ment to hold indices to a Vector
{
public:

  // Constructor
  IntArrayIndices( PyObject* op, unsigned int vector_size )
    :Indices(), _numpy_array(NULL), _vector_size(vector_size)
  {
    if ( op == Py_None or !( PyArray_Check(op) and PyTypeNum_ISINTEGER(PyArray_TYPE(op)) ) )
      throw std::runtime_error("expected numpy array of integers");
    
    // An initial check of the length of the array
    if (PyArray_NDIM(op)!=1)
      throw std::runtime_error("provide an 1D array");
    _index_size = PyArray_DIM(op,0);
    
    if ( _index_size > vector_size)
      throw std::runtime_error("index array too large");
    
    // Set members
    _vector_size = vector_size;
    _numpy_array = op;

    // Increase reference to numpy array
    Py_INCREF(_numpy_array);
  }
  
  // Destructor
  virtual ~IntArrayIndices(){
    Py_DECREF(_numpy_array);
  }
  
  // Returns the ith index, raises RuntimeError if i is out of range
  virtual unsigned int index( unsigned int i ) {

    // Check size of passed index
    if ( i >= size() )
      throw std::runtime_error("index out of range");
    
    // Return checked index
    return check_index(*static_cast<int*>(PyArray_GETPTR1(_numpy_array,i)));
  }
  
  // Check bounds of index by calling static function in base class
  unsigned int check_index(int index){
    return Indices::check_index(index, _vector_size);
  }

private:
  PyObject* _numpy_array;
  unsigned int _vector_size;
};


class BoolArrayIndices : public Indices
  /// BoolArrayIndices provides a c++ wrapper class for a NumPy array of bool,
  /// which is ment to hold indices to a Vector
{
public:

  // Constructor
  BoolArrayIndices( PyObject* op, unsigned int vector_size )
    :Indices()
  {
    unsigned int i, nz_ind;
    npy_bool* bool_data;
    PyArrayObject* npy_op;
    PyObject* sum_res;
    
    if ( op == Py_None or !( PyArray_Check(op) and PyArray_ISBOOL(op) ) )
      throw std::runtime_error("expected numpy array of boolean");
    npy_op = (PyArrayObject*) op;

    // An initial check of the length of the array
    if (PyArray_NDIM(op)!=1)
      throw std::runtime_error("provide an 1D array");
    
    if ( static_cast<unsigned int>(PyArray_DIM(npy_op,0)) != vector_size)
      throw std::runtime_error("non matching dimensions");
    
    bool_data = (npy_bool *) PyArray_DATA(npy_op);

    // Sum the array to get the numbers of indices
    
    sum_res = PyArray_Sum(npy_op, 0, NPY_INT, (PyArrayObject*)Py_None);
    _index_size = PyInt_AsLong(sum_res);
    Py_DECREF(sum_res);
    
    // Construct the array and fill it with indices
    _indices = new unsigned int[_index_size];
    
    nz_ind = 0;
    for ( i = 0; i < vector_size; i++ ){
      if ( bool_data[i] > 0){
	_indices[nz_ind] = i;
	nz_ind++;
      }
    }
  }
  
  // Destructor
  virtual ~BoolArrayIndices(){
  }
  
  // Returns the ith index, raises RuntimeError if i is out of range
  virtual unsigned int index( unsigned int i ) {

    // Check size of passed index
    if ( i >= size() )
      throw std::runtime_error("index out of range");
    
    // Return index
    return _indices[i];
  }
  
};


// Return a new Indice object correspondning to the input
Indices* indice_chooser( PyObject* op, unsigned int vector_size )
{
  //Check first for None
  if (op == Py_None)
    return 0;
  //  throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");
  
  // The indices
  Indices *inds;
    
  // If the provided indices are in a slice
  if (PySlice_Check(op))
    inds = new SliceIndices( op, vector_size );

  // If the provided indices are in a List
  else if (PyList_Check(op))
    inds = new ListIndices( op, vector_size );

  // If the provided indices are in a Numpy array of boolean
  else if (PyArray_Check(op) and PyArray_TYPE(op) == NPY_BOOL)
    inds = new BoolArrayIndices( op, vector_size );
  
  // If the provided indices are in a Numpy array of integers
  else if (PyArray_Check(op) and PyArray_ISINTEGER(op))
    inds = new IntArrayIndices( op, vector_size );
  else
    return 0;
  //throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  return inds;
}

