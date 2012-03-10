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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-04-27
// Last changed: 2010-03-09

// Enum for comparison functions
enum DolfinCompareType {dolfin_gt, dolfin_ge, dolfin_lt, dolfin_le, dolfin_eq, dolfin_neq};

// Returns the values from a Vector.
// copied_values are true if the returned values are copied and need clean up.
std::vector<double> _get_vector_values( dolfin::GenericVector* self)
{
  std::vector<double> values;
  self->get_local(values);
  return values;
}

/*
dolfin::Array<double>& _get_vector_values( dolfin::GenericVector* self)
{
  dolfin::Array<double>* values;
  try
  {
    // Try accessing the value pointer directly
    double* data_values = self->data();
    values = new dolfin::Array<double>(self->size(), data_values);
  }
  catch (std::runtime_error e)
  {
    // We couldn't access the values directly
    values = new dolfin::Array<double>(self->size());
    self->get_local(*values);
  }
  return *values;
}
*/

// A contains function for Vectors
bool _contains(dolfin::GenericVector* self, double value)
{
  bool contains = false;
  dolfin::uint i;
  std::vector<double> values = _get_vector_values(self);

  // Check if value is in values
  for (i = 0; i < self->size(); i++)
  {
    if (fabs(values[i]-value) < DOLFIN_EPS)
    {
      contains = true;
      break;
    }
  }
  return contains;
}
/*
bool _contains(dolfin::GenericVector* self, double value)
{
  bool contains = false;
  dolfin::uint i;
  dolfin::Array<double>& values = _get_vector_values(self);

  // Check if value is in values
  for (i = 0; i < self->size(); i++)
  {
    if (fabs(values[i]-value) < DOLFIN_EPS)
    {
      contains = true;
      break;
    }
  }

  // Clean up Array
  delete &values;
  return contains;
}
*/

// A general compare function for Vector vs scalar comparison
// The function returns a boolean numpy array
PyObject* _compare_vector_with_value( dolfin::GenericVector* self, double value, DolfinCompareType cmp_type )
{
  dolfin::uint i;
  npy_intp size = self->size();

  // Create the Numpy array
  PyArrayObject* return_array = (PyArrayObject*)PyArray_SimpleNew(1, &size, PyArray_BOOL);

  // Get the data array
  npy_bool* bool_data = (npy_bool *)PyArray_DATA(return_array);

  // Get the values
  //dolfin::Array<double>& values = _get_vector_values(self);
  std::vector<double> values = _get_vector_values(self);

  switch (cmp_type)
  {
  case dolfin_gt:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] > value) ? 1 : 0;
    break;
  case dolfin_ge:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] >= value) ? 1 : 0;
    break;
  case dolfin_lt:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] < value) ? 1 : 0;
    break;
  case dolfin_le:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] <= value) ? 1 : 0;
    break;
  case dolfin_eq:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] == value) ? 1 : 0;
    break;
  case dolfin_neq:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (values[i] != value) ? 1 : 0;
    break;
  default:
    throw std::runtime_error("invalid compare type");
  }

  // Clean up Array
  //delete &values;

  return PyArray_Return(return_array);
}

// A general compare function for Vector vs Vector comparison
// The function returns a boolean numpy array
PyObject* _compare_vector_with_vector( dolfin::GenericVector* self, dolfin::GenericVector* other, DolfinCompareType cmp_type )
{

  if (self->size() != other->size())
    throw std::runtime_error("non matching dimensions");

  dolfin::uint i;
  npy_intp size = self->size();

  // Create the Numpy array
  PyArrayObject* return_array = (PyArrayObject*)PyArray_SimpleNew(1, &size, PyArray_BOOL);

  // Get the data array
  npy_bool* bool_data = (npy_bool *)PyArray_DATA(return_array);

  // Get the values
  std::vector<double> self_values = _get_vector_values(self);
  std::vector<double> other_values = _get_vector_values(other);

  switch (cmp_type)
  {
  case dolfin_gt:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] > other_values[i]) ? 1 : 0;
    break;
  case dolfin_ge:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] >= other_values[i]) ? 1 : 0;
    break;
  case dolfin_lt:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] < other_values[i]) ? 1 : 0;
    break;
  case dolfin_le:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] <= other_values[i]) ? 1 : 0;
    break;
  case dolfin_eq:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] == other_values[i]) ? 1 : 0;
    break;
  case dolfin_neq:
    for ( i = 0; i < self->size(); i++)
      bool_data[i] = (self_values[i] != other_values[i]) ? 1 : 0;
    break;
  default:
    throw std::runtime_error("invalid compare type");
  }

  // If we have created temporary values, delete them
  delete &self_values;
  delete &other_values;

  return PyArray_Return(return_array);
}

// Get single Vector item
double _get_vector_single_item( dolfin::GenericVector* self, int index )
{
  double value;
  dolfin::uint i(Indices::check_index(index, self->size()));
  self->get_local(&value, 1, &i);
  return value;
}

// Get item for slice, list, or numpy array object
boost::shared_ptr<dolfin::GenericVector> _get_vector_sub_vector( const dolfin::GenericVector* self, PyObject* op )
{

  Indices* inds;
  double* values;
  dolfin::uint* range;
  dolfin::uint* indices;
  boost::shared_ptr<dolfin::GenericVector> return_vec;
  dolfin::uint m;

  // Get the correct Indices
  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  // Fill the return vector
  try {
    indices = inds->indices();
  }

  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e)
  {
    delete inds;
    throw;
  }

  m = inds->size();

  // Create a default Vector
  return_vec = self->factory().create_vector();

  // Resize the vector to the size of the indices
  return_vec->resize(m);

  range  = inds->range();

  values = new double[m];

  self->get_local(values, m, indices);
  return_vec->set(values, m, range);
  return_vec->apply("insert");

  delete inds;
  delete [] values;
  return return_vec;
}

// Set items using a GenericVector
void _set_vector_items_vector( dolfin::GenericVector* self, PyObject* op, dolfin::GenericVector& other )
{
  // Get the correct Indices
  Indices* inds;
  double* values;
  dolfin::uint* range;
  dolfin::uint* indices;
  dolfin::uint m;

  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  // Check for size of indices
  if ( inds->size() != other.size() )
  {
    delete inds;
    throw std::runtime_error("non matching dimensions on input");
  }

  // Get the indices
  try
  {
    indices = inds->indices();
  }

  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e)
  {
    delete inds;
    throw;
  }

  m = inds->size();
  range = inds->range();
  values = new double[m];

  // Get and set values
  other.get_local(values, m, range);
  self->set(values, m, indices);
  self->apply("insert");

  delete inds;
  delete[] values;
}

// Set items using a GenericVector
void _set_vector_items_array_of_float( dolfin::GenericVector* self, PyObject* op, PyObject* other )
{
  Indices* inds;
  double* values;
  dolfin::uint* indices;
  dolfin::uint m;
  bool casted = false;

  // Check type of values
  if ( !(other != Py_None and PyArray_Check(other) and PyArray_ISNUMBER(other) and PyArray_NDIM(other) == 1 ))
    throw std::runtime_error("expected a 1D numpy array of numbers");
  if (PyArray_TYPE(other)!=NPY_DOUBLE)
  {
    casted = true;
    other = PyArray_Cast(reinterpret_cast<PyArrayObject*>(other),NPY_DOUBLE);
  }


  // Get the correct Indices
  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  // Check for size of indices
  if ( inds->size() != static_cast<dolfin::uint>(PyArray_DIM(other,0)) )
  {
    delete inds;
    throw std::runtime_error("non matching dimensions on input");
  }

  // Fill the vector using the slice and the provided values
  try {
    indices = inds->indices();
  }

  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e)
  {
    delete inds;
    throw;
  }

  m = inds->size();

  // Get the contigous data from the numpy array
  values = (double*) PyArray_DATA(other);
  self->set(values, m, indices);
  self->apply("insert");

  // Clean casted array
  if (casted)
  {
    Py_DECREF(other);
  }
  delete inds;
}

// Set item(s) using single value
void _set_vector_items_value( dolfin::GenericVector* self, PyObject* op, double value )
{
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size())) == 0 )
  {

    // If the index is an integer
    if( op != Py_None and PyInteger_Check(op))
      self->setitem(Indices::check_index(static_cast<int>(PyInt_AsLong(op)), self->size()), value);
    else
      throw std::runtime_error("index must be either an integer, a slice, a list or a Numpy array of integer");
  }
  // The op is a Indices
  else
  {
    double* values;
    dolfin::uint* indices;
    dolfin::uint i;
    // Fill the vector using the slice
    try {
      indices = inds->indices();
    }

    // We can only throw and catch runtime_errors. These will be caught by swig.
    catch (std::runtime_error e){
      delete inds;
      throw;
    }

    // Fill and array with the value and call set()
    values = new double[inds->size()];
    for ( i = 0; i < inds->size(); i++)
      values[i] = value;

    self->set(values, inds->size(), indices);

    delete inds;
    delete[] values;
  }
  self->apply("insert");
}

// Get single item from Matrix
double _get_matrix_single_item( const dolfin::GenericMatrix* self, int m, int n )
{
  double value;
  dolfin::uint _m(Indices::check_index(m, self->size(0)));
  dolfin::uint _n(Indices::check_index(n, self->size(1)));
  self->get(&value, 1, &_m, 1, &_n);
  return value;
 }

// Get items for slice, list, or numpy array object
boost::shared_ptr<dolfin::GenericVector> _get_matrix_sub_vector( dolfin::GenericMatrix* self, dolfin::uint single, PyObject* op, bool row )
{
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size(row ? 1 : 0))) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  dolfin::uint* indices;
  try
  {
    // Get the indices in a c array
    indices = inds->indices();
  }
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e)
  {
    delete inds;
    throw;
  }

  // Create the value array and get the values from the matrix
  dolfin::Array<double>* values = new dolfin::Array<double>(inds->size());
  if (row)
    // If returning a single row
    self->get(values->data(), 1, &single, inds->size(), indices);
  else
    // If returning a single column
    self->get(values->data(), inds->size(), indices, 1, &single);

  // Create the return vector and set the values
  boost::shared_ptr<dolfin::GenericVector> return_vec = self->factory().create_vector();
  self->resize(*return_vec, 1);

  std::vector<double> _values(values->data(), values->data() + inds->size());
  return_vec->set_local(_values);
  return_vec->apply("insert");

  // Clean up
  delete values;
  delete inds;

  return return_vec;
}

/*
// Get items for slice, list, or numpy array object
dolfin::GenericMatrix* _get_matrix_sub_matrix(const dolfin::GenericMatrix* self,
                                              PyObject* row_op, PyObject* col_op )
{
  dolfin::GenericMatrix* return_mat;
  dolfin::uint i, j, k, m, n, nz_i;
  dolfin::uint* col_index_array;
  dolfin::uint* tmp_index_array;
  bool same_indices;
  Indices* row_inds;
  Indices* col_inds;
  double* values;

  // Instantiate the row indices
  if ( (row_inds = indice_chooser(row_op, self->size(0))) == 0 )
    throw std::runtime_error("row indices must be either a slice, a list or a Numpy array of integer");

  // The number of rows
  m = row_inds->size();

  // If None is provided for col_op we assume symmetric indices
  if (col_op == Py_None)
  {
    same_indices = true;

    // Check size of cols
    if (m > self->size(1))
    {
      delete row_inds;
      throw std::runtime_error("num indices excedes the number of columns");
    }

    // Symetric rows and columns yield equal column and row indices
    col_inds = row_inds;
    n = m;

  }
  // Non symetric rows and cols
  else
  {
    same_indices = false;

    // Instantiate the col indices
    if ( (col_inds = indice_chooser(col_op, self->size(1))) == 0 )
    {
      delete row_inds;
      throw std::runtime_error("col indices must be either a slice, a list or a Numpy array of integer");
    }

    // The number of columns
    n = col_inds->size();

  }

  // Access the column indices
  try
  {
    col_index_array = col_inds->indices();
  }

  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e)
  {
    delete row_inds;
    if (!same_indices)
      delete col_inds;
    throw;
  }

  // Create the return matrix
  return_mat = self->factory().create_matrix();

  throw std::runtime_error("Python interface for slices needs to be updated.");
  //return_mat->resize(m, n);

  // Zero the matrix (needed for the uBLASDenseMatrix)
  return_mat->zero();

  // Fill the get the values from me and set non zero values in return matrix
  tmp_index_array = new dolfin::uint[n];
  values = new double[n];
  for (i = 0; i < row_inds->size(); i++)
  {
    // Get all column values
    k = row_inds->index(i);
    self->get(values, 1, &k, n, col_index_array);
    // Collect non zero values
    nz_i = 0;
    for (j = 0; j < col_inds->size(); j++)
    {
      if ( !(fabs(values[j]) < DOLFIN_EPS) )
      {
        tmp_index_array[nz_i] = j;
        values[nz_i] = values[j];
        nz_i++;
      }
    }

    // Set the non zero values to return matrix
    return_mat->set(values, 1, &i, nz_i, tmp_index_array);
  }

  // Clean up
  delete row_inds;
  if ( !same_indices )
    delete col_inds;

  delete[] values;
  delete[] tmp_index_array;

  return_mat->apply("insert");
  return return_mat;
}
*/

// Set single item in Matrix
void _set_matrix_single_item( dolfin::GenericMatrix* self, int m, int n, double value )
{
  dolfin::uint _m(Indices::check_index(m, self->size(0)));
  dolfin::uint _n(Indices::check_index(n, self->size(1)));
  self->set(&value, 1, &_m, 1, &_n);
  self->apply("insert");
 }

void _set_matrix_items_array_of_float( dolfin::GenericMatrix* self,  PyObject* op, PyObject* other ){}

void _set_matrix_items_matrix(dolfin::GenericMatrix* self, dolfin::GenericMatrix*) {}

void _set_matrix_items_vector(dolfin::GenericMatrix* self, PyObject* op, dolfin::GenericVector& other){}
