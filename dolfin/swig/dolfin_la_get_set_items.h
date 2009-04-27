// Get single Vector item 
double _get_vector_single_item( dolfin::GenericVector* self, int index ) {
  double value(0); 
  dolfin::uint i(Indices::check_index(index, self->size()));
  self->get(&value, 1, &i); 
  return value; 
}

// Get item for slice, list, or numpy array object
dolfin::GenericVector* _get_vector_sub_vector( dolfin::GenericVector* self, PyObject* op ){
  
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");
  
  // Create a default Vector
  dolfin::GenericVector * return_vec = self->factory().create_vector();

  // Resize the vector to the size of the indices
  return_vec->resize(inds->size());
  
  // Fill the return vector
  try {
    for ( dolfin::uint i=0; i < inds->size(); i++)
      return_vec->setitem(i,self->getitem(inds->index(i)));
  }
  
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e){
    delete return_vec;
    delete inds;
    throw;
  }
  
  delete inds;
  return return_vec;
}

// Set items using a GenericVector
void _set_vector_items_vector( dolfin::GenericVector* self, PyObject* op, dolfin::GenericVector& values ){
  
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");
  
  // Check for size of indices
  if ( inds->size() != values.size() ){
    throw std::runtime_error("non matching dimensions on input");
  }
  
  // Fill the vector using the slice and the provided values
  try {
    for ( dolfin::uint i=0; i < inds->size(); i++)
      self->setitem(inds->index(i),values.getitem(i));
  }
  
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e){
    delete inds;
    throw;
  }
  delete inds;
}

// Set items using a GenericVector
void _set_vector_items_array_of_float( dolfin::GenericVector* self, PyObject* op, PyObject* values ){
  
  // Check type of values
  if ( !(op != Py_None and PyArray_Check(op) and 
	 PyArray_ISFLOAT(op) and PyArray_NDIM(op) == 1 ) )
    throw std::runtime_error("expected a 1D numpy array of float");
  
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size())) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");
  
  // Check for size of indices
  if ( inds->size() != static_cast<dolfin::uint>(PyArray_DIM(op,0)) )
    throw std::runtime_error("non matching dimensions on input");
  
  // Fill the vector using the slice and the provided values
  try {
    for ( dolfin::uint i=0; i < inds->size(); i++)
      self->setitem(inds->index(i), *static_cast<double*>( PyArray_GETPTR1(op,i) ));
  }
  
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e){
    delete inds;
    throw;
  }
  delete inds;
}

// Set item(s) using single value
void _set_vector_items_value( dolfin::GenericVector* self, PyObject* op, double value ){
  
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size())) == 0 ) {
    
    // If the index is an integer
    if( op != Py_None and PyInt_Check(op))
      self->setitem(Indices::check_index(PyInt_AsLong(op), self->size()), value);
    else
      throw std::runtime_error("index must be either an integer, a slice, a list or a Numpy array of integer");

  }
  
  // The op is a Indices
  else {
    
    // Fill the vector using the slice
    try {
      for ( dolfin::uint i=0; i < inds->size(); i++)
	self->setitem(inds->index(i),value);
    }
  
    // We can only throw and catch runtime_errors. These will be caught by swig.
    catch (std::runtime_error e){
      delete inds;
      throw;
    }
    delete inds;
  }
}

// Get single item from Matrix
double _get_matrix_single_item( dolfin::GenericMatrix* self, int m, int n ) {
  double value(0); 
  dolfin::uint _m(Indices::check_index(m, self->size(0)));
  dolfin::uint _n(Indices::check_index(n, self->size(1)));
  self->get(&value, 1, &_m, 1, &_n); 
  return value; 
 }

// Get items for slice, list, or numpy array object
dolfin::GenericVector* _get_matrix_sub_vector( dolfin::GenericMatrix* self, dolfin::uint single, PyObject* op, bool row ){
  
  // Get the correct Indices
  Indices* inds;
  if ( (inds = indice_chooser(op, self->size(row ? 1 : 0))) == 0 )
    throw std::runtime_error("index must be either a slice, a list or a Numpy array of integer");

  dolfin::uint* indices;
  try {
    // Get the indices in a c array
    indices = inds->array();
  }
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e){
    delete inds;
    throw;
  }
   
  // Create the value array and get the values from the matrix
  double* values = new double[inds->size()];
  if (row)
    // If returning a single row
    self->get(values, 1, &single, inds->size(), indices);
  else
    // If returning a single column
    self->get(values, inds->size(), indices, 1, &single);
  
  // Create the return vector and set the values
  dolfin::GenericVector * return_vec = self->factory().create_vector();
  return_vec->resize(inds->size());
  return_vec->set(values);
  
  // Clean up
  delete[] values;
  delete inds;
  
  return return_vec;
}

// Get items for slice, list, or numpy array object
dolfin::GenericMatrix* _get_matrix_sub_matrix( dolfin::GenericMatrix* self, PyObject* row_op, PyObject* col_op ){
  
  dolfin::GenericMatrix * return_mat;
  dolfin::uint i, j, k, m, n, nz_i;
  dolfin::uint *col_index_array;
  dolfin::uint *tmp_index_array;
  bool same_indices;
  Indices* row_inds;
  Indices* col_inds;
  double* values;
  
  // Instantiate the row indices
  if ( ( row_inds = indice_chooser( row_op, self->size(0))) == 0 )
    throw std::runtime_error("row indices must be either a slice, a list or a Numpy array of integer");
  
  // The number of rows
  m = row_inds->size();
  
  // If None is provided for col_op we assume symmetric indices
  if ( col_op == Py_None ){
    
    same_indices = true;
      
    // Check size of cols
    if ( m > self->size(1) ){
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
    if ( ( col_inds = indice_chooser( col_op, self->size(1))) == 0 ){
      delete row_inds;
      throw std::runtime_error("col indices must be either a slice, a list or a Numpy array of integer");
    }
    
    // The number of columns
    n = col_inds->size();

  }

  // Access the collumn indices
  try {
    col_index_array = col_inds->array();
  }
  
  // We can only throw and catch runtime_errors. These will be caught by swig.
  catch (std::runtime_error e){
    delete row_inds;
    if ( !same_indices )
      delete col_inds;
    throw;
  }

  // Create the return matrix
  return_mat = self->factory().create_matrix();
  return_mat->resize(m,n);
  
  // Fill the get the values from me and set non zero values in return matrix
  tmp_index_array = new dolfin::uint[n];
  values = new double[n];
  for ( i = 0; i < row_inds->size(); i++ ){
    
    // Get all column values
    k = row_inds->index(i);
    self->get(values, 1, &k, n, col_index_array);
    // Collect non zero values
    nz_i = 0;
    for ( j = 0; j < col_inds->size(); j++ ){
      if ( !( -DOLFIN_EPS < values[j] and values[j] < DOLFIN_EPS ) ){
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
  delete[] col_index_array;
  
  return_mat->apply();
  return return_mat;
}

// Set single item in Matrix
void _set_matrix_single_item( dolfin::GenericMatrix* self, int m, int n, double value ) {
  dolfin::uint _m(Indices::check_index(m, self->size(0)));
  dolfin::uint _n(Indices::check_index(n, self->size(1)));
  self->set(&value, 1, &_m, 1, &_n); 
  self->apply();
 }

