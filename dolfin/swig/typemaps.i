/* -*- C -*-  (not really, but good for syntax highlighting) */
// Type maps for PyDOLFIN

// Basic typemaps
//%typemap(in)  dolfin::uint = int;
//%typemap(out) dolfin::uint = int;

// A hack to get around incompatabilities with PyInt_Check and numpy int 
// types in python 2.6
%typecheck(SWIG_TYPECHECK_INTEGER) dolfin::uint{
    $1 = PyType_IsSubtype($input->ob_type, &PyInt_Type) ? 1 : 0;
}

%typemap(in) dolfin::uint{
  if (PyType_IsSubtype($input->ob_type, &PyInt_Type)){
    long tmp = PyInt_AsLong($input);
    if (tmp>=0)
      $1 = static_cast<dolfin::uint>(tmp);
    else
      SWIG_exception(SWIG_ValueError, "positive 'int' expected");
  }
  else
    SWIG_exception(SWIG_TypeError, "positive 'int' expected");
}

// Typemap for values (in Function)
%typemap(directorin) double* values {
  {
    // Compute size of value (number of entries in tensor value)
    dolfin::uint size = 1;
    for (dolfin::uint i = 0; i < this->function_space().element().value_rank(); i++)
      size *= this->function_space().element().value_dimension(i);

    npy_intp dims[1] = {size};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>($1_name));
  }
}

// Typemap for coordinates (in Function and SubDomain)
%typemap(directorin) const double* x {
  {
    // Compute size of x
    npy_intp dims[1] = {this->geometric_dimension()};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>(const_cast<double*>($1_name)));
  }
}

%typemap(directorin) double* y {
  {
    // Compute size of x
    npy_intp dims[1] = {this->geometric_dimension()};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>($1_name));
  }
}

//%apply const double* x { const double* y };

// Typemap check
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) double* {
    // General typemap
    $1 = PyArray_Check($input) ? 1 : 0;
}

// Typemap check
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::uint* {
    // General typemap
    $1 = PyArray_Check($input) ? 1 : 0;
}

// Typemap for sending numpy arrays as input to functions expecting a C array of real
%typemap(in) double* {
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE )
            $1  = static_cast<double*>(PyArray_DATA(xa));
        else
            SWIG_exception(SWIG_TypeError, "numpy array of doubles expected");
    } else
        SWIG_exception(SWIG_TypeError, "numpy array expected");
}

// Typemap for sending numpy arrays as input to functions expecting a C array of uint
%typemap(in) dolfin::uint* {
    if PyArray_Check($input) {
        PyArrayObject *xa = (PyArrayObject*)($input);
        printf("*** Checking: xa->descr->type = %c\n", xa->descr->type);
        if (xa->descr->type == 'L')
            $1 = (dolfin::uint *)(*xa).data;
        else
            SWIG_exception(SWIG_TypeError, "numpy array of unsigned integers expected");
    } else
        SWIG_exception(SWIG_TypeError, "numpy array expected");
}

%include numpy_typemaps.i
