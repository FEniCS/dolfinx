// Type maps for PyDOLFIN

// Basic typemaps
%typemap(in)  dolfin::real = double;
%typemap(out) dolfin::real = double;
%typemap(in)  dolfin::uint = int;
%typemap(out) dolfin::uint = int;

// Typemap for dolfin::arrays as input arguments to overloaded functions.
// This converts a C++ dolfin::simple_array to a numpy array in Python.
%typemap(directorin) dolfin::simple_array<dolfin::real>& {
  {
    npy_intp dims[1] = {$1_name.size};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>($1_name.data));
  }
}

// Typemap check
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::real* {
    // General typemap
    $1 = PyArray_Check($input) ? 1 : 0;
}

// Typemap check
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::uint* {
    // General typemap
    $1 = PyArray_Check($input) ? 1 : 0;
}

// Typemap for sending numpy arrays as input to functions expecting a C array of real
%typemap(in) dolfin::real* {
    if PyArray_Check($input) {
        PyArrayObject *xa = (PyArrayObject*)($input);
        if (xa->descr->type == 'd')
            $1 = (double *)(*xa).data;
        else
            SWIG_exception(SWIG_ValueError, "numpy array of doubles expected");
    } else 
        SWIG_exception(SWIG_ValueError, "numpy array expected");
}

// Typemap for sending numpy arrays as input to functions expecting a C array of uint
%typemap(in) dolfin::uint* {
    if PyArray_Check($input) {
        PyArrayObject *xa = (PyArrayObject*)($input);
        printf("*** Checking: xa->descr->type = %c\n", xa->descr->type);
        if (xa->descr->type == 'L')
            $1 = (dolfin::uint *)(*xa).data;
        else
            SWIG_exception(SWIG_ValueError, "numpy array of unsigned integers expected");
    } else 
        SWIG_exception(SWIG_ValueError, "numpy array expected");
}

// Typecheck for Parameter typemaps
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::Parameter {
    bool pass = 0;
    pass = PyString_Check($input) ? 1 : pass;
    pass = PyInt_Check($input)    ? 1 : pass;
    pass = PyFloat_Check($input)  ? 1 : pass;
    $1 = pass;
}

// Typemap for Parameter (in)
%typemap(in) dolfin::Parameter {
    if PyString_Check($input) {
        std::string input = PyString_AsString($input);
        dolfin::Parameter tmp(input);
        $1 = tmp;
    }
    else if PyInt_Check($input) {
        int val = PyInt_AsLong($input);
        dolfin::Parameter tmp(val);
        $1 = tmp;
    }
    else if PyFloat_Check($input) {
        dolfin::Parameter tmp(PyFloat_AsDouble($input));
        $1 = tmp;
    }
}

// Typemap for Parameter (out)
%typemap(out) dolfin::Parameter {
  {
    // Custom typemap
    // std::string tmp; 

    switch ( $1.type() )
    {
    case Parameter::type_real:
      
      $result = SWIG_From_double(*&($1));
      break;

    case Parameter::type_int:
      
      $result = SWIG_From_int((int)*&($1));
      break;
      
    case Parameter::type_bool:
      
      $result = SWIG_From_bool(*&($1));
      break;
      
    case Parameter::type_string:
      
      //$result = SWIG_From_std_string(*&($1));
      //tmp = (std::string)*&($1);
      //$result = SWIG_FromCharPtrAndSize(tmp.c_str(), tmp.size());
      $result = SWIG_FromCharPtrAndSize(((std::string)*&($1)).c_str(), ((std::string)*&($1)).size());

      break;
      
    default:
      error("Unknown type for parameter.");
    }
  }
}

%include numpy_typemaps.i
