// Type maps for PyDOLFIN

%typemap(in) dolfin::real = double;
%typemap(out) dolfin::real = double;
%typemap(in) dolfin::uint = int;
%typemap(out) dolfin::uint = int;

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


%typemap(directorin) dolfin::real* {
  {
    // Custom typemap
    $input = SWIG_NewPointerObj((void *) $1_name, $1_descriptor, $owner);
  }
}

%typemap(directorin) dolfin::real const * {
  {
    // Custom typemap
    $input = SWIG_NewPointerObj((void *) $1_name, $1_descriptor, $owner);
  }
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::real* {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::uint* {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) dolfin::Parameter {
    bool pass = 0;
    pass = PyString_Check($input) ? 1 : pass;
    pass = PyInt_Check($input)    ? 1 : pass;
    pass = PyFloat_Check($input)  ? 1 : pass;
    $1 = pass;
}

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
