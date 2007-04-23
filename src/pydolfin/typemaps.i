// Type maps for PyDOLFIN

%typemap(in) dolfin::real = double;
%typemap(out) dolfin::real = double;
%typemap(in) dolfin::uint = int;
%typemap(out) dolfin::uint = int;

%typemap(out) dolfin::Parameter {
  {
    // Custom typemap

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
      
      $result = SWIG_From_std_string(*&($1));
      break;
      
    default:
      dolfin_error("Unknown type for parameter.");
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
