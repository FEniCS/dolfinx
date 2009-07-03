// Renames for NewParameter
%rename (__int__) dolfin::NewParameter::operator int() const;
%rename (__float__) dolfin::NewParameter::operator double() const;
%rename (__str__) dolfin::NewParameter::operator std::string() const;
%rename (_assign) dolfin::NewParameter::operator=;

// Renames and ignores for NewParameters
%rename (_parameter_keys) dolfin::NewParameters::parameter_keys;
%rename (_parameter_set_keys) dolfin::NewParameters::parameter_set_keys;
%rename (_get_parameter_set) dolfin::NewParameters::operator[];
%rename (_get_parameter) dolfin::NewParameters::operator();
%rename (__str__) dolfin::NewParameters::str const;
%ignore dolfin::NewParameters::parse;
%ignore dolfin::NewParameters::update;

// Typemaps (in) for std::vectors<std::string>
%typecheck(SWIG_TYPECHECK_STRING_ARRAY) std::set<std::string> {
    $1 = PySequence_Check($input) ? 1 : 0;
}

%typemap(in) std::set<std::string> (std::set<std::string> tmp) {
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"expected a list of 'str'");
    return NULL;
  }
  int list_length = PyList_Size($input);
  if (!list_length > 0){
    PyErr_SetString(PyExc_ValueError,"expected a list with length > 0");	
    return NULL;
  }
  for (i = 0; i < list_length; i++) {
    PyObject *o = PyList_GetItem($input,i);
    if (PyString_Check(o)) {
      tmp.insert(std::string(PyString_AsString(o)));
    } else {
      PyErr_SetString(PyExc_TypeError,"provide a list of strings");
      return NULL;
    }
  }
  $1 = tmp;
}

