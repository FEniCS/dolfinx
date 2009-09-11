// Renames for Parameter
// For some obscure reason we need to rename Parameter
%rename (NewParameter) dolfin::Parameter;
%rename (__int__) dolfin::Parameter::operator int() const;
%rename (__float__) dolfin::Parameter::operator double() const;
%rename (__str__) dolfin::Parameter::operator std::string() const;
%rename (_assign) dolfin::Parameter::operator=;
%rename (_assign_bool) dolfin::Parameter::operator= (bool value);
%rename (_add) dolfin::Parameters::add;
%rename (_add_bool) dolfin::Parameters::add(std::string key, bool value);

// Renames and ignores for Parameters
%rename (_get_parameter_keys) dolfin::Parameters::get_parameter_keys;
%rename (_get_parameter_set_keys) dolfin::Parameters::get_parameter_set_keys;
%rename (_get_parameter_set) dolfin::Parameters::operator();
%rename (_get_parameter) dolfin::Parameters::operator[];
%ignore dolfin::Parameters::parse;
%ignore dolfin::Parameters::update;

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

%typemap(in, numinputs=0) std::vector<std::string>& keys (std::vector<std::string> tmp_vec){
  $1 = &tmp_vec;
}

%typemap(argout) std::vector<std::string>& keys
{
  int size = $1->size();
  PyObject* ret = PyList_New(size);
  PyObject* tmp_Py_str = 0;
  for (int i=0; i < size; i++)
  {
    tmp_Py_str = PyString_FromString((*$1)[i].c_str());
    if (PyList_SetItem(ret,i,tmp_Py_str)<0)
    {
      PyErr_SetString(PyExc_ValueError,"something wrong happened when copying std::string to Python");
      return NULL;
    }
  }
  $result = ret;
}

