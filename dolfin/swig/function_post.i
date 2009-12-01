//-----------------------------------------------------------------------------
// Extend FunctionSpace so one can check if a Function is in a FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::FunctionSpace {
%pythoncode %{
def __contains__(self,u):
    " Check whether a function is in the FunctionSpace"
    assert(isinstance(u,Function))
    return u._in(self)
%}
}

//-----------------------------------------------------------------------------
// Extend Function so f.function_space() returns a dolfin.FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::Function {
%pythoncode %{
def function_space(self):
    " Return the FunctionSpace"
    from functionspace import FunctionSpaceFromCpp
    return FunctionSpaceFromCpp(self._function_space())
%}
}

//-----------------------------------------------------------------------------
// Extend the Data class with an accessor function for the x coordinates
//-----------------------------------------------------------------------------
%extend dolfin::Data {
  PyObject* x_() {
    npy_intp adims[1];
    adims[0] = self->cell().dim();
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, adims, NPY_DOUBLE, reinterpret_cast<char *>( &(const_cast<std::vector<double>& >(self->x))[0] )));
    if ( array == NULL ) return NULL;
    PyArray_INCREF(array);
    return reinterpret_cast<PyObject*>(array);
  }
}
//-----------------------------------------------------------------------------
// Clear director typemaps
//-----------------------------------------------------------------------------
%clear const double* x;
%clear double* y;
//%clear const std::vector<double>& x;
//%clear std::vector<double>& values;
