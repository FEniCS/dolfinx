//-----------------------------------------------------------------------------
// Extend FunctionSpace so one can check if a Function is in a FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::FunctionSpace {
%pythoncode %{
def __contains__(self,u):
    "Check whether a function is in the FunctionSpace"
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
    "Return the FunctionSpace"
    from dolfin.function.functionspace import FunctionSpaceFromCpp
    return FunctionSpaceFromCpp(self._function_space())
%}
}

//-----------------------------------------------------------------------------
// Extend the Data class with an accessor function for the x coordinates
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Data::x_ "Missing docstring";
%extend dolfin::Data {
  PyObject* x_() {
    npy_intp adims[1];
    adims[0] = self->x.size();
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, adims, NPY_DOUBLE, (char *)(self->x.data().get())));
    if ( array == NULL ) return NULL;
    return reinterpret_cast<PyObject*>(array);
  }
}

