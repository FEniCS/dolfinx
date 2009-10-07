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
// Extend the Data class with an accessor function for the x coordinates
//-----------------------------------------------------------------------------
%extend dolfin::Data {
  PyObject* x_() {
    npy_intp adims[1];
    adims[0] = self->cell().dim();
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, adims, NPY_DOUBLE, (char *)(self->x)));
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
