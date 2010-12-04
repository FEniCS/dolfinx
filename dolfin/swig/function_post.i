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
