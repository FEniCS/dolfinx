// Include code to generate a __swigversion__ attribute to the cpp module
// Add prefix to avoid naming problems with other modules
%inline %{
int dolfin_swigversion() { return  SWIGVERSION; }
%}

%pythoncode %{
"""Preliminary code for adding swig version to cpp module. Someone (tm) finish
this.
"""
tmp = hex(dolfin_swigversion())
__swigversion__ = ".".join([tmp[-5],tmp[-3],tmp[-2:]])

del tmp, dolfin_swigversion
%}
