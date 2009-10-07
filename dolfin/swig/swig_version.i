// Copyright (C) 2006-2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-06
// Last changed: 2009-09-23

//-----------------------------------------------------------------------------
// Include code to generate a __swigversion__ attribute to the cpp module
// Add prefix to avoid naming problems with other modules
//-----------------------------------------------------------------------------
%inline %{
int dolfin_swigversion() { return  SWIGVERSION; }
%}

%pythoncode %{
tmp = hex(dolfin_swigversion())
__swigversion__ = ".".join([tmp[-5],tmp[-3],tmp[-2:]])

del tmp, dolfin_swigversion
%}
