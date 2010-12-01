// Copyright (C) 2006-2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-06
// Last changed: 2010-12-01

//-----------------------------------------------------------------------------
// Include code to generate a __swigversion__ and a __dolfinversion__ 
// attributes, from defines during compile time, to the cpp module
//-----------------------------------------------------------------------------
%inline %{
int dolfin_swigversion() { return  SWIGVERSION; }
std::string dolfin_version() {return DOLFIN_VERSION;}
%}

%pythoncode %{
tmp = hex(dolfin_swigversion())
__swigversion__ = ".".join([tmp[-5],tmp[-3],tmp[-2:]])
__dolfinversion__ = dolfin_version()
del tmp, dolfin_swigversion, dolfin_version
%}
