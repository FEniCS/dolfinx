/* -*- C -*- */
// Copyright (C) 2006-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006-2007
// Modified by Ola Skavhaug 2006-2007
// Modified by Garth Wells 2007-2009
// Modified by Johan Hake 2008-2009
// 
// First added:  2006-09-20
// Last changed: 2010-03-09

//=============================================================================
// SWIG directives for the DOLFIN Mesh kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Extend mesh entity iterators to work as Python iterators
//-----------------------------------------------------------------------------
%extend dolfin::MeshEntityIterator {
%pythoncode
%{
def __iter__(self):
    self.first = True
    return self

def next(self):
    self.first = self.first if hasattr(self,"first") else True
    if not self.first:
        self._increment()
    if self.end():
        self._decrease()
        raise StopIteration
    self.first = False
    return self._dereference()
%}
}

//-----------------------------------------------------------------------------
// MeshFunction macro
//-----------------------------------------------------------------------------
%define MESH_FUNCTION(TYPE,TYPENAME)
%template(MeshFunction ## TYPENAME) dolfin::MeshFunction<TYPE>;
//%typedef dolfin::MeshFunction<TYPE> MeshFunction ## TYPENAME;
%extend dolfin::MeshFunction<TYPE> 
{
  TYPE __getitem__(unsigned int i) { return (*self)[i]; }
  void __setitem__(unsigned int i, TYPE val) { (*self)[i] = val; }
  
  TYPE __getitem__(dolfin::MeshEntity& e) { return (*self)[e]; }
  void __setitem__(dolfin::MeshEntity& e, TYPE val) { (*self)[e] = val; }
}
%enddef

//-----------------------------------------------------------------------------
// Run MeshFunction macros
//-----------------------------------------------------------------------------
MESH_FUNCTION(int,Int)
MESH_FUNCTION(dolfin::uint,UInt)
MESH_FUNCTION(double,Double)
MESH_FUNCTION(bool,Bool)

%pythoncode
%{
_doc_string = MeshFunctionInt.__doc__
_doc_string += """     
    Arguments
//-----------------------------------------------------------------------------\n      String defining the type of the MeshFunction
      Allowed: 'int', 'uint', 'double', and 'bool'
    @param mesh:
      A DOLFIN mesh.
      Optional.
    @param dim:
      The topological dimension of the MeshFunction.
      Optional.
    @param filename:
      A filename with a stored MeshFunction.
      Optional.

"""
class MeshFunction(object):
    __doc__ = _doc_string
    def __new__(cls, tp, *args):
        if tp == "int":
            return MeshFunctionInt(*args)
        if tp == "uint":
            return MeshFunctionUInt(*args)
        elif tp == "double":
            return MeshFunctionDouble(*args)
        elif tp == "bool":
            return MeshFunctionBool(*args)
        else:
            raise RuntimeError, "Cannot create a MeshFunction of type '%s'." % (tp,)

del _doc_string

%}

//-----------------------------------------------------------------------------
// Extend Point interface with Python selectors
//-----------------------------------------------------------------------------
%extend dolfin::Point {
  double __getitem__(int i) { return (*self)[i]; }
  void __setitem__(int i, double val) { (*self)[i] = val; }
}
