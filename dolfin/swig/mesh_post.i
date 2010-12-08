/* -*- C -*- */
// Copyright (C) 2006-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006-2007
// Modified by Ola Skavhaug 2006-2007
// Modified by Garth Wells 2007-2010
// Modified by Johan Hake 2008-2009
//
// First added:  2006-09-20
// Last changed: 2010-12-08

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
// Extend subset iterator to work as Python iterators
//-----------------------------------------------------------------------------
%extend dolfin::SubsetIterator {
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
        raise StopIteration
    self.first = False
    return self._dereference()
%}
}

//-----------------------------------------------------------------------------
// MeshFunction macro
//-----------------------------------------------------------------------------
%define DECLARE_MESHFUNCTION(MESHFUNCTION,TYPE,TYPENAME)
%template(MESHFUNCTION ## TYPENAME) dolfin::MESHFUNCTION<TYPE>;

%feature("docstring") dolfin::MESHFUNCTION::__getitem__ "Missing docstring";
%feature("docstring") dolfin::MESHFUNCTION::__setitem__ "Missing docstring";
%extend dolfin::MESHFUNCTION<TYPE>
{
  TYPE __getitem__(unsigned int i) { return (*self)[i]; }
  void __setitem__(unsigned int i, TYPE val) { (*self)[i] = val; }

  TYPE __getitem__(dolfin::MeshEntity& e) { return (*self)[e]; }
  void __setitem__(dolfin::MeshEntity& e, TYPE val) { (*self)[e] = val; }
}
%enddef

//-----------------------------------------------------------------------------
// Macro for declaring MeshFunctions
//-----------------------------------------------------------------------------
%define DECLARE_MESHFUNCTIONS(MESHFUNCTION)
DECLARE_MESHFUNCTION(MESHFUNCTION,int,Int)
DECLARE_MESHFUNCTION(MESHFUNCTION,dolfin::uint,UInt)
DECLARE_MESHFUNCTION(MESHFUNCTION,double,Double)
DECLARE_MESHFUNCTION(MESHFUNCTION,bool,Bool)
%enddef

//-----------------------------------------------------------------------------
// Run Macros to declare the different MeshFunctions
//-----------------------------------------------------------------------------
DECLARE_MESHFUNCTIONS(MeshFunction)
DECLARE_MESHFUNCTIONS(VertexFunction)
DECLARE_MESHFUNCTIONS(EdgeFunction)
DECLARE_MESHFUNCTIONS(FaceFunction)
DECLARE_MESHFUNCTIONS(FacetFunction)
DECLARE_MESHFUNCTIONS(CellFunction)

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
        if not isinstance(tp, str):
            raise TypeError, "expected a 'str' as first argument"
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

def _new_closure(MeshType):
    assert(isinstance(MeshType,str))
    def new(cls, tp, mesh):
        if not isinstance(tp, str):
            raise TypeError, "expected a 'str' as first argument"
        if tp == "int":
            return eval("%sInt(mesh)"%MeshType)
        if tp == "uint":
            return eval("%sUInt(mesh)"%MeshType)
        elif tp == "double":
            return eval("%sDouble(mesh)"%MeshType)
        elif tp == "bool":
            return eval("%sBool(mesh)"%MeshType)
        else:
            raise RuntimeError, "Cannot create a %sFunction of type '%s'." % (MeshType, tp)

    def new(cls, tp, mesh, value):
        if not isinstance(tp, str):
            raise TypeError, "expected a 'str' as first argument"
        if tp == "int":
            return eval("%sInt(mesh, value)"%MeshType)
        if tp == "uint":
            return eval("%sUInt(mesh, value)"%MeshType)
        elif tp == "double":
            return eval("%sDouble(mesh, value)"%MeshType)
        elif tp == "bool":
            return eval("%sBool(mesh, value)"%MeshType)
        else:
            raise RuntimeError, "Cannot create a %sFunction of type '%s'." % (MeshType, tp)

    return new

# Create the named MeshFunction types
VertexFunction = type("VertexFunction", (), {"__new__":_new_closure("VertexFunction"),
                                             "__doc__":"Create MeshFunction of topological"\
                                             " dimension 0 on given mesh."})
EdgeFunction = type("EdgeFunction", (), {"__new__":_new_closure("EdgeFunction"),
                                             "__doc__":"Create MeshFunction of topological"
                                         " dimension 1 on given mesh."})
FaceFunction = type("FaceFunction", (), {"__new__":_new_closure("FaceFunction"),
                                             "__doc__":"Create MeshFunction of topological"
                                         " dimension 2 on given mesh."})
FacetFunction = type("FacetFunction", (), {"__new__":_new_closure("FacetFunction"),
                                             "__doc__":"Create MeshFunction of topological"
                                           " codimension 1 on given mesh."})
CellFunction = type("CellFunction", (), {"__new__":_new_closure("CellFunction"),
                                             "__doc__":"Create MeshFunction of topological"
                                         " codimension 0 on given mesh."})

%}


//-----------------------------------------------------------------------------
// Extend Point interface with Python selectors
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Point::__getitem__ "Missing docstring";
%feature("docstring") dolfin::Point::__setitem__ "Missing docstring";
%extend dolfin::Point {
  double __getitem__(int i) { return (*self)[i]; }
  void __setitem__(int i, double val) { (*self)[i] = val; }
}
