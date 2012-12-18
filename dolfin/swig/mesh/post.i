/* -*- C -*- */
// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Jansson 2006-2007
// Modified by Ola Skavhaug 2006-2007
// Modified by Garth Wells 2007-2010
// Modified by Johan Hake 2008-2009
//
// First added:  2006-09-20
// Last changed: 2011-03-11

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
// Extend SubDomain
//-----------------------------------------------------------------------------
%pythoncode
%{
_subdomain_mark_doc_string = SubDomain._mark.__doc__
%}

%extend dolfin::SubDomain {
%pythoncode
%{
# NOTE: This is a hardcoded check, which rely on SubDomain::mark only taking
# a MeshFunction as its first argument when mark is called with two arguments
def mark(self, *args):
    import common
    if len(args) == 2 and not isinstance(args[0], \
                    (MeshFunctionSizet, MeshFunctionInt,
                     MeshFunctionDouble, MeshFunctionBool)):
        common.dolfin_error("dolfin.cpp.mesh.py",
                            "mark MeshFunction",
                            "Expected a MeshFunction of type \"size_t\", \"int\", \"double\" or \"bool\"")
            
    self._mark(*args)

%}
}

%pythoncode
%{
SubDomain.mark.__func__.__doc__ = _subdomain_mark_doc_string
del _subdomain_mark_doc_string
%}

//-----------------------------------------------------------------------------
// Macro for declaring MeshFunctions
//-----------------------------------------------------------------------------
%define DECLARE_MESHFUNCTION(TYPE, TYPENAME)
%feature("docstring") dolfin::MeshFunction::__getitem__ "Missing docstring";
%feature("docstring") dolfin::MeshFunction::__setitem__ "Missing docstring";

// Extend MeshFunction interface for get and set items
%extend dolfin::MeshFunction<TYPE>
{
  TYPE __getitem__(std::size_t i) { return (*self)[i]; }
  void __setitem__(std::size_t i, TYPE val) { (*self)[i] = val; }

  TYPE __getitem__(dolfin::MeshEntity& e) { return (*self)[e]; }
  void __setitem__(dolfin::MeshEntity& e, TYPE val) { (*self)[e] = val; }

%pythoncode%{
def array(self):
    """
    Return a NumPy array view of the data
    """
    data = self._array()
    _attach_base_to_numpy_array(data, self)
    return data

%}
}

// Declare templates
%template(MeshFunction ## TYPENAME) dolfin::MeshFunction<TYPE>;
%template(CellFunction ## TYPENAME) dolfin::CellFunction<TYPE>;
%template(EdgeFunction ## TYPENAME) dolfin::EdgeFunction<TYPE>;
%template(FaceFunction ## TYPENAME) dolfin::FaceFunction<TYPE>;
%template(FacetFunction ## TYPENAME) dolfin::FacetFunction<TYPE>;
%template(VertexFunction ## TYPENAME) dolfin::VertexFunction<TYPE>;

//-----------------------------------------------------------------------------
// Modifying the interface of Hierarchical
//-----------------------------------------------------------------------------
%pythoncode %{
HierarchicalMeshFunction ## TYPENAME.leaf_node = HierarchicalMeshFunction ## TYPENAME._leaf_node
HierarchicalMeshFunction ## TYPENAME.root_node = HierarchicalMeshFunction ## TYPENAME._root_node
HierarchicalMeshFunction ## TYPENAME.child = HierarchicalMeshFunction ## TYPENAME._child
HierarchicalMeshFunction ## TYPENAME.parent = HierarchicalMeshFunction ## TYPENAME._parent
%}
%enddef


//-----------------------------------------------------------------------------
// Run Macros to declare the different MeshFunctions
//-----------------------------------------------------------------------------
DECLARE_MESHFUNCTION(std::size_t, Sizet)
DECLARE_MESHFUNCTION(unsigned int, UInt)
DECLARE_MESHFUNCTION(int, Int)
DECLARE_MESHFUNCTION(double, Double)
DECLARE_MESHFUNCTION(bool, Bool)

// Create docstrings to the MeshFunctions
%pythoncode
%{
_doc_string = MeshFunctionInt.__doc__
_doc_string += """
  *Arguments*
    tp (str)
      String defining the type of the MeshFunction
      Allowed: 'int', 'size_t', 'uint', 'double', and 'bool'
    mesh (_Mesh_)
      A DOLFIN mesh.
      Optional.
    dim (uint)
      The topological dimension of the MeshFunction.
      Optional.
    filename (str)
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
        elif tp == "size_t":
            return MeshFunctionSizet(*args)
        elif tp == "double":
            return MeshFunctionDouble(*args)
        elif tp == "bool":
            return MeshFunctionBool(*args)
        else:
            raise RuntimeError, "Cannot create a MeshFunction of type '%s'." % (tp,)

del _doc_string

def _new_closure(MeshType):
    assert(isinstance(MeshType,str))
    def new(cls, tp, mesh, value=0):
        if not isinstance(tp, str):
            raise TypeError, "expected a 'str' as first argument"
        if tp == "int":
            return eval("%sInt(mesh, value)"%MeshType)
        if tp == "uint":
            return eval("%sUInt(mesh, value)"%MeshType)
        if tp == "size_t":
            return eval("%sSizet(mesh, value)"%MeshType)
        elif tp == "double":
            return eval("%sDouble(mesh, float(value))"%MeshType)
        elif tp == "bool":
            return eval("%sBool(mesh, value)"%MeshType)
        else:
            raise RuntimeError, "Cannot create a %sFunction of type '%s'." % (MeshType, tp)

    return new

# Create the named MeshFunction types
VertexFunction = type("VertexFunction", (), \
		      {"__new__":_new_closure("VertexFunction"),\
                       "__doc__":"Create MeshFunction of topological"\
                       " dimension 0 on given mesh."})
EdgeFunction = type("EdgeFunction", (), \
                    {"__new__":_new_closure("EdgeFunction"),\
                     "__doc__":"Create MeshFunction of topological"\
                     " dimension 1 on given mesh."})
FaceFunction = type("FaceFunction", (),\
                    {"__new__":_new_closure("FaceFunction"),\
                     "__doc__":"Create MeshFunction of topological"\
                     " dimension 2 on given mesh."})
FacetFunction = type("FacetFunction", (),\
                     {"__new__":_new_closure("FacetFunction"),
                      "__doc__":"Create MeshFunction of topological"\
                      " codimension 1 on given mesh."})
CellFunction = type("CellFunction", (),\
                    {"__new__":_new_closure("CellFunction"),\
                     "__doc__":"Create MeshFunction of topological"\
                     " codimension 0 on given mesh."})
%}

//-----------------------------------------------------------------------------
// MeshValueCollection macro
//-----------------------------------------------------------------------------
%define DECLARE_MESHVALUECOLLECTION(TYPE, TYPENAME)
%template(MeshValueCollection ## TYPENAME) dolfin::MeshValueCollection<TYPE>;

%feature("docstring") dolfin::MeshValueCollection::assign "Missing docstring";

// Extend MeshFunction interface for assign methods
%extend dolfin::MeshValueCollection<TYPE>
{

  void assign(const dolfin::MeshFunction<TYPE>& mesh_function)
  {
    (*self) = mesh_function;
  }

  void assign(const dolfin::MeshValueCollection<TYPE>& mesh_value_collection)
  {
    (*self) = mesh_value_collection;
  }
}

%enddef

//-----------------------------------------------------------------------------
// Run macros for declaring MeshValueCollection
//-----------------------------------------------------------------------------
DECLARE_MESHVALUECOLLECTION(std::size_t, Sizet)
DECLARE_MESHVALUECOLLECTION(unsigned int, UInt)
DECLARE_MESHVALUECOLLECTION(int, Int)
DECLARE_MESHVALUECOLLECTION(double, Double)
DECLARE_MESHVALUECOLLECTION(bool, Bool)

// Create docstrings to the MeshValueCollection
%pythoncode
%{
_meshvaluecollection_doc_string = MeshValueCollectionInt.__doc__
_meshvaluecollection_doc_string += """
  *Arguments*
      tp (str)
         String defining the type of the MeshValueCollection
          Allowed: 'int', 'uint', 'size_t', 'double', and 'bool'
      dim (uint)
          The topological dimension of the MeshValueCollection.
          Optional.
      mesh_function (_MeshFunction_)
          The MeshValueCollection will get the values from the mesh_function
          Optional.
       mesh (Mesh)
          A mesh associated with the collection. The mesh is used to
          map collection values to the appropriate process.
          Optional, used when read from file.
      filename (std::string)
          The XML file name.
          Optional, used when read from file.
      dim (uint)
          The mesh entity dimension for the mesh value collection.
          Optional, used when read from file
"""
class MeshValueCollection(object):
    __doc__ = _meshvaluecollection_doc_string
    def __new__(cls, tp, *args):
        if not isinstance(tp, str):
            raise TypeError, "expected a 'str' as first argument"
        if tp == "int":
            return MeshValueCollectionInt(*args)
        if tp == "uint":
            return MeshValueCollectionUInt(*args)
        elif tp == "size_t":
            return MeshValueCollectionSizet(*args)
        elif tp == "double":
            return MeshValueCollectionDouble(*args)
        elif tp == "bool":
            return MeshValueCollectionBool(*args)
        else:
            raise RuntimeError, "Cannot create a MeshValueCollection of type '%s'." % (tp,)

del _meshvaluecollection_doc_string
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

//-----------------------------------------------------------------------------
// Extend Mesh interface with ufl cell method
//-----------------------------------------------------------------------------
%extend dolfin::Mesh {
%pythoncode
%{
def ufl_cell(self):
    """
    Returns the ufl cell of the mesh

    The cell corresponds to the topological dimension of the mesh.
    """
    import ufl
    cells = { 1: ufl.interval, 2: ufl.triangle, 3: ufl.tetrahedron }
    return cells[self.topology().dim()]

def coordinates(self):
    """
    * coordinates\ ()

      Get vertex coordinates.

      *Returns*
          numpy.array(float)
              Coordinates of all vertices.

      *Example*
          .. code-block:: python

              >>> mesh = dolfin.UnitSquare(1,1)
              >>> mesh.coordinates()
              array([[ 0.,  0.],
                     [ 1.,  0.],
                     [ 0.,  1.],
                     [ 1.,  1.]])
    """

    # Get coordinates
    coord = self._coordinates()

    # Attach a reference to the Mesh to the coord array
    _attach_base_to_numpy_array(coord, self)

    return coord

def cells(self):
    """
    Get cell connectivity.

    *Returns*
        numpy.array(int)
            Connectivity for all cells.

    *Example*
        .. code-block:: python

            >>> mesh = dolfin.UnitSquare(1,1)
            >>> mesh.cells()
            array([[0, 1, 3],
                  [0, 2, 3]])
    """
    # Get coordinates
    cells = self._cells()

    # Attach a reference to the Mesh to the cells array
    _attach_base_to_numpy_array(cells, self)

    return cells

%}
}

//-----------------------------------------------------------------------------
// Modifying the interface of Hierarchical
//-----------------------------------------------------------------------------
%pythoncode %{
HierarchicalMesh.leaf_node = HierarchicalMesh._leaf_node
HierarchicalMesh.root_node = HierarchicalMesh._root_node
HierarchicalMesh.child = HierarchicalMesh._child
HierarchicalMesh.parent = HierarchicalMesh._parent
%}

