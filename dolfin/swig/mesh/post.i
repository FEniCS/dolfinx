/* -*- C -*- */
// Copyright (C) 2006-2015 Anders Logg
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
// Modified by Martin Alnaes 2013-2015

//=============================================================================
// SWIG directives for the DOLFIN Mesh kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Extend Point for Python 3
//-----------------------------------------------------------------------------
%extend dolfin::Point {
%pythoncode %{
__truediv__ = __div__
__itruediv__ = __idiv__
%}
}

//-----------------------------------------------------------------------------
// Extend mesh entity iterators to work as Python iterators
//-----------------------------------------------------------------------------
%extend dolfin::MeshEntityIterator {
%pythoncode
%{
def __iter__(self):
    self.first = True
    return self

def __next__(self):
    self.first = self.first if hasattr(self,"first") else True
    if not self.first:
        self._increment()
    if self.end():
        self._decrease()
        raise StopIteration
    self.first = False
    return self._dereference()
# Py2/Py3
next = __next__
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

def __next__(self):
    self.first = self.first if hasattr(self,"first") else True
    if not self.first:
        self._increment()
    if self.end():
        raise StopIteration
    self.first = False
    return self._dereference()
# Py2/Py3
next = __next__
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
def mark(self, *args, **kwargs):
    from . import common
    if len(args) == 2 and not isinstance(args[0], \
                    (MeshFunctionSizet, MeshFunctionInt,
                     MeshFunctionDouble, MeshFunctionBool)):
        common.dolfin_error("dolfin.cpp.mesh.py",
                            "mark MeshFunction",
                            "Expected a MeshFunction of type \"size_t\", \"int\", \"double\" or \"bool\"")

    if ("check_midpoint" in kwargs):
        args = args + (kwargs["check_midpoint"],)
    self._mark(*args)
%}
}

%pythoncode
%{
import sys
if sys.version_info[0] > 2:
    SubDomain.mark.__doc__ = _subdomain_mark_doc_string
else:
    SubDomain.mark.__func__.__doc__ = _subdomain_mark_doc_string
del _subdomain_mark_doc_string
%}

//-----------------------------------------------------------------------------
// Macro for declaring MeshFunctions
//-----------------------------------------------------------------------------
%define DECLARE_MESHFUNCTION(TYPE, TYPENAME)
%feature("docstring") dolfin::MeshFunction::_getitem "Missing docstring";
%feature("docstring") dolfin::MeshFunction::_setitem "Missing docstring";

// Extend MeshFunction interface for get and set items
%extend dolfin::MeshFunction<TYPE>
{
  TYPE _getitem(std::size_t i)
  { return (*self)[i]; }
  void _setitem(std::size_t i, TYPE val)
  { (*self)[i] = val; }

  TYPE _getitem(dolfin::MeshEntity& e)
  { return (*self)[e]; }
  void _setitem(dolfin::MeshEntity& e, TYPE val)
  { (*self)[e] = val; }

%pythoncode%{
def array(self):
    """
    Return a NumPy array view of the data
    """
    data = self._array()
    _attach_base_to_numpy_array(data, self)
    return data

def __getitem__(self, index):
    if not isinstance(index, (int, MeshEntity)):
        raise TypeError("expected an int or a MeshEntity as index argument")

    if isinstance(index, MeshEntity):
        entity = index
        assert entity.mesh().id() == self.mesh().id(), "MeshEntity and MeshFunction do not share the same mesh"
        assert entity.dim() == self.dim(), "MeshEntity and MeshFunction do not share the same topological dimensions"

        index = entity.index()

    while index < 0:
        index += self.size()
    if index >= self.size():
        raise IndexError("index out of range")
    return self._getitem(index)

def __setitem__(self, index, value):
    if not isinstance(index, (int, MeshEntity)):
        raise TypeError("expected an int or a MeshEntity as index argument")

    if isinstance(index, MeshEntity):
        entity = index
        assert entity.mesh().id() == self.mesh().id(), "MeshEntity and MeshFunction do not share the same mesh"
        assert entity.dim() == self.dim(), "MeshEntity and MeshFunction do not share the same topological dimensions"

        index = entity.index()

    while index < 0:
        index += self.size()
    if index >= self.size():
        raise IndexError("index out of range")
    self._setitem(index, value)

def __len__(self):
    return self.size()

def ufl_id(self):
    "Returns an id that UFL can use to decide if two objects are the same."
    return self.id()

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
      Allowed: 'int', 'size_t', 'double', and 'bool'
    mesh (_Mesh_)
      A DOLFIN mesh.
      Optional.
    dim (unsigned int)
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
            raise TypeError("expected a 'str' as first argument")
        if tp == "int":
            return MeshFunctionInt(*args)
        if tp == "uint":
            from . import common
            common.deprecation("uint-valued MeshFunction", "1.1.0", "TBA",
                               "Typename \"uint\" has been changed to \"size_t\".")
            return MeshFunctionSizet(*args)
        elif tp == "size_t":
            return MeshFunctionSizet(*args)
        elif tp == "double":
            return MeshFunctionDouble(*args)
        elif tp == "bool":
            return MeshFunctionBool(*args)
        else:
            raise RuntimeError("Cannot create a MeshFunction of type '%s'." % (tp,))

del _doc_string

def _new_closure(MeshType):
    assert(isinstance(MeshType, str))
    def new(cls, tp, mesh, value=0):
        if not isinstance(tp, str):
            raise TypeError("expected a 'str' as first argument")
        if tp == "int":
            return eval("%sInt(mesh, value)"%MeshType)
        if tp == "uint":
            return eval("%sSizet(mesh, value)"%MeshType)
        if tp == "size_t":
            return eval("%sSizet(mesh, value)"%MeshType)
        elif tp == "double":
            return eval("%sDouble(mesh, float(value))"%MeshType)
        elif tp == "bool":
            value = bool(value) if isinstance(value, int) else value
            return eval("%sBool(mesh, value)"%MeshType)
        else:
            raise RuntimeError("Cannot create a %sFunction of type '%s'." % (MeshType, tp))

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
          Allowed: 'int', 'size_t', 'double', and 'bool'
      dim (unsigned int)
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
      dim (unsigned int)
          The mesh entity dimension for the mesh value collection.
          Optional, used when read from file
"""
class MeshValueCollection(object):
    __doc__ = _meshvaluecollection_doc_string
    def __new__(cls, tp, *args):
        if not isinstance(tp, str):
            raise TypeError("expected a 'str' as first argument")
        if tp == "int":
            return MeshValueCollectionInt(*args)
        if tp == "uint":
            from . import common
            common.deprecation("uint-valued MeshFunction", "1.1.0", "TBA",
                               "Typename \"uint\" has been changed to \"size_t\".")
            return MeshValueCollectionSizet(*args)
        elif tp == "size_t":
            return MeshValueCollectionSizet(*args)
        elif tp == "double":
            return MeshValueCollectionDouble(*args)
        elif tp == "bool":
            return MeshValueCollectionBool(*args)
        else:
            raise RuntimeError("Cannot create a MeshValueCollection of type '%s'." % (tp,))

del _meshvaluecollection_doc_string
%}

//-----------------------------------------------------------------------------
// Extend Mesh interface with some convenient data access methods
//-----------------------------------------------------------------------------
%extend dolfin::Mesh {
%pythoncode
%{
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

def cell_orientations(self):
    """
    Get the cell orientations set.

    *Returns*
        numpy.array(int)
            Cell orientations
    """
    # Get coordinates
    orientations = self._cell_orientations()

    # Attach a reference to the Mesh to the orientations array
    _attach_base_to_numpy_array(orientations, self)

    return orientations

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
// Extend Mesh interface with some ufl_* methods
//-----------------------------------------------------------------------------
// TODO: This was intended as steps toward letting dolfin.Mesh inherit from ufl.Mesh.
//       That seems to be harder than anticipated, because fundamental properties
//       of dolfin::Mesh are mutable and undefined at construction time.
//       This work is therefore on hold. I don't think this is a showstopper,
//       it mainly means the code won't be as clean as I wanted.
// Note: The extensions to the C++ dolfin::Mesh class here are added to
//       all C++ subclasses of Mesh by swig, e.g. SubMesh, RectangleMesh, UnitSquareMesh
%extend dolfin::Mesh
{
%pythoncode
%{
def ufl_id(self):
    "Returns an id that UFL can use to decide if two objects are the same."
    return self.id()

def ufl_cell(self):
    """Returns the ufl cell of the mesh."""
    import ufl
    gdim = self.geometry().dim()
    cellname = self.type().description(False)
    return ufl.Cell(cellname, geometric_dimension=gdim)

def ufl_coordinate_element(self):
    "Return the finite element of the coordinate vector field of this domain."
    import ufl
    cell = self.ufl_cell()
    degree = self.geometry().degree()
    return ufl.VectorElement("Lagrange", cell, degree, dim=cell.geometric_dimension())

def ufl_domain(self):
    """Returns the ufl domain corresponding to the mesh."""
    import ufl
    # Cache object to avoid recreating it a lot
    if not hasattr(self, "_ufl_domain"):
        self._ufl_domain = ufl.Mesh(self.ufl_coordinate_element(), ufl_id=self.ufl_id(), cargo=self)
    return self._ufl_domain
%}
}

//-----------------------------------------------------------------------------
// Modifying the interface of Hierarchical
//-----------------------------------------------------------------------------
%pythoncode %{
HierarchicalMesh.leaf_node = new_instancemethod(_mesh.HierarchicalMesh__leaf_node,None,HierarchicalMesh)
HierarchicalMesh.root_node = new_instancemethod(_mesh.HierarchicalMesh__root_node,None,HierarchicalMesh)
HierarchicalMesh.child = new_instancemethod(_mesh.HierarchicalMesh__child,None,HierarchicalMesh)
HierarchicalMesh.parent = new_instancemethod(_mesh.HierarchicalMesh__parent,None,HierarchicalMesh)
%}

//-----------------------------------------------------------------------------
// Map __getitem__ to operator[] for MeshHierarchy
//-----------------------------------------------------------------------------
%extend dolfin::MeshHierarchy
{
  std::shared_ptr<const dolfin::Mesh> __getitem__(int i)
  {
    return (*($self))[i];
  }
}
