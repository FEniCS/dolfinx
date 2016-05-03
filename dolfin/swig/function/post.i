/* -*- C -*- */
// Copyright (C) 2007-2009 Johan Hake
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
// First added:  2008-11-02
// Last changed: 2012-11-30

//-----------------------------------------------------------------------------
// Extend FunctionAXPY
//-----------------------------------------------------------------------------
%extend dolfin::FunctionAXPY
{
%pythoncode %{
__truediv__ = __div__
%}
}

//-----------------------------------------------------------------------------
// Extend FunctionSpace so one can check if a Function is in a FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::FunctionSpace {
%pythoncode %{
def __contains__(self,u):
    "Check whether a function is in the FunctionSpace"
    if isinstance(u, Function):
        return u._in(self)
    return False

def leaf_node(self):
    "Return the finest FunctionSpace in hierarchy"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(HierarchicalFunctionSpace._leaf_node(self))

def root_node(self):
    "Return the coarsest FunctionSpace in hierarchy"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(HierarchicalFunctionSpace._root_node(self))

def child(self):
    "Return the child FunctionSpace in the hierarchy"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(HierarchicalFunctionSpace._child(self))

def parent(self):
    "Return the parent FunctionSpace in the hierarchy"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(HierarchicalFunctionSpace._parent(self))

%}
}

//-----------------------------------------------------------------------------
// Extend GenericFunction interace
//-----------------------------------------------------------------------------
/*
%extend dolfin::GenericFunction {
%pythoncode %{
def compute_vertex_values(self, mesh=None):
    """
    Compute values at all mesh vertices

    *Arguments*
        vertex_values (_Array_ <double>)
            The values at all vertices.
        mesh (_Mesh_)
            The mesh.
    """

    # Check that we get a mesh if we need it
    if mesh is None:
        if not isinstance(self, Function):
            common.dolfin_error("dolfin/swig/function/post.i",
                                "compute vertex values",
                                "You need to supply a mesh");
        elif:
            mesh = self.function_space().mesh()

    # Argument checks
    #from numpy import ndarray, zeros
    #if not isinstance(values, ndarray) or len(values.shape) != 1:
    #    common.dolfin_error("function/post.i",
		#	    "compute values at the vertices",
		#	    "Expected a numpy array with dimension 1 as first argument")

    #value_size = self.value_size()*mesh.num_vertices()
    #if len(values) != value_size:
    #    common.dolfin_error("function_post.i",
		#	    "compute values at the vertices",
		#	    "The provided array need to be of size value_size()*mesh.num_vertices()")

    # Call the actual method
    #self._compute_vertex_values(values, mesh)

    return self._compute_vertex_values(mesh)
%}
}
*/

//-----------------------------------------------------------------------------
// Extend Function interace
//-----------------------------------------------------------------------------
%extend dolfin::Function {
%pythoncode %{
def function_space(self):
    "Return the FunctionSpace"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(self._function_space())

def copy(self, deepcopy=False):
    """
    Return a copy of itself

    *Arguments*
        deepcopy (bool)
            If false (default) the dof vector is shared.

    *Returns*
         _Function_
             The Function

    """
    from dolfin.functions.function import Function
    if deepcopy:
        return Function(self.function_space(), self.vector().copy())
    return Function(self.function_space(), self.vector())

def leaf_node(self):
    "Return the finest Function in hierarchy"
    from dolfin.functions.function import Function
    return Function(HierarchicalFunction._leaf_node(self))

def root_node(self):
    "Return the coarsest Function in hierarchy"
    from dolfin.functions.function import Function
    return Function(HierarchicalFunction._root_node(self))

def child(self):
    "Return the child Function in the hierarchy"
    from dolfin.functions.function import Function
    return Function(HierarchicalFunction._child(self))

def parent(self):
    "Return the parent Function in the hierarchy"
    from dolfin.functions.function import Function
    return Function(HierarchicalFunction._parent(self))
%}
}
