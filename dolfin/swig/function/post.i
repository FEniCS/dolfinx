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
// Last changed: 2012-01-30

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
// Extend GenericFunction interace
//-----------------------------------------------------------------------------
%extend dolfin::GenericFunction {
%pythoncode %{
def compute_vertex_values(self, values, mesh):
    """
    Compute values at all mesh vertices
    
    *Arguments*
        vertex_values (_Array_ <double>)
            The values at all vertices.
        mesh (_Mesh_)
            The mesh.
    """
    # Argument checks
    from numpy import ndarray
    if not isinstance(values, ndarray) or len(values.shape) != 1:
        common.dolfin_error("function/post.i",
			    "compute values at the vertices",
			    "Expected a numpy array with dimension 1 as first argument")
    
    value_size = self.value_size()*mesh.num_vertices()
    if len(values) != value_size:
        common.dolfin_error("function_post.i",
			    "compute values at the vertices",
			    "The provided array need to be of size value_size()*mesh.num_vertices()")

    # Call the actuall method
    self._compute_vertex_values(values, mesh)
%}
}

//-----------------------------------------------------------------------------
// Extend Function interace
//-----------------------------------------------------------------------------
%extend dolfin::Function {
%pythoncode %{
def function_space(self):
    "Return the FunctionSpace"
    from dolfin.functions.functionspace import FunctionSpaceFromCpp
    return FunctionSpaceFromCpp(self._function_space())

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

#  -----------------------------------------------------------------------------
#  f.leaf_node()/root_node() returns a dolfin.Function.
#  Not doing this on purpose for child()/parent().
#  -----------------------------------------------------------------------------
def leaf_node(self):
    "Return the finest Function in hierarchy"
    f = HierarchicalFunction.leaf_node(self)
    return f.copy()

def root_node(self):
    "Return the coarsest Function in hierarchy"
    f = HierarchicalFunction.root_node(self)
    return f.copy()
%}
}
