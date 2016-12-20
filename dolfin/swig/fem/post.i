/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
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
// First added:  2007-10-22
// Last changed: 2012-11-30

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================


#ifdef HAS_PETSC
#ifdef HAS_PETSC4PY
// Override default .dm() calls.
// These are wrapped up by petsc4py typemaps so that
// we see a petsc4py object on the python side.

%feature("docstring") dolfin::PETScDMCollection::dm "Return petsc4py representation of PETSc DM";
%extend dolfin::PETScDMCollection
{
  void dm(DM& dm)
  { dm = self->dm(); }
}
#else
%extend dolfin::PETScDMCollection {
    %pythoncode %{
        def dm(self):
            common.dolfin_error("dolfin/swig/fem/post.i",
                                "access PETScDMCollection objects in Python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}
#endif
#endif

//-----------------------------------------------------------------------------
// Extend GenericDofMap
//-----------------------------------------------------------------------------
%extend dolfin::GenericDofMap
{
%pythoncode %{
    def cell_dofs(self, i):
        "Return the dofmap for a cell"
        return self._cell_dofs(i)
%}
}


//-----------------------------------------------------------------------------
// Extend Function so f.function_space() return a dolfin.FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::DirichletBC
{
%pythoncode %{
def function_space(self):
    "Return the FunctionSpace"
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(self._function_space())
%}
}

//-----------------------------------------------------------------------------
// Extend Function so f.function_space() return a dolfin.FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::Form {
%pythoncode %{
def function_space(self, i):
    """
    Return function space for given argument

     *Arguments*
         i (std::size_t)
             Index

     *Returns*
         _FunctionSpace_
             Function space shared pointer.
    """
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(self._function_space(i))
%}
}

//-----------------------------------------------------------------------------
// Extend FiniteElement.tabulate_dof_coordinates()
//-----------------------------------------------------------------------------
%extend dolfin::FiniteElement {
  void _tabulate_dof_coordinates(PyObject* coordinates, const Cell& cell)
  {
    // NOTE: No NumPy array check. Assuming everything is correct!

    // Get NumPy array
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>(coordinates);

    // Initialize the boost::multi_array structure
    boost::multi_array<double, 2> tmparray;

    // Get cell vertex coordinates
    std::vector<double> coordinate_dofs;
    cell.get_coordinate_dofs(coordinate_dofs);

    // Tabulate the coordinates
    self->tabulate_dof_coordinates(tmparray, coordinate_dofs, cell);

    // Get geometric dimension
    std::size_t gdim = tmparray.shape()[1];

    // Copy data
    double* data = static_cast<double*>(PyArray_DATA(xa));
    for (std::size_t i = 0; i < self->space_dimension(); i++)
      for (std::size_t j = 0; j < gdim; j++)
        data[i*gdim + j] = tmparray[i][j];
  }

%pythoncode %{
def tabulate_dof_coordinates(self, cell, coordinates=None):
    """ Tabulate the coordinates of all dofs on a cell

    *Arguments*
        cell (_Cell_)
             The cell.
        coordinates (NumPy array)
             Optional argument: The coordinates of all dofs on a cell.
    *Returns*
        coordinates
             The coordinates of all dofs on a cell.
    """
    import numpy as np

    # Check coordinate argument
    gdim = self.geometric_dimension()
    shape = (self.space_dimension(), gdim)
    if coordinates is None:
        coordinates = np.zeros(shape, 'd')
    if not isinstance(coordinates, np.ndarray) or \
       not (coordinates.flags.c_contiguous and \
            coordinates.dtype == np.dtype('d') and \
            coordinates.shape==shape):
        raise TypeError("expected a C-contiguous numpy array " \
              "of 'double' (dtype='d') with shape %s"%str(shape))

    # Call the extended method
    self._tabulate_dof_coordinates(coordinates, cell)
    return coordinates
%}
}

//-----------------------------------------------------------------------------
// Modifying the interface of FooProblem
//-----------------------------------------------------------------------------
%define PROBLEM_EXTENDS(TYPE)
%extend dolfin::TYPE
{
%pythoncode %{
def solution(self):
    """
    Return the solution
    """
    from dolfin.functions.function import Function
    return Function(self._solution())

def trial_space(self):
    """
    Return the trial space
    """
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(self._trial_space())

def test_space(self):
    """
    Return the test space
    """
    from dolfin.functions.functionspace import FunctionSpace
    return FunctionSpace(self._test_space())

%}
}
%enddef

//-----------------------------------------------------------------------------
// Modifying the interface of Hierarchical
//-----------------------------------------------------------------------------
%define HIERARCHICAL_FEM_EXTENDS(TYPE)
%pythoncode %{
Hierarchical ## TYPE.leaf_node = Hierarchical ## TYPE._leaf_node
Hierarchical ## TYPE.root_node = Hierarchical ## TYPE._root_node
Hierarchical ## TYPE.child = Hierarchical ## TYPE._child
Hierarchical ## TYPE.parent = Hierarchical ## TYPE._parent
%}
%enddef


PROBLEM_EXTENDS(LinearVariationalProblem)
PROBLEM_EXTENDS(NonlinearVariationalProblem)
HIERARCHICAL_FEM_EXTENDS(LinearVariationalProblem)
HIERARCHICAL_FEM_EXTENDS(NonlinearVariationalProblem)
HIERARCHICAL_FEM_EXTENDS(Form)
HIERARCHICAL_FEM_EXTENDS(DirichletBC)
