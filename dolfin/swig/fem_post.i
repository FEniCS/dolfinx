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
// Last changed: 2011-08-18

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Extend Function so f.function_space() return a dolfin.FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::BoundaryCondition {
%pythoncode %{
def function_space(self):
    "Return the FunctionSpace"
    from dolfin.functions.functionspace import FunctionSpaceFromCpp
    return FunctionSpaceFromCpp(self._function_space())
%}
}

//-----------------------------------------------------------------------------
// Extend GenericDofMap.tabulate_coordinates()
//-----------------------------------------------------------------------------
%extend dolfin::GenericDofMap {
  void _tabulate_coordinates(PyObject* coordinates, const Cell& cell)
  {
    // NOTE: No NumPy array check. Assumed everything is coorect!
        
    // Get NumPy array
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>(coordinates);

    // Initialize the boost::multi_array structure
    boost::multi_array<double, 2>::extent_gen extents;
    boost::multi_array<double, 2> tmparray;
    tmparray.resize(extents[self->cell_dimension(cell.index())]\
		    [self->geometric_dimension()]);
    
    // Tabulate the coordinates
    dolfin::UFCCell ufc_cell(cell);
    self->tabulate_coordinates(tmparray, ufc_cell);
    
    // Copy data
    double* data = static_cast<double*>(PyArray_DATA(xa));
    for (uint i=0; i<self->cell_dimension(cell.index()); i++)
      for (uint j=0; j<self->geometric_dimension(); j++)
	data[i*self->geometric_dimension()+j] = tmparray[i][j];
  }

%pythoncode %{
def tabulate_coordinates(self, cell, coordinates=None):
    """ Tabulate the coordinates of the dofs in a cell"""
    import numpy as np

    # Check coordinate argument
    shape = (self.max_cell_dimension(), self.geometric_dimension())
    if coordinates is None:
        coordinates = np.zeros(shape, 'd')
    if not isinstance(coordinates, np.ndarray) or \
       not (coordinates.flags.c_contiguous and \
            coordinates.dtype == np.dtype('d') and \
            coordinates.shape==shape):
        raise TypeError, "expected a C-contiguous numpy array " \
              "of 'double' (dtype='d') with shape %s"%str(shape)

    # Call the extended method
    self._tabulate_coordinates(coordinates, cell)
    return coordinates
%}
}
