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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Jansson 2006-2007
// Modified by Ola Skavhaug 2006-2007
// Modified by Garth Wells 2007
// Modified by Johan Hake 2008-2009
//
// First added:  2006-09-20
// Last changed: 2011-04-04

//=============================================================================
// SWIG directives for the DOLFIN Mesh kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Macro for constructing a NumPy array from a data ponter
//-----------------------------------------------------------------------------
%define MAKE_ARRAY(dim_size, m, n, dataptr, TYPE)
        npy_intp adims[dim_size];

        adims[0] = m;
        if (dim_size == 2)
            adims[1] = n;

        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(dim_size, adims, TYPE, (char *)(dataptr)));
        if ( array == NULL ) return NULL;
%enddef

//-----------------------------------------------------------------------------
// Return NumPy arrays for Mesh::cells() and Mesh::coordinates()
//-----------------------------------------------------------------------------
%extend dolfin::Mesh {
    PyObject* coordinates() {
        int m = self->num_vertices();
        int n = self->geometry().dim();

        MAKE_ARRAY(2, m, n, self->coordinates(), NPY_DOUBLE)

        return reinterpret_cast<PyObject*>(array);
    }

    PyObject* cells() {
        int m = self->num_cells();
        int n = 0;

        if(self->topology().dim() == 1)
          n = 2;
        else if(self->topology().dim() == 2)
          n = 3;
        else
          n = 4;

        MAKE_ARRAY(2, m, n, self->cells(), NPY_INT)

        return reinterpret_cast<PyObject*>(array);
    }
}

//-----------------------------------------------------------------------------
// Return NumPy arrays for MeshFunction.values
//-----------------------------------------------------------------------------
%define ALL_VALUES(name, TYPE)
%extend name {
    PyObject* values()
    {
        dolfin::warning("MeshFunction.values() is depricated and will be removed."\
			" Use MeshFunction.array() instead.");
        int m = self->size();
        int n = 0;

        MAKE_ARRAY(1, m, n, self->values(), TYPE)

        return reinterpret_cast<PyObject*>(array);
    }

    PyObject* array()
  	{
        int m = self->size();
        int n = 0;

        MAKE_ARRAY(1, m, n, self->values(), TYPE)

        return reinterpret_cast<PyObject*>(array);
    }
}
%enddef

//-----------------------------------------------------------------------------
// Run the macros
//-----------------------------------------------------------------------------
ALL_VALUES(dolfin::MeshFunction<double>, NPY_DOUBLE)
ALL_VALUES(dolfin::MeshFunction<int>, NPY_INT)
ALL_VALUES(dolfin::MeshFunction<bool>, NPY_BOOL)
ALL_VALUES(dolfin::MeshFunction<dolfin::uint>, NPY_UINT)
ALL_VALUES(dolfin::MeshFunction<unsigned int>, NPY_UINT)

//-----------------------------------------------------------------------------
// Ignore methods that is superseded by extended versions
//-----------------------------------------------------------------------------
%ignore dolfin::Mesh::cells;
%ignore dolfin::Mesh::coordinates;
%ignore dolfin::MeshFunction::values;

//-----------------------------------------------------------------------------
// Misc ignores
//-----------------------------------------------------------------------------
%ignore dolfin::Mesh::partition(dolfin::uint num_partitions, dolfin::MeshFunction<dolfin::uint>& partitions);
%ignore dolfin::MeshEditor::open(Mesh&, CellType::Type, uint, uint);
%ignore dolfin::Point::operator=;
%ignore dolfin::Point::operator[];
%ignore dolfin::Mesh::operator=;
%ignore dolfin::MeshData::operator=;
%ignore dolfin::MeshFunction::operator=;
%ignore dolfin::MeshFunction::operator[];
%ignore dolfin::MeshGeometry::operator=;
%ignore dolfin::MeshTopology::operator=;
%ignore dolfin::MeshConnectivity::operator=;
%ignore dolfin::MeshEntityIterator::operator->;
%ignore dolfin::MeshEntityIterator::operator[];

//-----------------------------------------------------------------------------
// Map increment, decrease and dereference operators for iterators
//-----------------------------------------------------------------------------
%rename(_increment) dolfin::MeshEntityIterator::operator++;
%rename(_decrease) dolfin::MeshEntityIterator::operator--;
%rename(_dereference) dolfin::MeshEntityIterator::operator*;
%rename(_increment) dolfin::SubsetIterator::operator++;
%rename(_dereference) dolfin::SubsetIterator::operator*;

//-----------------------------------------------------------------------------
// Rename the iterators to better match the Python syntax
//-----------------------------------------------------------------------------
%rename(vertices) dolfin::VertexIterator;
%rename(edges) dolfin::EdgeIterator;
%rename(faces) dolfin::FaceIterator;
%rename(facets) dolfin::FacetIterator;
%rename(cells) dolfin::CellIterator;
%rename(entities) dolfin::MeshEntityIterator;

//-----------------------------------------------------------------------------
// Return NumPy arrays for MeshConnectivity() and MeshEntity.entities()
//-----------------------------------------------------------------------------
%ignore dolfin::MeshGeometry::x(uint n, uint i) const;
%ignore dolfin::MeshConnectivity::operator();
%ignore dolfin::MeshEntity::entities;

%extend dolfin::MeshConnectivity {
  PyObject* __call__() {
    int m = self->size();
    int n = 0;

    MAKE_ARRAY(1, m, n, (*self)(), NPY_UINT)

    return reinterpret_cast<PyObject*>(array);
  }

  PyObject* __call__(dolfin::uint entity) {
    int m = self->size(entity);
    int n = 0;

    MAKE_ARRAY(1, m, n, (*self)(entity), NPY_UINT)

    return reinterpret_cast<PyObject*>(array);
  }
}

%extend dolfin::MeshEntity {
%pythoncode
%{
    def entities(self, dim):
        """ Return number of incident mesh entities of given topological dimension"""
        return self.mesh().topology()(self.dim(), dim)(self.index())

    def __str__(self):
        """Pretty print of MeshEntity"""
        return self.str(0)
%}
}

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::SubDomain;

//-----------------------------------------------------------------------------
// Instantiate Hierarchical Mesh template classes
//-----------------------------------------------------------------------------
namespace dolfin {
  class Mesh;
  template<class T> class MeshFunction;
}

%template (HierarchicalMesh) dolfin::Hierarchical<dolfin::Mesh>;
%template (HierarchicalMeshFunctionUInt) \
    dolfin::Hierarchical<dolfin::MeshFunction<unsigned int> >;
%template (HierarchicalMeshFunctionInt) \
    dolfin::Hierarchical<dolfin::MeshFunction<int> >;
%template (HierarchicalMeshFunctionBool) \
    dolfin::Hierarchical<dolfin::MeshFunction<bool> >;
%template (HierarchicalMeshFunctionDouble) \
    dolfin::Hierarchical<dolfin::MeshFunction<double> >;
