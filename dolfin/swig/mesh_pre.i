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
// Return NumPy arrays for Mesh::cells() and Mesh::coordinates()
//-----------------------------------------------------------------------------
%extend dolfin::Mesh {
    PyObject* coordinates() {
      return %make_numpy_array(2, double)(self->num_vertices(),
					  self->geometry().dim(),
					  self->coordinates(), true);
    }
    
    PyObject* cells() {
      int n = 4;
      
      if(self->topology().dim() == 1)
	n = 2;
      else if(self->topology().dim() == 2)
	n = 3;
      
      return %make_numpy_array(2, uint)(self->num_cells(),
					n, self->cells(), false);
    }
}

//-----------------------------------------------------------------------------
// Return NumPy arrays for MeshFunction.values
//-----------------------------------------------------------------------------
%define ALL_VALUES(name, TYPE_NAME)
%extend name {
PyObject* values()
{
  dolfin::warning("MeshFunction.values() is depricated and will be removed." \
		  " Use MeshFunction.array() instead.");
  return %make_numpy_array(1, TYPE_NAME)(self->size(), self->values(), true);
}

PyObject* array()
{
  return %make_numpy_array(1, TYPE_NAME)(self->size(), self->values(), true);
}
}
%enddef

//-----------------------------------------------------------------------------
// Run the macros
//-----------------------------------------------------------------------------
ALL_VALUES(dolfin::MeshFunction<double>, double)
ALL_VALUES(dolfin::MeshFunction<int>, int)
ALL_VALUES(dolfin::MeshFunction<bool>, bool)
ALL_VALUES(dolfin::MeshFunction<dolfin::uint>, uint)
ALL_VALUES(dolfin::MeshFunction<unsigned int>, uint)

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
    return %make_numpy_array(1, uint)(self->size(), (*self)(), false);
  }

  PyObject* __call__(dolfin::uint entity) {
    return %make_numpy_array(1, uint)(self->size(entity), (*self)(entity), false);
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
