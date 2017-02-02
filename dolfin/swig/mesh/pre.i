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
// Modified by Garth Wells 2007
// Modified by Johan Hake 2008-2011
//
// First added:  2006-09-20
// Last changed: 2013-05-22

//=============================================================================
// SWIG directives for the DOLFIN Mesh kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

%ignore dolfin::LocalMeshData::Geometry;
%ignore dolfin::LocalMeshData::Topology;

//-----------------------------------------------------------------------------
// Return NumPy arrays for Mesh::cells() and Mesh::coordinates()
//-----------------------------------------------------------------------------
%extend dolfin::Mesh {
  PyObject* _coordinates() {
    return %make_numpy_array(2, double)(self->geometry().num_points(),
                                        self->geometry().dim(),
                                        self->coordinates().data(), true);
  }

  PyObject* _cells() {
    // FIXME: Works only for Mesh with Intervals, Triangles and Tetrahedrons
    return %make_numpy_array(2, uint)(self->num_cells(),
                                      self->type().num_entities(0),
                                      self->cells().data(), false);
  }

  PyObject* _cell_orientations()
  {
    if (!self->cell_orientations().empty())
      dolfin_assert(self->cell_orientations().size() == self->num_cells());
    return %make_numpy_array(1, int)(self->cell_orientations().size(),
                                     self->cell_orientations().data(), true);
  }
}

//-----------------------------------------------------------------------------
// Return NumPy arrays for MeshFunction.values
//-----------------------------------------------------------------------------
%define ALL_VALUES(name, TYPE_NAME)
%extend name {
PyObject* _array()
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
ALL_VALUES(dolfin::MeshFunction<std::size_t>, size_t)

//-----------------------------------------------------------------------------
// Ignore methods that is superseded by extended versions
//-----------------------------------------------------------------------------
%ignore dolfin::Mesh::cells;
%ignore dolfin::Mesh::coordinates;
%ignore dolfin::Mesh::cell_orientations;
%ignore dolfin::MeshFunction::values;
%warnfilter(503) dolfin::create_mesh;

//-----------------------------------------------------------------------------
// Rename methods which get called by a re-implemented method from the
// Python layer
//-----------------------------------------------------------------------------
%rename (_mark) dolfin::SubDomain::mark;

//-----------------------------------------------------------------------------
// Misc ignores
//-----------------------------------------------------------------------------
%ignore dolfin::MeshEditor::open(Mesh&, CellType::Type, std::size_t, std::size_t);
%ignore dolfin::Mesh::operator=;
%ignore dolfin::MeshData::operator=;
%ignore dolfin::MeshFunction::operator=;
%ignore dolfin::MeshFunction::operator[];
%ignore dolfin::MeshValueCollection::operator=;
%ignore dolfin::MeshGeometry::operator=;
%ignore dolfin::MeshTopology::operator=;
%ignore dolfin::MeshTopology::shared_entities(unsigned int) const;
%ignore dolfin::MeshValueCollection::operator=;
%ignore dolfin::MeshConnectivity::operator=;
%ignore dolfin::MeshConnectivity::set;
%ignore dolfin::MeshEntityIterator::operator->;
%ignore dolfin::MeshEntityIterator::operator[];
%ignore dolfin::MeshEntity::operator->;
%ignore dolfin::SubsetIterator::operator->;
%ignore dolfin::SubsetIterator::operator[];
%ignore dolfin::MeshDomains::operator=;
%ignore dolfin::MeshDomains::markers(std::size_t) const;
%ignore dolfin::MeshData::array(std::string) const;
%ignore dolfin::MeshHierarchy::operator[];

//-----------------------------------------------------------------------------
// Map increment, decrease and dereference operators for iterators
//-----------------------------------------------------------------------------
%rename(_increment) dolfin::MeshEntityIterator::operator++;
%rename(_decrease) dolfin::MeshEntityIterator::operator--;
%rename(_dereference) dolfin::MeshEntityIterator::operator*;
%rename(_increment) dolfin::SubsetIterator::operator++;
%rename(_dereference) dolfin::SubsetIterator::operator*;

//-----------------------------------------------------------------------------
// MeshEntityIteratorBase
//-----------------------------------------------------------------------------

// Ignore for all specializations done before importing the type
%ignore dolfin::MeshEntityIteratorBase::operator=;
%ignore dolfin::MeshEntityIteratorBase::operator->;
%ignore dolfin::MeshEntityIteratorBase::operator[];
%ignore dolfin::MeshEntityIteratorBase::operator[];
%ignore dolfin::MeshEntityIteratorBase::operator++;
%ignore dolfin::MeshEntityIteratorBase::operator--;
%ignore dolfin::MeshEntityIteratorBase::operator*;

#ifdef MESHMODULE // Conditional statements only for MESH module

// Forward import base template type
%import"dolfin/mesh/MeshEntityIteratorBase.h"

%define MESHENTITYITERATORBASE(ENTITY, name)
%template(name) dolfin::MeshEntityIteratorBase<dolfin::ENTITY>;

// Extend the interface (instead of renaming, doesn't seem to work)
%extend dolfin::MeshEntityIteratorBase<dolfin::ENTITY>
{
  dolfin::MeshEntityIteratorBase<dolfin::ENTITY>& _increment()
  { return self->operator++(); }

  dolfin::MeshEntityIteratorBase<dolfin::ENTITY>& _decrease()
  { return self->operator--(); }

  dolfin::ENTITY _dereference()
  { return *self->operator->(); }

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

%enddef

MESHENTITYITERATORBASE(Cell, cells)
MESHENTITYITERATORBASE(Edge, edges)
MESHENTITYITERATORBASE(Face, faces)
MESHENTITYITERATORBASE(Facet, facets)
MESHENTITYITERATORBASE(Vertex, vertices)

//-----------------------------------------------------------------------------
// Rename the iterators to better match the Python syntax
//-----------------------------------------------------------------------------
%rename(entities) dolfin::MeshEntityIterator;

//-----------------------------------------------------------------------------
// Return NumPy arrays for MeshConnectivity() and MeshEntity.entities()
//-----------------------------------------------------------------------------
%ignore dolfin::MeshGeometry::x(std::size_t n, std::size_t i) const;
%ignore dolfin::MeshConnectivity::operator();
%ignore dolfin::MeshEntity::entities;

%extend dolfin::MeshConnectivity
{
  PyObject* __call__()
  { return %make_numpy_array(1, uint)(self->size(), (*self)().data(), false); }

  PyObject* __call__(std::size_t entity)
  {
    return %make_numpy_array(1, uint)(self->size(entity), (*self)(entity),
                                      false);
  }
}

%extend dolfin::MeshEntity
{
%pythoncode
%{
    def entities(self, dim):
        """ Return number of incident mesh entities of given topological dimension"""
        return self.mesh().topology()(self.dim(), dim)(self.index())

    def __str__(self):
        """Pretty print of MeshEntity"""
        return self.str(False)
%}
}

// Exclude from ifdef as it is used by other modules
%define FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(TYPE, TYPENAME)

%shared_ptr(dolfin::Hierarchical<dolfin::MeshFunction<TYPE> >)
%template (HierarchicalMeshFunction ## TYPENAME) \
    dolfin::Hierarchical<dolfin::MeshFunction<TYPE> >;

%enddef

FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(unsigned int, UInt)
FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(int, Int)
FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(double, Double)
FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(bool, Bool)
FORWARD_DECLARE_HIERARCHICAL_MESHFUNCTIONS(std::size_t, Sizet)

#endif // End ifdef for MESHMODULE

%define FORWARD_DECLARE_MESHFUNCTIONS(TYPE, TYPENAME)

// Forward declaration of template
%template() dolfin::MeshFunction<TYPE>;

// Shared_ptr declarations
%shared_ptr(dolfin::MeshFunction<TYPE>)
%shared_ptr(dolfin::CellFunction<TYPE>)
%shared_ptr(dolfin::EdgeFunction<TYPE>)
%shared_ptr(dolfin::FaceFunction<TYPE>)
%shared_ptr(dolfin::FacetFunction<TYPE>)
%shared_ptr(dolfin::VertexFunction<TYPE>)

// Include shared_ptr declaration of MeshValueCollection
%shared_ptr(dolfin::MeshValueCollection<TYPE>)

%enddef

FORWARD_DECLARE_MESHFUNCTIONS(unsigned int, UInt)
FORWARD_DECLARE_MESHFUNCTIONS(int, Int)
FORWARD_DECLARE_MESHFUNCTIONS(double, Double)
FORWARD_DECLARE_MESHFUNCTIONS(bool, Bool)
FORWARD_DECLARE_MESHFUNCTIONS(std::size_t, Sizet)

%template (HierarchicalMesh) dolfin::Hierarchical<dolfin::Mesh>;

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::SubDomain;

//-----------------------------------------------------------------------------
// Ignore all of MeshPartitioning except
// void build_distributed_mesh(Mesh&);
//-----------------------------------------------------------------------------
%ignore dolfin::MeshPartitioning::build_distributed_mesh(Mesh&, const std::vector<std::size_t>&);
%ignore dolfin::MeshPartitioning::build_distributed_mesh(Mesh&, const LocalMeshData&);
%ignore dolfin::MeshPartitioning::build_distributed_value_collection;

//-----------------------------------------------------------------------------
// Ignores for MultiMesh
//-----------------------------------------------------------------------------
%ignore dolfin::plot(const MultiMesh&);
%ignore dolfin::plot(std::shared_ptr<const MultiMesh>);

//-----------------------------------------------------------------------------
// Add typemap functions for MultiMesh quadrature rules
//-----------------------------------------------------------------------------
%fragment("convert_dolfin_quadrature_rule", "header"){ 
SWIGINTERNINLINE PyObject * convert_dolfin_quadrature_rule(dolfin::quadrature_rule qr)
{
  // Typemap Function for dolfin::quadrature_rule 
  npy_intp n0 = qr.first.size();
  npy_intp n1 = qr.second.size();

  // return None if there are no quadrature points
  if (n0 == 0)
    return Py_None;  

  PyArrayObject *x0 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n0, NPY_DOUBLE));
  PyArrayObject *x1 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n1, NPY_DOUBLE));

  double* data0 = static_cast<double*>(PyArray_DATA(x0));
  double* data1 = static_cast<double*>(PyArray_DATA(x1));

  std::copy(qr.first.begin(),  qr.first.end(),  data0);
  std::copy(qr.second.begin(), qr.second.end(), data1);

  PyObject * result;
  result = Py_BuildValue("OO", x0, x1);
  return result;
}
}
%fragment("convert_dolfin_quadrature_rule_vector", "header"){
SWIGINTERNINLINE PyObject * convert_dolfin_quadrature_rule(std::vector<dolfin::quadrature_rule> qr_vector)
{
  // Typemap function for std::vec<dolfin::quadrature_rule>
  dolfin::quadrature_rule qr;
  for (dolfin::quadrature_rule qr_j : qr_vector)
  {
    qr.first.insert(qr.first.end(), qr_j.first.begin(), qr_j.first.end());
    qr.second.insert(qr.second.end(), qr_j.second.begin(), qr_j.second.end());
  }
  return convert_dolfin_quadrature_rule(qr); 
}
}
// Force fragments to be instantiated
%fragment("convert_dolfin_quadrature_rule");
%fragment("convert_dolfin_quadrature_rule_vector");
//-----------------------------------------------------------------------------
// Modifying MultiMesh interface
//-----------------------------------------------------------------------------
%define EXTEND_MULTIMESH_QUADRATURE_RULE(cell_type)

%extend dolfin::MultiMesh
{
PyObject* quadrature_rule_##cell_type(std::size_t part)
{
  PyObject* ret = PyDict_New();
    
  auto qr_map = ($self)->quadrature_rule_##cell_type (part);
  
  for (auto it = qr_map.begin(); it != qr_map.end(); it++)
  {
    PyObject* key = SWIG_From_dec(unsigned int)(it->first);
    PyObject* val  = convert_dolfin_quadrature_rule(it->second);
    if (val != Py_None)
      PyDict_SetItem(ret, key,  val);
    Py_XDECREF(key);
    Py_XDECREF(val);
  }
  return ret;
}
PyObject* quadrature_rule_##cell_type(std::size_t part, unsigned int cell)
{
  auto qr_map = ($self)->quadrature_rule_##cell_type(part); 
  auto qr = qr_map[cell];
  return convert_dolfin_quadrature_rule(qr);
}
}
%ignore dolfin::MultiMesh::quadrature_rule_##cell_type;
dolfin::MultiMesh::quadrature_rule_##cell_type(std::size_t part);


%enddef

EXTEND_MULTIMESH_QUADRATURE_RULE(cut_cells)
EXTEND_MULTIMESH_QUADRATURE_RULE(interface)
EXTEND_MULTIMESH_QUADRATURE_RULE(overlap)

