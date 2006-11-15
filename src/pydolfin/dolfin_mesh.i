// Return Numeric arrays for Mesh::cells() and Mesh::coordinates().
// This is used in the PyCC mesh interface.
%extend dolfin::Mesh {

PyObject * cells() {
    PyArrayObject *arr;
    int n[2];
    n[0] = self->numCells();
    n[1] = self->type().numVertices(self->topology().dim());
    arr = (PyArrayObject *) PyArray_FromDimsAndData(2, n, PyArray_INT, (char *) self->cells());
    arr->flags |= OWN_DATA;
    Py_INCREF((PyObject *)arr);
    return (PyObject *)arr;
}

PyObject * coordinates() {
    PyArrayObject *arr;
    int n[2];
    n[0] = self->numVertices();
    n[1] = self->geometry().dim();
    arr = (PyArrayObject *) PyArray_FromDimsAndData(2, n, PyArray_DOUBLE, (char *) self->coordinates());
    arr->flags |= OWN_DATA;
    Py_INCREF((PyObject *)arr);
    return (PyObject *)arr;
}

}

%ignore dolfin::Mesh::cells;
%ignore dolfin::Mesh::coordinates;

// Map increment operator and dereference operators for iterators
%rename(increment) dolfin::MeshEntityIterator::operator++;
%rename(dereference) dolfin::MeshEntityIterator::operator*;

// Extend mesh entity iterators to work as a Python iterators
%extend dolfin::MeshEntityIterator {
%pythoncode
%{
def __iter__(self):
  self.first = True
  return self

def next(self):
  if not self.first:
    self.increment()
  if self.end():
    raise StopIteration
  self.first = False
  return self.dereference()
%}
}

// Rename the iterators to better match the Python syntax
%rename(vertices) dolfin::VertexIterator;
%rename(edges) dolfin::EdgeIterator;
%rename(faces) dolfin::FaceIterator;
%rename(facets) dolfin::FacetIterator;
%rename(cells) dolfin::CellIterator;
%rename(entities) dolfin::MeshEntityIterator;

%include "dolfin/MeshConnectivity.h"
%include "dolfin/MeshEditor.h"
%include "dolfin/MeshEntity.h"
%include "dolfin/MeshEntityIterator.h"
%include "dolfin/MeshGeometry.h"
%include "dolfin/MeshTopology.h"
%include "dolfin/Mesh.h"
%include "dolfin/MeshData.h"
%include "dolfin/MeshFunction.h"
%include "dolfin/Vertex.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/Facet.h"
%include "dolfin/Cell.h"
%include "dolfin/TopologyComputation.h"
%include "dolfin/CellType.h"
%include "dolfin/Interval.h"
%include "dolfin/Triangle.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/UniformMeshRefinement.h"
%include "dolfin/Point.h"
%include "dolfin/BoundaryComputation.h"
%include "dolfin/BoundaryMesh.h"
%include "dolfin/UnitCube.h"
%include "dolfin/UnitSquare.h"

// Extend Point interface with Python selectors
%extend dolfin::Point {
  real get(int i)
  {
    return (*self)[i];
  }

  void set(int i, real val)
  {
    (*self)[i] = val;
  }
}

%pythoncode
%{
  def __getitem__(self, i):
      return self.get(i)
  def __setitem__(self, i, val):
      self.set(i, val)

  Point.__getitem__ = __getitem__
  Point.__setitem__ = __setitem__
%}
