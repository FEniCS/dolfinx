// Return Numeric arrays for Mesh::cells() and Mesh::vertices().
// This is used in the PyCC mesh interface.
%extend dolfin::NewMesh {

PyObject * cells() {
    PyArrayObject *arr;
    int n[2];
    n[0] = self->numCells();
    n[1] = self->type().numVertices(self->dim());
    arr = (PyArrayObject *) PyArray_FromDimsAndData(2, n, PyArray_INT, (char *) self->cells());
    arr->flags |= OWN_DATA;
    Py_INCREF((PyObject *)arr);
    return (PyObject *)arr;
}

PyObject * vertices() {
    PyArrayObject *arr;
    int n[2];
    n[0] = self->numVertices();
    n[1] = self->geometry().dim();
    arr = (PyArrayObject *) PyArray_FromDimsAndData(2, n, PyArray_DOUBLE, (char *) self->vertices());
    arr->flags |= OWN_DATA;
    Py_INCREF((PyObject *)arr);
    return (PyObject *)arr;
}

}

%ignore dolfin::NewMesh::cells;
%ignore dolfin::NewMesh::vertices;

// Map increment operator and dereference operators for iterators
%rename(increment) dolfin::MeshEntityIterator::operator++;
%rename(dereference) dolfin::MeshEntityIterator::operator*;

// Extend iterator to work as a Python iterator
%extend dolfin::MeshEntityIterator {
%pythoncode
%{
def __iter__(self):
  self.first = True
  return self

def next(self):
  if self.end():
    raise StopIteration
  if not self.first:  
    self.increment()
  self.first = False
  return self.dereference()
%}
}

%rename(increment) dolfin::VertexIterator::operator++;
%rename(increment) dolfin::CellIterator::operator++;
%rename(increment) dolfin::EdgeIterator::operator++;

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


