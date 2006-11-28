// --- Return NumPy arrays for Mesh::cells() and Mesh::coordinates() ---

%define ALL_COORDINATES(name)
%extend name {
    PyObject* coordinates() {
        // Get coordinates for all vertices in structure.
        // returns a 3xnoVertices Numeric array, x in array[0], y in array[1]
        // and z in array[2]
        int noVert = self->numVertices();
        int dim = self->geometry().dim();
        int nadims = 2;
        npy_intp adims[nadims];

        adims[0] = noVert;
        adims[1] = dim;

        PyArrayObject* arr = (PyArrayObject *)PyArray_SimpleNewFromData(nadims, adims, PyArray_DOUBLE, (char *)(self->coordinates()));
        if ( arr == NULL ) return NULL;
        PyArray_INCREF(arr);
        return (PyObject *)arr;
    }
// End brace extend:
}
%enddef

%define ALL_CELLS(name)
%extend name {
    PyObject* cells() {
       // Get the node-id for all vertices.
        int  nadims = 2;
        npy_intp adims[nadims];
        int no_cells = self->numCells();
        int vertices_per_cell = (self->topology().dim() == 2) ? 3 : 4;

        adims[0] = no_cells;
        adims[1] = vertices_per_cell;

        // the std. way of creating a Numeric array
        PyArrayObject* arr = (PyArrayObject *)PyArray_SimpleNewFromData(nadims, adims, PyArray_INT, (char *)(self->cells()));
        if (arr == NULL) return NULL;
        PyArray_INCREF(arr);
        return (PyObject *)arr;
    }
}
%enddef

ALL_COORDINATES(dolfin::Mesh)
ALL_CELLS(dolfin::Mesh)

%ignore dolfin::Mesh::cells;
%ignore dolfin::Mesh::coordinates;

//--- Mesh iterators ---

// Map increment operator and dereference operators for iterators
%rename(increment) dolfin::MeshEntityIterator::operator++;
%rename(dereference) dolfin::MeshEntityIterator::operator*;

// Rename the iterators to better match the Python syntax
%rename(vertices) dolfin::VertexIterator;
%rename(edges) dolfin::EdgeIterator;
%rename(faces) dolfin::FaceIterator;
%rename(facets) dolfin::FacetIterator;
%rename(cells) dolfin::CellIterator;
%rename(entities) dolfin::MeshEntityIterator;

//--- DOLFIN mesh interface ---

%include "dolfin/BoundaryComputation.h"
%include "dolfin/BoundaryMesh.h"
%include "dolfin/Cell.h"
%include "dolfin/CellType.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/Facet.h"
%include "dolfin/Interval.h"
%include "dolfin/MeshConnectivity.h"
%include "dolfin/MeshData.h"
%include "dolfin/MeshEditor.h"
%include "dolfin/MeshEntity.h"
%include "dolfin/MeshEntityIterator.h"
%include "dolfin/MeshFunction.h"
%include "dolfin/MeshGeometry.h"
%include "dolfin/Mesh.h"
%include "dolfin/MeshTopology.h"
%include "dolfin/Point.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/Triangle.h"
%include "dolfin/UnitCube.h"
%include "dolfin/UnitSquare.h"
%include "dolfin/Vertex.h"

// Extend mesh entity iterators to work as Python iterators
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

//--- Map MeshFunction template to Python ---

%template(MeshFunctionInt) dolfin::MeshFunction<int>;
%template(MeshFunctionFloat) dolfin::MeshFunction<double>;
%template(MeshFunctionBool) dolfin::MeshFunction<bool>;

%pythoncode
%{
class MeshFunction(object):
    def __new__(self, tp):
        if tp == int:
            return MeshFunctionInt()
        elif tp == float:
            return MeshFunctionFloat()
        elif tp == bool:
            return MeshFunctionBool()
        else:
            raise RuntimeError, "Cannot create a MeshFunction of %s" % (tp,)

MeshFunctionInt.__call__   = MeshFunctionInt.get
MeshFunctionFloat.__call__ = MeshFunctionFloat.get
MeshFunctionBool.__call__  = MeshFunctionBool.get

%}

//--- Extend Point interface with Python selectors ---
%extend dolfin::Point {
  real get(int i) { return (*self)[i]; }
  void set(int i, real val) { (*self)[i] = val; }
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
