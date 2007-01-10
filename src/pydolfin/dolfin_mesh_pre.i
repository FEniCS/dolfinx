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
