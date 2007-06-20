// --- Return NumPy arrays for Mesh::cells() and Mesh::coordinates() ---

%define MAKE_ARRAY(dim_size, m, n, dataptr, TYPE)
        npy_intp adims[dim_size];

        adims[0] = m;
        if (dim_size == 2)
            adims[1] = n;

        PyArrayObject* array = (PyArrayObject *)PyArray_SimpleNewFromData(dim_size, adims, TYPE, (char *)(dataptr));
        if ( array == NULL ) return NULL;
        PyArray_INCREF(array);
%enddef


%define ALL_COORDINATES(name)
%extend name {
    PyObject* coordinates() {
        // Get coordinates for all vertices in structure.
        // returns a 3xnumVertices numpy array, x in array[0], y in array[1]
        // and z in array[2]
        int m = self->numVertices();
        int n = self->geometry().dim();

        MAKE_ARRAY(2, m, n, self->coordinates(), NPY_DOUBLE)

        return (PyObject *)array;
    }
}
%enddef

%define ALL_CELLS(name)
%extend name {
    PyObject* cells() {
       // Get the node-id for all vertices.
        int m = self->numCells();
        int n = 0;
        if(self->topology().dim() == 1)
          n = 2;
        else if(self->topology().dim() == 2)
          n = 3;
        else
          n = 4;

        MAKE_ARRAY(2, m, n, self->cells(), NPY_INT)

        return (PyObject *)array;
    }
}
%enddef

%define ALL_VALUES(name, TYPE)
%extend name {
    PyObject* values() {
        int m = self->size();
        int n = 0;

        MAKE_ARRAY(1, m, n, self->values(), TYPE)

        return (PyObject *)array;
    }
}
%enddef

ALL_COORDINATES(dolfin::Mesh)
ALL_CELLS(dolfin::Mesh)
ALL_VALUES(dolfin::MeshFunction<real>, NPY_DOUBLE)
ALL_VALUES(dolfin::MeshFunction<int>, NPY_INT)
ALL_VALUES(dolfin::MeshFunction<bool>, NPY_BOOL)
ALL_VALUES(dolfin::MeshFunction<unsigned int>, NPY_UINT)

%ignore dolfin::Mesh::cells;
%ignore dolfin::Mesh::coordinates;
%ignore dolfin::MeshFunction::values;
%ignore dolfin::MeshEditor::open(Mesh&, CellType::Type, uint, uint);

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
