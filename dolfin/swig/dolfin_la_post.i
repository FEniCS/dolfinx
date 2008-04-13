// Instantiate uBlas matrix types (typedefs are ignored)
%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;
%typedef dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix> uBlasSparseMatrix;
%typedef dolfin::uBlasMatrix<dolfin::ublas_dense_matrix> uBlasDenseMatrix;


#ifdef HAS_SLEPC
%extend dolfin::SLEPcEigenvalueSolver{

PyObject* getEigenpair(dolfin::PETScVector& rr, dolfin::PETScVector& cc, const int emode) {
    dolfin::real err, ecc;
    self->getEigenpair(err, ecc, rr, cc, emode);

    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(err));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ecc));
    Py_INCREF(result);
    return result;

}

}
#endif

%extend dolfin::Matrix {
  %pythoncode %{
    def __mul__(self, other):
      v = Vector(self.size(0))
      self.mult(other, v)
      return v

  %}
}

#ifdef HAS_PETSC
%extend dolfin::PETScMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}
#else
%extend dolfin::uBlasMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}
#endif


%extend dolfin::Vector {
  %pythoncode %{

    def __add__(self, v): 
      a = self.copy() 
      a += v
      return a

    def __sub__(self, v): 
      a = self.copy() 
      a -= v
      return a

    def __mul__(self, v): 
      a = self.copy() 
      a *= v
      return a

    def __rmul__(self, v):
      return self.__mul__(v)
  %}
}



#ifdef HAS_PETSC
%extend dolfin::PETScVector {
  %pythoncode %{

    def __add__(self, v): 
      a = self.copy() 
      a += v
      return a

    def __sub__(self, v):
      a = self.copy() 
      a -= v
      return a

    def __mul__(self, v): 
      a = self.copy() 
      a *= v
      return a

    def __rmul__(self, v):
      return self.__mul__(v)
  %}
}
#else
%extend dolfin::uBlasVector {
  %pythoncode %{

    def __add__(self, v):
      a = self.copy() 
      a += v
      return a

    def __sub__(self, v):
      a = self.copy() 
      a -= v
      return a

    def __mul__(self, v): 
      a = self.copy() 
      a *= v
      return a

    def __rmul__(self, v):
      return self.__mul__(v)
  %}
}
#endif



// Initialize tensor type map
%pythoncode %{
_down_cast_map = {}
%}


// Add PETSc support
#ifdef HAS_PETSC

%inline %{
dolfin::PETScVector & down_cast_petsc_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::PETScVector>(); }

dolfin::PETScMatrix & down_cast_petsc_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::PETScMatrix>(); }
%}

%pythoncode %{
_down_cast_map[PETScVector] = down_cast_petsc_vector
_down_cast_map[PETScMatrix] = down_cast_petsc_matrix
%}

#endif

// Add uBlas support
//#ifdef HAS_BOOST

%inline %{
dolfin::uBlasVector & down_cast_ublas_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::uBlasVector>(); }

uBlasSparseMatrix   & down_cast_ublas_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<uBlasSparseMatrix>(); }
%}

%pythoncode %{
_down_cast_map[uBlasVector] = down_cast_ublas_vector
_down_cast_map[uBlasSparseMatrix] = down_cast_ublas_matrix
%}

//#endif


// Dynamic wrapper for down_cast, using dict of tensor types to select from C++ template instantiations
%pythoncode %{
def down_cast(tensor, subclass):
    global _down_cast_map
    assert _down_cast_map
    assert subclass in _down_cast_map
    return _down_cast_map[subclass](tensor)
%}


